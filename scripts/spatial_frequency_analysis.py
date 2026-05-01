"""
空间-频率联合分析：
1. 将 pred-init 残差分解为低频/高频空间分量
2. 可视化高频残差在 triplane 上的空间分布
3. 通过 decoder 渲染低频/高频残差对应的图像差异
4. 结合可见性分析高频残差是否集中在遮挡区域

用法：
    python scripts/spatial_frequency_analysis.py \
        --predinit_path ./exps/gt_triplane/<uid>_triplane_predinit.pt \
        --data_dir ./data/rendered/<uid> \
        --model_name zxhezexin/openlrm-mix-base-1.1 \
        --infer_config ./configs/infer-b.yaml \
        --output_dir ./exps/spatial_freq
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from accelerate import PartialState
PartialState()
torch._dynamo.config.suppress_errors = True

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openlrm.datasets.cam_utils import (
    build_camera_standard,
    build_camera_principle,
    camera_normalization_objaverse,
    create_intrinsics,
)
from openlrm.utils.hf_hub import wrap_model_hub


def load_render_data(data_dir, num_views=32):
    rgba_dir = os.path.join(data_dir, 'rgba')
    pose_dir = os.path.join(data_dir, 'pose')
    intrinsics_path = os.path.join(data_dir, 'intrinsics.npy')
    intrinsics = torch.from_numpy(np.load(intrinsics_path)).float()
    images, poses = [], []
    for i in range(num_views):
        img = np.array(Image.open(os.path.join(rgba_dir, f'{i:03d}.png'))).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        rgb = img_tensor[:3] * img_tensor[3:4] + (1 - img_tensor[3:4])
        images.append(rgb)
        pose = np.load(os.path.join(pose_dir, f'{i:03d}.npy'))
        poses.append(torch.from_numpy(pose).float())
    return torch.stack(images), torch.stack(poses), intrinsics


def freq_split(tensor_2d, cutoff=0.3):
    """将 2D 张量分解为低频和高频分量"""
    H, W = tensor_2d.shape
    fft = torch.fft.fftshift(torch.fft.fft2(tensor_2d))
    cy, cx = H // 2, W // 2
    y = torch.arange(H).float() - cy
    x = torch.arange(W).float() - cx
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    r_norm = torch.sqrt(xx**2 + yy**2) / min(cy, cx)
    low_mask = r_norm < cutoff
    fft_low = fft * low_mask.float()
    fft_high = fft * (~low_mask).float()
    low = torch.fft.ifft2(torch.fft.ifftshift(fft_low)).real
    high = torch.fft.ifft2(torch.fft.ifftshift(fft_high)).real
    return low, high


def split_triplane_freq(triplane, cutoff=0.3):
    """将 triplane (1, 3, D, H, W) 分解为低频和高频"""
    tp = triplane[0]  # (3, D, H, W)
    low = torch.zeros_like(tp)
    high = torch.zeros_like(tp)
    for pi in range(3):
        for ci in range(tp.shape[1]):
            l, h = freq_split(tp[pi, ci], cutoff)
            low[pi, ci] = l
            high[pi, ci] = h
    return low.unsqueeze(0), high.unsqueeze(0)


def render_views(model, triplane, cameras, render_size, device, view_indices):
    triplane = triplane.to(device)
    results = []
    for vi in view_indices:
        cam = cameras[vi:vi+1].to(device)
        with torch.no_grad():
            out = model.synthesizer(
                planes=triplane,
                cameras=cam.unsqueeze(0),
                anchors=torch.zeros(1, 1, 2, device=device),
                resolutions=torch.ones(1, 1, 1, device=device) * render_size,
                bg_colors=torch.ones(1, 1, 1, device=device),
                region_size=render_size,
            )
        results.append(out['images_rgb'].squeeze().cpu())
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predinit_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="zxhezexin/openlrm-mix-base-1.1")
    parser.add_argument("--infer_config", type=str, default="./configs/infer-b.yaml")
    parser.add_argument("--output_dir", type=str, default="./exps/spatial_freq")
    parser.add_argument("--cutoff", type=float, default=0.3)
    parser.add_argument("--num_views", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.infer_config)
    render_size = cfg.render_size

    # 加载数据
    print("[1/5] 加载数据...")
    data = torch.load(args.predinit_path, map_location='cpu')
    t_pred = data['t_pred']
    t_gt = data['t_gt']
    uid = os.path.basename(args.predinit_path).replace('_triplane_predinit.pt', '')
    print(f"  UID: {uid}")

    # 残差
    residual = t_pred - t_gt  # (1, 3, 48, 64, 64)
    print(f"  残差 L1: {residual.abs().mean():.4f}")

    # 频率分解残差
    print("[2/5] 频率分解残差...")
    res_low, res_high = split_triplane_freq(residual, cutoff=args.cutoff)
    print(f"  低频残差 L1: {res_low.abs().mean():.4f}")
    print(f"  高频残差 L1: {res_high.abs().mean():.4f}")
    print(f"  低频占比: {res_low.abs().mean() / residual.abs().mean():.1%}")
    print(f"  高频占比: {res_high.abs().mean() / residual.abs().mean():.1%}")

    # 加载模型
    print("[3/5] 加载模型...")
    from openlrm.models import model_dict
    hf_model_cls = wrap_model_hub(model_dict['lrm'])
    model = hf_model_cls.from_pretrained(args.model_name).to(device)
    model.eval()

    # 加载相机
    print("[4/5] 加载相机...")
    images, poses, intrinsics = load_render_data(args.data_dir)
    normalized_poses = camera_normalization_objaverse('auto', poses)
    intrinsics_batch = intrinsics.unsqueeze(0).repeat(poses.shape[0], 1, 1)
    render_cameras = build_camera_standard(normalized_poses, intrinsics_batch)

    total_views = poses.shape[0]
    step = max(1, total_views // args.num_views)
    view_indices = list(range(0, total_views, step))[:args.num_views]
    print(f"  视角: {view_indices}")

    # 渲染对比
    print("[5/5] 渲染分析...")
    renders_pred = render_views(model, t_pred, render_cameras, render_size, device, view_indices)
    renders_gt = render_views(model, t_gt, render_cameras, render_size, device, view_indices)

    # 渲染 T_pred + 低频修正 和 T_pred + 高频修正
    t_pred_lowfix = t_pred - res_low  # 修正低频 → T_pred 的低频被替换为 T_GT 的低频
    t_pred_highfix = t_pred - res_high  # 修正高频
    renders_lowfix = render_views(model, t_pred_lowfix, render_cameras, render_size, device, view_indices)
    renders_highfix = render_views(model, t_pred_highfix, render_cameras, render_size, device, view_indices)

    # 统计
    l1_pred = np.mean([(rp - rg).abs().mean().item() for rp, rg in zip(renders_pred, renders_gt)])
    l1_lowfix = np.mean([(rp - rg).abs().mean().item() for rp, rg in zip(renders_lowfix, renders_gt)])
    l1_highfix = np.mean([(rp - rg).abs().mean().item() for rp, rg in zip(renders_highfix, renders_gt)])

    print(f"\n  渲染 L1 对比:")
    print(f"    T_pred (baseline):   {l1_pred:.6f}")
    print(f"    修正低频 (fix low):  {l1_lowfix:.6f} ({(1-l1_lowfix/l1_pred)*100:+.1f}%)")
    print(f"    修正高频 (fix high): {l1_highfix:.6f} ({(1-l1_highfix/l1_pred)*100:+.1f}%)")

    os.makedirs(args.output_dir, exist_ok=True)

    # ─── 图 1：Triplane 残差空间分布 ───
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(4, 6, figure=fig, hspace=0.35, wspace=0.3)

    plane_names = ['XY', 'XZ', 'YZ']
    res_np = residual[0].numpy()  # (3, 48, 64, 64)
    res_low_np = res_low[0].numpy()
    res_high_np = res_high[0].numpy()

    # 行1: 每个 plane 的平均残差幅度 (channel-averaged)
    for pi in range(3):
        full_map = np.mean(np.abs(res_np[pi]), axis=0)  # (64, 64)
        low_map = np.mean(np.abs(res_low_np[pi]), axis=0)
        high_map = np.mean(np.abs(res_high_np[pi]), axis=0)

        vmax = max(full_map.max(), low_map.max(), high_map.max())

        ax = fig.add_subplot(gs[0, pi*2])
        im = ax.imshow(low_map, cmap='hot', vmin=0, vmax=vmax)
        ax.set_title(f'{plane_names[pi]} Low-Freq |Residual|', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = fig.add_subplot(gs[0, pi*2+1])
        im = ax.imshow(high_map, cmap='hot', vmin=0, vmax=vmax)
        ax.set_title(f'{plane_names[pi]} High-Freq |Residual|', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # 行2: 高频/低频比值空间图
    for pi in range(3):
        low_map = np.mean(np.abs(res_low_np[pi]), axis=0)
        high_map = np.mean(np.abs(res_high_np[pi]), axis=0)
        ratio_map = high_map / (low_map + 1e-6)

        ax = fig.add_subplot(gs[1, pi*2:pi*2+2])
        im = ax.imshow(ratio_map, cmap='RdBu_r', vmin=0, vmax=ratio_map.mean() * 3)
        ax.set_title(f'{plane_names[pi]} High/Low Ratio Map', fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # 行3: 渲染对比 (选2个视角)
    sample_views = [0, len(view_indices)//2]
    for si, svi in enumerate(sample_views):
        ax = fig.add_subplot(gs[2, si*3:si*3+3])
        imgs = [
            renders_pred[svi].permute(1,2,0).clamp(0,1).numpy(),
            renders_lowfix[svi].permute(1,2,0).clamp(0,1).numpy(),
            renders_highfix[svi].permute(1,2,0).clamp(0,1).numpy(),
            renders_gt[svi].permute(1,2,0).clamp(0,1).numpy(),
        ]
        combined = np.concatenate(imgs, axis=1)
        ax.imshow(combined)
        w = imgs[0].shape[1]
        labels = ['Pred', 'Fix-Low', 'Fix-High', 'GT']
        for li, label in enumerate(labels):
            ax.text(li*w + 3, 12, label, color='yellow', fontsize=9, fontweight='bold')
        ax.set_title(f'View {view_indices[svi]}', fontsize=11)
        ax.axis('off')

    # 行4: 渲染误差图 (同2个视角)
    for si, svi in enumerate(sample_views):
        ax = fig.add_subplot(gs[3, si*3:si*3+3])
        err_pred = (renders_pred[svi] - renders_gt[svi]).abs().mean(0).numpy()
        err_lowfix = (renders_lowfix[svi] - renders_gt[svi]).abs().mean(0).numpy()
        err_highfix = (renders_highfix[svi] - renders_gt[svi]).abs().mean(0).numpy()

        vmax_err = max(err_pred.max(), err_lowfix.max(), err_highfix.max())
        errors = [err_pred, err_lowfix, err_highfix]
        combined_err = np.concatenate(errors, axis=1)
        im = ax.imshow(combined_err, cmap='hot', vmin=0, vmax=vmax_err)
        w = err_pred.shape[1]
        labels = ['Pred Error', 'Fix-Low Error', 'Fix-High Error']
        for li, label in enumerate(labels):
            ax.text(li*w + 3, 12, label, color='cyan', fontsize=9, fontweight='bold')
        ax.set_title(f'View {view_indices[svi]} Error Maps', fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.02)

    plt.suptitle(
        f'Spatial-Frequency Analysis (pred-init)\n{uid}\n'
        f'Pred L1={l1_pred:.4f} | Fix-Low L1={l1_lowfix:.4f} ({(1-l1_lowfix/l1_pred)*100:+.1f}%) | '
        f'Fix-High L1={l1_highfix:.4f} ({(1-l1_highfix/l1_pred)*100:+.1f}%)',
        fontsize=14, fontweight='bold'
    )
    save_path = os.path.join(args.output_dir, f'{uid}_spatial_freq.png')
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\n  保存: {save_path}")

    # ─── 图 2：逐视角修正效果 ───
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(view_indices))
    w = 0.25
    l1_per_pred = [(rp - rg).abs().mean().item() for rp, rg in zip(renders_pred, renders_gt)]
    l1_per_low = [(rp - rg).abs().mean().item() for rp, rg in zip(renders_lowfix, renders_gt)]
    l1_per_high = [(rp - rg).abs().mean().item() for rp, rg in zip(renders_highfix, renders_gt)]

    ax.bar(x - w, l1_per_pred, w, label='Pred (baseline)', color='#F44336', alpha=0.8)
    ax.bar(x, l1_per_low, w, label='Fix Low-Freq', color='#2196F3', alpha=0.8)
    ax.bar(x + w, l1_per_high, w, label='Fix High-Freq', color='#4CAF50', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'V{vi}' for vi in view_indices])
    ax.set_ylabel('L1 Error vs T_GT render')
    ax.set_title(f'Per-View Correction Effect ({uid[:8]}...)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    save_path2 = os.path.join(args.output_dir, f'{uid}_correction_effect.png')
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path2}")

    # ─── 源视角 vs 其他视角的修正效果差异 ───
    print(f"\n  源视角 (V0) vs 其他视角的修正效果:")
    print(f"    V0  lowfix: {l1_per_low[0]:.5f} ({(1-l1_per_low[0]/l1_per_pred[0])*100:+.1f}%)")
    print(f"    V0  highfix: {l1_per_high[0]:.5f} ({(1-l1_per_high[0]/l1_per_pred[0])*100:+.1f}%)")
    other_low = np.mean(l1_per_low[1:])
    other_high = np.mean(l1_per_high[1:])
    other_pred = np.mean(l1_per_pred[1:])
    print(f"    Others lowfix: {other_low:.5f} ({(1-other_low/other_pred)*100:+.1f}%)")
    print(f"    Others highfix: {other_high:.5f} ({(1-other_high/other_pred)*100:+.1f}%)")

    print("\n  完成！")


if __name__ == "__main__":
    main()
