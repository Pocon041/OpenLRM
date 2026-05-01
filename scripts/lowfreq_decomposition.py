"""
低频残差精细分解 + Decoder Null Space 维度估计

实验 A：将低频残差进一步分解为 DC（均值偏移）和非 DC 低频分量，
       测量各自修正后的渲染改善。

实验 B：在 T_GT 上施加不同方向的随机扰动，统计渲染变化，
       估计 decoder 有效维度（非 null space 的维度）。

用法：
    python scripts/lowfreq_decomposition.py \
        --predinit_path ./exps/gt_triplane/<uid>_triplane_predinit.pt \
        --data_dir ./data/rendered/<uid> \
        --output_dir ./exps/lowfreq_decomp
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
from accelerate import PartialState
PartialState()
torch._dynamo.config.suppress_errors = True

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openlrm.datasets.cam_utils import (
    build_camera_standard,
    camera_normalization_objaverse,
)
from openlrm.utils.hf_hub import wrap_model_hub


def load_render_data(data_dir, num_views=32):
    rgba_dir = os.path.join(data_dir, 'rgba')
    pose_dir = os.path.join(data_dir, 'pose')
    intrinsics = torch.from_numpy(np.load(os.path.join(data_dir, 'intrinsics.npy'))).float()
    images, poses = [], []
    for i in range(num_views):
        img = np.array(Image.open(os.path.join(rgba_dir, f'{i:03d}.png'))).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        rgb = img_tensor[:3] * img_tensor[3:4] + (1 - img_tensor[3:4])
        images.append(rgb)
        poses.append(torch.from_numpy(np.load(os.path.join(pose_dir, f'{i:03d}.npy'))).float())
    return torch.stack(images), torch.stack(poses), intrinsics


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


def mean_l1(renders_a, renders_b):
    return np.mean([(a - b).abs().mean().item() for a, b in zip(renders_a, renders_b)])


def freq_split(tensor_2d, cutoff):
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


def decompose_residual(residual, cutoff=0.3):
    """将残差 (1,3,D,H,W) 分解为 DC、非DC低频、高频三个分量"""
    tp = residual[0]  # (3, D, H, W)

    dc = torch.zeros_like(tp)
    low_nodc = torch.zeros_like(tp)
    high = torch.zeros_like(tp)

    for pi in range(3):
        for ci in range(tp.shape[1]):
            ch = tp[pi, ci]  # (H, W)
            ch_mean = ch.mean()
            dc[pi, ci] = ch_mean  # DC = 常数平面
            ch_centered = ch - ch_mean

            l, h = freq_split(ch_centered, cutoff)
            low_nodc[pi, ci] = l
            high[pi, ci] = h

    return dc.unsqueeze(0), low_nodc.unsqueeze(0), high.unsqueeze(0)


def experiment_a(model, t_pred, t_gt, residual, cameras, render_size, device, view_indices, cutoff=0.3):
    """实验 A：DC vs 结构化低频 vs 高频修正效果"""
    print("\n" + "="*60)
    print("  实验 A：低频残差精细分解")
    print("="*60)

    dc, low_nodc, high = decompose_residual(residual, cutoff)

    # 各分量的 L1
    print(f"  DC 分量 L1:       {dc.abs().mean():.4f}")
    print(f"  非DC低频 L1:      {low_nodc.abs().mean():.4f}")
    print(f"  高频 L1:          {high.abs().mean():.4f}")
    print(f"  总残差 L1:        {residual.abs().mean():.4f}")

    # 渲染 baseline
    renders_pred = render_views(model, t_pred, cameras, render_size, device, view_indices)
    renders_gt = render_views(model, t_gt, cameras, render_size, device, view_indices)
    l1_baseline = mean_l1(renders_pred, renders_gt)

    # 各种修正
    corrections = {
        'Fix DC only': t_pred - dc,
        'Fix non-DC low': t_pred - low_nodc,
        'Fix DC + non-DC low': t_pred - dc - low_nodc,
        'Fix high only': t_pred - high,
        'Fix all (=T_GT)': t_gt,
    }

    results = {'Pred (baseline)': l1_baseline}
    print(f"\n  渲染 L1 对比:")
    print(f"    {'方法':<25} {'L1':>10} {'改善':>10}")
    print(f"    {'':-<25} {'':-<10} {'':-<10}")
    print(f"    {'Pred (baseline)':<25} {l1_baseline:>10.6f} {'':>10}")

    for name, tp_corrected in corrections.items():
        renders = render_views(model, tp_corrected, cameras, render_size, device, view_indices)
        l1 = mean_l1(renders, renders_gt)
        improvement = (1 - l1 / l1_baseline) * 100
        results[name] = l1
        print(f"    {name:<25} {l1:>10.6f} {improvement:>+9.1f}%")

    return results, renders_pred, renders_gt


def experiment_b(model, t_gt, cameras, render_size, device, view_indices,
                 n_directions=100, epsilon=0.1):
    """实验 B：Decoder null space 维度估计"""
    print("\n" + "="*60)
    print("  实验 B：Decoder Null Space 维度估计")
    print("="*60)

    renders_gt = render_views(model, t_gt, cameras, render_size, device, view_indices)

    total_dim = t_gt.numel()
    print(f"  Triplane 总维度: {total_dim}")
    print(f"  扰动幅度 ε: {epsilon}")
    print(f"  随机方向数: {n_directions}")

    sensitivities = []
    for i in range(n_directions):
        # 随机单位方向
        direction = torch.randn_like(t_gt)
        direction = direction / direction.norm() * epsilon

        t_perturbed = t_gt + direction
        renders_p = render_views(model, t_perturbed, cameras, render_size, device, view_indices)
        delta = mean_l1(renders_p, renders_gt)
        sensitivities.append(delta)

        if (i + 1) % 20 == 0:
            print(f"    方向 {i+1}/{n_directions}: 平均 Δ={np.mean(sensitivities):.6f}")

    sensitivities = np.array(sensitivities)

    # 统计
    print(f"\n  Δ_render 统计:")
    print(f"    Mean:   {sensitivities.mean():.6f}")
    print(f"    Std:    {sensitivities.std():.6f}")
    print(f"    Min:    {sensitivities.min():.6f}")
    print(f"    Max:    {sensitivities.max():.6f}")
    print(f"    Median: {np.median(sensitivities):.6f}")

    # 如果 null space 很大，大部分随机方向的 Δ 都很小
    # 用不同阈值统计 "无效方向" 的比例
    mean_delta = sensitivities.mean()
    thresholds = [0.1, 0.2, 0.5, 1.0]
    print(f"\n  方向敏感度分布 (相对于 mean Δ):")
    for t in thresholds:
        frac = (sensitivities < mean_delta * t).mean()
        print(f"    Δ < {t:.0%} × mean: {frac:.1%} 的方向")

    return sensitivities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predinit_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="zxhezexin/openlrm-mix-base-1.1")
    parser.add_argument("--infer_config", type=str, default="./configs/infer-b.yaml")
    parser.add_argument("--output_dir", type=str, default="./exps/lowfreq_decomp")
    parser.add_argument("--cutoff", type=float, default=0.3)
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--null_directions", type=int, default=100)
    parser.add_argument("--null_epsilon", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.infer_config)
    render_size = cfg.render_size

    # 加载数据
    print("[1/4] 加载数据...")
    data = torch.load(args.predinit_path, map_location='cpu')
    t_pred = data['t_pred']
    t_gt = data['t_gt']
    uid = os.path.basename(args.predinit_path).replace('_triplane_predinit.pt', '')
    residual = t_pred - t_gt

    # 加载模型
    print("[2/4] 加载模型...")
    from openlrm.models import model_dict
    hf_model_cls = wrap_model_hub(model_dict['lrm'])
    model = hf_model_cls.from_pretrained(args.model_name).to(device)
    model.eval()

    # 加载相机
    print("[3/4] 加载相机...")
    images, poses, intrinsics = load_render_data(args.data_dir)
    normalized_poses = camera_normalization_objaverse('auto', poses)
    intrinsics_batch = intrinsics.unsqueeze(0).repeat(poses.shape[0], 1, 1)
    render_cameras = build_camera_standard(normalized_poses, intrinsics_batch)
    total_views = poses.shape[0]
    step = max(1, total_views // args.num_views)
    view_indices = list(range(0, total_views, step))[:args.num_views]

    print("[4/4] 运行实验...")

    # 实验 A
    results_a, renders_pred, renders_gt = experiment_a(
        model, t_pred, t_gt, residual, render_cameras,
        render_size, device, view_indices, args.cutoff
    )

    # 实验 B
    sensitivities = experiment_b(
        model, t_gt, render_cameras, render_size, device, view_indices,
        n_directions=args.null_directions, epsilon=args.null_epsilon,
    )

    # ─── 保存图表 ───
    os.makedirs(args.output_dir, exist_ok=True)

    # 图 1：修正效果柱状图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    names = list(results_a.keys())
    values = [results_a[n] for n in names]
    colors = ['#F44336', '#FF9800', '#FFC107', '#2196F3', '#4CAF50', '#009688']
    ax = axes[0]
    bars = ax.bar(range(len(names)), values, color=colors[:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('L1 Error vs T_GT render')
    ax.set_title(f'Correction Decomposition ({uid[:8]}...)')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 图 2：Null space 敏感度分布
    ax = axes[1]
    ax.hist(sensitivities, bins=30, color='#2196F3', alpha=0.7, edgecolor='white')
    ax.axvline(sensitivities.mean(), color='red', linestyle='--', label=f'Mean={sensitivities.mean():.5f}')
    ax.axvline(np.median(sensitivities), color='orange', linestyle='--', label=f'Median={np.median(sensitivities):.5f}')
    ax.set_xlabel('Δ_render (L1 change per random direction)')
    ax.set_ylabel('Count')
    ax.set_title(f'Decoder Sensitivity Distribution (ε={args.null_epsilon})')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(f'Low-Freq Decomposition & Null Space Analysis\n{uid}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(args.output_dir, f'{uid}_lowfreq_decomp.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  保存: {save_path}")

    print("\n  完成！")


if __name__ == "__main__":
    main()
