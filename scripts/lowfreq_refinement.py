"""
低频专注的测试时优化 (Low-Freq Test-Time Refinement)

核心思想：encoder 已在好的 basin（r≈0.83），渲染误差主要来自 triplane 低频偏差。
仅优化 triplane 的低频分量，冻结高频，应更快收敛且效果接近全量优化。

对比方案：
  1. T_pred（baseline，无优化）
  2. 全量优化 N 步（baseline optimization）
  3. 仅低频优化 N 步（proposed）
  4. T_GT（upper bound）

可视化输出：
  - 多视角渲染对比（Pred / LowFreq-Refine / Full-Refine / GT）
  - 收敛曲线对比
  - 误差热力图对比

用法：
    python scripts/lowfreq_refinement.py \
        --data_dir ./data/rendered/<uid> \
        --model_name zxhezexin/openlrm-mix-base-1.1 \
        --output_dir ./exps/lowfreq_refine
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
from tqdm import tqdm
from accelerate import PartialState
PartialState()
torch._dynamo.config.suppress_errors = True

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openlrm.datasets.cam_utils import (
    build_camera_standard,
    build_camera_principle,
    create_intrinsics,
    camera_normalization_objaverse,
)
from openlrm.utils.hf_hub import wrap_model_hub


def get_predicted_triplane(model, source_image, source_cam_dist, source_size, device):
    image = source_image.unsqueeze(0).to(device)
    image = F.interpolate(image, size=(source_size, source_size), mode='bicubic', align_corners=True)
    image = torch.clamp(image, 0, 1)
    canonical_camera_extrinsics = torch.tensor([[
        [1, 0, 0, 0],
        [0, 0, -1, -source_cam_dist],
        [0, 1, 0, 0],
    ]], dtype=torch.float32, device=device)
    canonical_camera_intrinsics = create_intrinsics(f=0.75, c=0.5, device=device).unsqueeze(0)
    source_camera = build_camera_principle(canonical_camera_extrinsics, canonical_camera_intrinsics)
    with torch.no_grad():
        planes = model.forward_planes(image, source_camera)
    return planes


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


def render_single(model, triplane, camera, render_size, device):
    triplane = triplane.to(device)
    camera = camera.to(device)
    with torch.no_grad():
        out = model.synthesizer(
            planes=triplane,
            cameras=camera.unsqueeze(0),
            anchors=torch.zeros(1, 1, 2, device=device),
            resolutions=torch.ones(1, 1, 1, device=device) * render_size,
            bg_colors=torch.ones(1, 1, 1, device=device),
            region_size=render_size,
        )
    return out['images_rgb'].squeeze().cpu()


def render_for_loss(model, triplane, camera, render_size, device):
    """渲染用于 loss 计算，保留梯度"""
    camera = camera.to(device)
    out = model.synthesizer(
        planes=triplane,
        cameras=camera.unsqueeze(0),
        anchors=torch.zeros(1, 1, 2, device=device),
        resolutions=torch.ones(1, 1, 1, device=device) * render_size,
        bg_colors=torch.ones(1, 1, 1, device=device),
        region_size=render_size,
    )
    return out['images_rgb']


def build_freq_masks(H, W, cutoff=0.3):
    """构建低频/高频掩码"""
    cy, cx = H // 2, W // 2
    y = torch.arange(H).float() - cy
    x = torch.arange(W).float() - cx
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    r_norm = torch.sqrt(xx**2 + yy**2) / min(cy, cx)
    low_mask = (r_norm < cutoff).float()
    high_mask = 1.0 - low_mask
    return low_mask, high_mask


class LowFreqTriplane(torch.nn.Module):
    """可学习的低频 triplane 参数化：仅优化低频分量，高频冻结"""

    def __init__(self, triplane_init, cutoff=0.3):
        super().__init__()
        # triplane_init: (1, 3, D, H, W)
        self.register_buffer('triplane_init', triplane_init.clone())
        H, W = triplane_init.shape[-2], triplane_init.shape[-1]

        low_mask, high_mask = build_freq_masks(H, W, cutoff)
        self.register_buffer('low_mask', low_mask)
        self.register_buffer('high_mask', high_mask)

        # 提取初始低频分量作为可学习参数
        # delta 从零开始，表示对低频的修正量
        self.low_delta = torch.nn.Parameter(
            torch.zeros_like(triplane_init)
        )

    def forward(self):
        # 对 delta 做低频投影：只保留低频
        B, P, D, H, W = self.triplane_init.shape
        delta = self.low_delta
        delta_projected = torch.zeros_like(delta)
        for pi in range(P):
            for ci in range(D):
                fft = torch.fft.fftshift(torch.fft.fft2(delta[0, pi, ci]))
                fft_low = fft * self.low_mask
                delta_projected[0, pi, ci] = torch.fft.ifft2(
                    torch.fft.ifftshift(fft_low)
                ).real

        return self.triplane_init + delta_projected


def triplane_proximity_loss(triplane, triplane_ref):
    """Keep triplane close to reference (encoder prediction)"""
    return (triplane - triplane_ref).pow(2).mean()


def triplane_tv_loss(triplane):
    """Total Variation loss on triplane to encourage spatial smoothness"""
    # triplane: (1, 3, D, H, W)
    tv_h = (triplane[:, :, :, 1:, :] - triplane[:, :, :, :-1, :]).abs().mean()
    tv_w = (triplane[:, :, :, :, 1:] - triplane[:, :, :, :, :-1]).abs().mean()
    return tv_h + tv_w


def optimize_triplane(model, triplane_init, images, cameras, render_size, device,
                      num_iters=200, lr=0.01, mode='full', cutoff=0.3,
                      eval_cameras=None, eval_images=None, eval_every=10,
                      lambda_tv=0.0, lambda_opacity=0.0, lambda_prox=0.0):
    """
    优化 triplane
    mode: 'full' = 全量优化, 'lowfreq' = 仅低频优化
    lambda_tv: TV 正则化权重
    lambda_opacity: opacity sparsity 正则化权重
    lambda_prox: proximity prior 权重（保持接近 T_pred）
    """
    triplane_ref = triplane_init.clone().to(device)  # anchor
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if mode == 'lowfreq':
        tp_module = LowFreqTriplane(triplane_init.clone(), cutoff).to(device)
        optimizer = torch.optim.Adam(tp_module.parameters(), lr=lr)
    else:
        tp_param = torch.nn.Parameter(triplane_init.clone().to(device))
        optimizer = torch.optim.Adam([tp_param], lr=lr)

    num_train_views = images.shape[0]
    history = {'iter': [], 'train_loss': [], 'eval_l1': []}

    pbar = tqdm(range(num_iters), desc=f'Optimize ({mode})')
    for it in pbar:
        optimizer.zero_grad()

        vi = it % num_train_views
        gt_img = images[vi:vi+1].to(device)
        gt_img = F.interpolate(gt_img, size=(render_size, render_size), mode='bilinear', align_corners=False)
        cam = cameras[vi:vi+1].to(device)

        if mode == 'lowfreq':
            triplane = tp_module()
        else:
            triplane = tp_param

        out = model.synthesizer(
            planes=triplane,
            cameras=cam.unsqueeze(0),
            anchors=torch.zeros(1, 1, 2, device=device),
            resolutions=torch.ones(1, 1, 1, device=device) * render_size,
            bg_colors=torch.ones(1, 1, 1, device=device),
            region_size=render_size,
        )
        pred_img = out['images_rgb'].squeeze(0)
        loss_rgb = F.mse_loss(pred_img, gt_img)

        # Regularization
        loss_reg = torch.tensor(0.0, device=device)
        if lambda_tv > 0:
            loss_reg = loss_reg + lambda_tv * triplane_tv_loss(triplane)
        if lambda_opacity > 0:
            weights = out['images_weight'].squeeze(0)  # (1, 1, H, W)
            loss_reg = loss_reg + lambda_opacity * (weights ** 2).mean()
        if lambda_prox > 0:
            loss_reg = loss_reg + lambda_prox * triplane_proximity_loss(triplane, triplane_ref)

        loss = loss_rgb + loss_reg
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=f'{loss_rgb.item():.5f}', reg=f'{loss_reg.item():.5f}')

        if it % eval_every == 0 or it == num_iters - 1:
            if mode == 'lowfreq':
                tp_eval = tp_module().detach()
            else:
                tp_eval = tp_param.detach()

            if eval_cameras is not None:
                eval_l1s = []
                for evi in range(len(eval_cameras)):
                    r = render_single(model, tp_eval, eval_cameras[evi:evi+1], render_size, device)
                    gt = eval_images[evi]
                    gt_resized = F.interpolate(gt.unsqueeze(0), size=(render_size, render_size),
                                               mode='bilinear', align_corners=False).squeeze(0)
                    eval_l1s.append((r - gt_resized).abs().mean().item())
                eval_l1 = np.mean(eval_l1s)
            else:
                eval_l1 = loss.item()

            history['iter'].append(it)
            history['train_loss'].append(loss.item())
            history['eval_l1'].append(eval_l1)

    if mode == 'lowfreq':
        final_tp = tp_module().detach().cpu()
    else:
        final_tp = tp_param.detach().cpu()

    return final_tp, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="zxhezexin/openlrm-mix-base-1.1")
    parser.add_argument("--infer_config", type=str, default="./configs/infer-b.yaml")
    parser.add_argument("--output_dir", type=str, default="./exps/lowfreq_refine")
    parser.add_argument("--num_iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--cutoff", type=float, default=0.3)
    parser.add_argument("--source_view", type=int, default=0)
    parser.add_argument("--num_train_views", type=int, default=8,
                        help="用于优化的视角数（模拟少量参考视角）")
    parser.add_argument("--lambda_tv", type=float, default=1e-4,
                        help="TV 正则化权重")
    parser.add_argument("--lambda_opacity", type=float, default=0.0,
                        help="Opacity sparsity 正则化权重")
    parser.add_argument("--lambda_prox", type=float, default=1.0,
                        help="Proximity prior 权重（保持接近 T_pred）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.infer_config)
    render_size = cfg.render_size

    uid = os.path.basename(args.data_dir.rstrip('/'))
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print("[1/5] 加载模型...")
    from openlrm.models import model_dict
    hf_model_cls = wrap_model_hub(model_dict['lrm'])
    model = hf_model_cls.from_pretrained(args.model_name).to(device)
    model.eval()

    # 加载数据
    print("[2/5] 加载渲染数据...")
    images, poses, intrinsics = load_render_data(args.data_dir)
    normalized_poses = camera_normalization_objaverse('auto', poses)
    intrinsics_batch = intrinsics.unsqueeze(0).repeat(poses.shape[0], 1, 1)
    render_cameras = build_camera_standard(normalized_poses, intrinsics_batch)

    # 获取 T_pred
    print("[3/5] 获取 T_pred...")
    t_pred = get_predicted_triplane(
        model, images[args.source_view],
        source_cam_dist=cfg.source_cam_dist,
        source_size=cfg.source_size,
        device=device,
    )
    t_pred_cpu = t_pred.detach().cpu()
    print(f"  T_pred shape: {t_pred_cpu.shape}")

    # 选择训练/评估视角
    total = poses.shape[0]
    train_step = max(1, total // args.num_train_views)
    train_views = list(range(0, total, train_step))[:args.num_train_views]
    # 评估视角：在训练视角之间均匀插入
    all_views = set(range(total))
    remaining = sorted(all_views - set(train_views))
    eval_step = max(1, len(remaining) // 8)
    eval_views = remaining[::eval_step][:8]
    if len(eval_views) == 0:
        eval_views = [v for v in [2, 6, 10, 14, 18, 22, 26, 30] if v < total and v not in train_views][:8]
    if len(eval_views) == 0:
        eval_views = [1, 3, 5, 7][:min(4, total)]
    print(f"  训练视角: {train_views}")
    print(f"  评估视角: {eval_views}")

    train_images = images[train_views]
    train_cameras = render_cameras[train_views]
    eval_images_list = [images[i] for i in eval_views]
    eval_cameras = render_cameras[eval_views]

    # 全量优化
    print(f"[4/5] 全量优化 ({args.num_iters} iters, prox={args.lambda_prox})...")
    tp_full, hist_full = optimize_triplane(
        model, t_pred_cpu, train_images, train_cameras, render_size, device,
        num_iters=args.num_iters, lr=args.lr, mode='full',
        eval_cameras=eval_cameras, eval_images=eval_images_list,
        lambda_tv=args.lambda_tv, lambda_opacity=args.lambda_opacity,
        lambda_prox=args.lambda_prox,
    )

    # 低频优化
    print(f"[5/5] 低频优化 ({args.num_iters} iters, cutoff={args.cutoff}, prox={args.lambda_prox})...")
    tp_lowfreq, hist_lowfreq = optimize_triplane(
        model, t_pred_cpu, train_images, train_cameras, render_size, device,
        num_iters=args.num_iters, lr=args.lr, mode='lowfreq', cutoff=args.cutoff,
        eval_cameras=eval_cameras, eval_images=eval_images_list,
        lambda_tv=args.lambda_tv, lambda_opacity=args.lambda_opacity,
        lambda_prox=args.lambda_prox,
    )

    # ─── 渲染所有方案 ───
    print("\n  渲染对比...")
    vis_views = eval_views[:6]
    all_renders = {
        'Pred': [], 'LowFreq-Refine': [], 'Full-Refine': [], 'GT': []
    }
    for vi in vis_views:
        cam = render_cameras[vi:vi+1]
        all_renders['Pred'].append(render_single(model, t_pred_cpu, cam, render_size, device))
        all_renders['LowFreq-Refine'].append(render_single(model, tp_lowfreq, cam, render_size, device))
        all_renders['Full-Refine'].append(render_single(model, tp_full, cam, render_size, device))
        gt_resized = F.interpolate(images[vi].unsqueeze(0), size=(render_size, render_size),
                                   mode='bilinear', align_corners=False).squeeze(0)
        all_renders['GT'].append(gt_resized)

    # 统计
    methods = ['Pred', 'LowFreq-Refine', 'Full-Refine']
    print(f"\n  {'方法':<20} {'Eval L1':>10}")
    print(f"  {'':-<20} {'':-<10}")
    for m in methods:
        l1 = np.mean([(r - g).abs().mean().item()
                       for r, g in zip(all_renders[m], all_renders['GT'])])
        print(f"  {m:<20} {l1:>10.6f}")

    # ─── 图 1：大图可视化 ───
    n_views = len(vis_views)
    fig = plt.figure(figsize=(4 * n_views, 16))
    gs = GridSpec(4, n_views, figure=fig, hspace=0.05, wspace=0.05)

    method_names = ['Pred (Encoder)', 'LowFreq-Refine (Ours)', 'Full-Refine', 'GT']
    method_keys = ['Pred', 'LowFreq-Refine', 'Full-Refine', 'GT']

    for row, (mname, mkey) in enumerate(zip(method_names, method_keys)):
        for col in range(n_views):
            ax = fig.add_subplot(gs[row, col])
            img = all_renders[mkey][col]
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0)
            ax.imshow(img.clamp(0, 1).numpy())
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(mname, fontsize=12, fontweight='bold', rotation=90, labelpad=10)
            if row == 0:
                ax.set_title(f'View {vis_views[col]}', fontsize=11)

    plt.suptitle(f'Low-Frequency Refinement Results\n{uid}',
                 fontsize=14, fontweight='bold', y=1.01)
    save_path1 = os.path.join(args.output_dir, f'{uid}_render_comparison.png')
    plt.savefig(save_path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  保存: {save_path1}")

    # ─── 图 2：误差热力图 ───
    fig = plt.figure(figsize=(4 * n_views, 12))
    gs = GridSpec(3, n_views, figure=fig, hspace=0.08, wspace=0.05)

    err_methods = ['Pred', 'LowFreq-Refine', 'Full-Refine']
    err_labels = ['Pred Error', 'LowFreq-Refine Error', 'Full-Refine Error']

    # 找全局 vmax
    all_errs = []
    for m in err_methods:
        for col in range(n_views):
            gt = all_renders['GT'][col]
            pred = all_renders[m][col]
            if gt.shape[0] == 3:
                err = (pred - gt).abs().mean(0).numpy()
            else:
                err = (pred - gt).abs().mean(-1).numpy()
            all_errs.append(err)
    vmax = np.percentile(np.concatenate([e.flatten() for e in all_errs]), 95)

    idx = 0
    for row, (m, label) in enumerate(zip(err_methods, err_labels)):
        for col in range(n_views):
            ax = fig.add_subplot(gs[row, col])
            err = all_errs[idx]
            im = ax.imshow(err, cmap='hot', vmin=0, vmax=vmax)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(label, fontsize=11, fontweight='bold', rotation=90, labelpad=10)
            if row == 0:
                ax.set_title(f'View {vis_views[col]}', fontsize=11)
            idx += 1

    plt.colorbar(im, ax=fig.axes, fraction=0.02, pad=0.02, label='L1 Error')
    plt.suptitle(f'Error Maps Comparison\n{uid}', fontsize=14, fontweight='bold')
    save_path2 = os.path.join(args.output_dir, f'{uid}_error_maps.png')
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path2}")

    # ─── 图 3：收敛曲线对比 ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(hist_full['iter'], hist_full['train_loss'], 'b-', label='Full Optimize', linewidth=2)
    ax.plot(hist_lowfreq['iter'], hist_lowfreq['train_loss'], 'r-', label='LowFreq Only', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Train Loss (MSE)')
    ax.set_title('Training Loss Convergence')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')

    ax = axes[1]
    ax.plot(hist_full['iter'], hist_full['eval_l1'], 'b-', label='Full Optimize', linewidth=2)
    ax.plot(hist_lowfreq['iter'], hist_lowfreq['eval_l1'], 'r-', label='LowFreq Only', linewidth=2)
    # baseline
    pred_eval_l1 = np.mean([(r - g).abs().mean().item()
                             for r, g in zip(all_renders['Pred'], all_renders['GT'])])
    ax.axhline(pred_eval_l1, color='gray', linestyle='--', alpha=0.7, label=f'Pred baseline ({pred_eval_l1:.4f})')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Eval L1 (novel views)')
    ax.set_title('Novel View Quality Convergence')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(f'Convergence: Full vs LowFreq-Only Optimization\n{uid}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path3 = os.path.join(args.output_dir, f'{uid}_convergence.png')
    plt.savefig(save_path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path3}")

    # ─── 图 4：Summary figure ───
    fig = plt.figure(figsize=(20, 5))
    gs = GridSpec(1, n_views + 1, figure=fig, wspace=0.08,
                  width_ratios=[1]*n_views + [0.8])

    # 选一个视角展示 4 个方案
    best_view = 0
    for col, (mname, mkey) in enumerate(zip(
        ['Input View', 'Pred', 'LF-Refine (Ours)', 'Full-Refine', 'GT', 'Pred Err', 'LF-Refine Err'],
        ['source', 'Pred', 'LowFreq-Refine', 'Full-Refine', 'GT', 'pred_err', 'lf_err']
    )):
        if col >= n_views:
            break
        ax = fig.add_subplot(gs[0, col])
        if mkey == 'source':
            img = images[args.source_view].permute(1, 2, 0).clamp(0, 1).numpy()
            ax.imshow(img)
        elif mkey in ('pred_err', 'lf_err'):
            ref = 'Pred' if mkey == 'pred_err' else 'LowFreq-Refine'
            gt = all_renders['GT'][best_view]
            pred = all_renders[ref][best_view]
            if gt.shape[0] == 3:
                err = (pred - gt).abs().mean(0).numpy()
            else:
                err = (pred - gt).abs().mean(-1).numpy()
            ax.imshow(err, cmap='hot', vmin=0, vmax=vmax)
        else:
            img = all_renders[mkey][best_view]
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0)
            ax.imshow(img.clamp(0, 1).numpy())
        ax.set_title(mname, fontsize=11, fontweight='bold')
        ax.axis('off')

    # 最后一列：数值对比
    ax = fig.add_subplot(gs[0, -1])
    ax.axis('off')
    l1s = {}
    for m in methods:
        l1s[m] = np.mean([(r - g).abs().mean().item()
                           for r, g in zip(all_renders[m], all_renders['GT'])])
    text = f"Novel View L1:\n\n"
    text += f"Pred:           {l1s['Pred']:.4f}\n"
    text += f"LF-Refine:   {l1s['LowFreq-Refine']:.4f}\n"
    text += f"Full-Refine: {l1s['Full-Refine']:.4f}\n\n"
    imp_lf = (1 - l1s['LowFreq-Refine'] / l1s['Pred']) * 100
    imp_full = (1 - l1s['Full-Refine'] / l1s['Pred']) * 100
    text += f"LF-Refine:  {imp_lf:+.1f}%\n"
    text += f"Full-Refine: {imp_full:+.1f}%\n\n"
    text += f"Iters: {args.num_iters}\n"
    text += f"Train views: {args.num_train_views}"
    ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(f'Low-Frequency Refinement Summary | {uid[:16]}...',
                 fontsize=13, fontweight='bold')
    save_path4 = os.path.join(args.output_dir, f'{uid}_summary.png')
    plt.savefig(save_path4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path4}")

    print("\n  完成！")


if __name__ == "__main__":
    main()
