"""
方向 A：Decoder 频率敏感度分析
方向 C：全局缩放修正实验

A: 对 T_GT 施加不同频段的可控扰动，测量渲染变化量 → decoder 的频率传递函数
C: 计算 per-channel optimal scaling，看简单缩放能修复多少

用法：
    python scripts/decoder_sensitivity_and_calibration.py \
        --triplane_path ./exps/gt_triplane/<uid>_triplane.pt \
        --data_dir ./data/rendered/<uid> \
        --model_name zxhezexin/openlrm-mix-base-1.1 \
        --infer_config ./configs/infer-b.yaml \
        --output_dir ./exps/decoder_analysis
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


# ─────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────

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


def render_multiview(model, triplane, cameras, render_size, device, view_indices):
    """渲染多个视角，返回 list of (3, H, W) tensors"""
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


def mean_l1_across_views(renders_a, renders_b):
    """计算两组渲染结果的平均 L1"""
    l1s = []
    for ra, rb in zip(renders_a, renders_b):
        l1s.append((ra - rb).abs().mean().item())
    return np.mean(l1s)


# ─────────────────────────────────────────
# 方向 A：Decoder 频率敏感度
# ─────────────────────────────────────────

def make_bandpass_noise(shape, r_low, r_high, amplitude):
    """生成指定频段的带通噪声"""
    # shape: (1, 3, D, H, W)
    noise = torch.zeros(shape)
    D, H, W = shape[2], shape[3], shape[4]
    cy, cx = H // 2, W // 2
    y = torch.arange(H).float() - cy
    x = torch.arange(W).float() - cx
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    r_norm = torch.sqrt(xx**2 + yy**2) / min(cy, cx)
    mask = (r_norm >= r_low) & (r_norm < r_high)

    for pi in range(3):
        for ci in range(D):
            # 随机相位的频域噪声
            random_phase = torch.randn(H, W) + 1j * torch.randn(H, W)
            random_fft = torch.fft.fftshift(random_phase)
            random_fft[~mask] = 0  # 只保留目标频段
            spatial = torch.fft.ifft2(torch.fft.ifftshift(random_fft)).real
            # 归一化到指定幅度
            spatial = spatial / (spatial.std() + 1e-8) * amplitude
            noise[0, pi, ci] = spatial

    return noise


def experiment_a(model, t_gt, render_cameras, render_size, device, view_indices, output_dir, uid):
    """方向 A：测量 decoder 对不同频段扰动的敏感度"""
    print("\n" + "="*60)
    print("  方向 A：Decoder 频率敏感度分析")
    print("="*60)

    # 基准渲染
    renders_gt = render_multiview(model, t_gt, render_cameras, render_size, device, view_indices)

    # 频段划分（更细粒度）
    num_bands = 10
    band_edges = np.linspace(0, 1.0, num_bands + 1)
    amplitude = 0.5  # 扰动幅度

    sensitivities = []
    band_centers = []

    for bi in range(num_bands):
        r_low, r_high = band_edges[bi], band_edges[bi + 1]
        band_center = (r_low + r_high) / 2
        band_centers.append(band_center)

        noise = make_bandpass_noise(t_gt.shape, r_low, r_high, amplitude)
        t_perturbed = t_gt + noise

        renders_perturbed = render_multiview(
            model, t_perturbed, render_cameras, render_size, device, view_indices
        )

        delta = mean_l1_across_views(renders_gt, renders_perturbed)
        sensitivities.append(delta / amplitude)

        print(f"  频段 [{r_low:.2f}, {r_high:.2f}): Δ_render/ε = {delta/amplitude:.6f}")

    sensitivities = np.array(sensitivities)
    band_centers = np.array(band_centers)

    # 归一化
    sensitivities_norm = sensitivities / sensitivities.sum()

    print(f"\n  低频 (0-0.3) 敏感度占比: {sensitivities_norm[:3].sum():.1%}")
    print(f"  中频 (0.3-0.6) 敏感度占比: {sensitivities_norm[3:6].sum():.1%}")
    print(f"  高频 (0.6-1.0) 敏感度占比: {sensitivities_norm[6:].sum():.1%}")

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 图1: 原始敏感度曲线
    ax = axes[0]
    ax.bar(band_centers, sensitivities, width=0.08, color='steelblue', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Normalized Frequency', fontsize=12)
    ax.set_ylabel('Sensitivity (Δ_render / ε)', fontsize=12)
    ax.set_title('Decoder Frequency Sensitivity', fontsize=13)
    ax.grid(axis='y', alpha=0.3)

    # 图2: 归一化占比
    ax = axes[1]
    colors = ['#2196F3'] * 3 + ['#FF9800'] * 3 + ['#F44336'] * 4
    ax.bar(band_centers, sensitivities_norm, width=0.08, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Normalized Frequency', fontsize=12)
    ax.set_ylabel('Fraction of Total Sensitivity', fontsize=12)
    ax.set_title('Normalized Sensitivity Distribution', fontsize=13)
    ax.grid(axis='y', alpha=0.3)

    # 图3: 与 triplane 残差能量的对比
    ax = axes[2]
    # 重新计算 triplane 残差在相同频段的能量分布
    residual = (t_gt - torch.load(
        os.path.join(os.path.dirname(output_dir), 'gt_triplane',
                     f'{uid}_triplane.pt'), map_location='cpu'
    )['t_pred'])[0]  # (3, D, H, W)

    residual_energy = []
    H, W = residual.shape[2], residual.shape[3]
    cy, cx = H // 2, W // 2
    y_grid = torch.arange(H).float() - cy
    x_grid = torch.arange(W).float() - cx
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
    r_norm_grid = torch.sqrt(xx**2 + yy**2) / min(cy, cx)

    for bi in range(num_bands):
        r_low, r_high = band_edges[bi], band_edges[bi + 1]
        mask = (r_norm_grid >= r_low) & (r_norm_grid < r_high)
        total_energy = 0
        count = 0
        for pi in range(3):
            for ci in range(residual.shape[1]):
                fft = torch.fft.fftshift(torch.fft.fft2(residual[pi, ci]))
                total_energy += fft.abs()[mask].sum().item()
                count += 1
        residual_energy.append(total_energy / count)

    residual_energy = np.array(residual_energy)
    residual_energy_norm = residual_energy / residual_energy.sum()

    # 计算"有效影响" = 残差能量 × 敏感度
    effective_impact = residual_energy_norm * sensitivities_norm
    effective_impact_norm = effective_impact / effective_impact.sum()

    x_pos = np.arange(num_bands)
    width = 0.25
    ax.bar(x_pos - width, residual_energy_norm, width, label='Triplane Residual Energy', color='#F44336', alpha=0.7)
    ax.bar(x_pos, sensitivities_norm, width, label='Decoder Sensitivity', color='#2196F3', alpha=0.7)
    ax.bar(x_pos + width, effective_impact_norm, width, label='Effective Impact', color='#4CAF50', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{band_centers[i]:.2f}' for i in range(num_bands)], fontsize=8)
    ax.set_xlabel('Frequency Band Center', fontsize=12)
    ax.set_ylabel('Normalized Fraction', fontsize=12)
    ax.set_title('Residual × Sensitivity = Effective Impact', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Decoder Frequency Sensitivity Analysis\n{uid}', fontsize=15, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{uid}_decoder_sensitivity.png')
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path}")

    return sensitivities, residual_energy_norm, effective_impact_norm


# ─────────────────────────────────────────
# 方向 C：全局缩放修正
# ─────────────────────────────────────────

def experiment_c(model, t_pred, t_gt, render_cameras, render_size, device, view_indices, output_dir, uid):
    """方向 C：计算 per-channel optimal scaling，测试简单修正效果"""
    print("\n" + "="*60)
    print("  方向 C：全局缩放修正实验")
    print("="*60)

    # 基准渲染
    renders_pred = render_multiview(model, t_pred, render_cameras, render_size, device, view_indices)
    renders_gt = render_multiview(model, t_gt, render_cameras, render_size, device, view_indices)
    l1_pred = mean_l1_across_views(renders_pred, renders_gt)
    print(f"  原始 T_pred L1: {l1_pred:.6f}")

    # 方法 1: 全局标量缩放
    flat_pred = t_pred.flatten()
    flat_gt = t_gt.flatten()
    alpha_global = (flat_pred * flat_gt).sum() / (flat_pred * flat_pred).sum()
    t_global_scaled = t_pred * alpha_global
    renders_global = render_multiview(model, t_global_scaled, render_cameras, render_size, device, view_indices)
    l1_global = mean_l1_across_views(renders_global, renders_gt)
    print(f"  全局缩放 (α={alpha_global:.4f}) L1: {l1_global:.6f} ({(1-l1_global/l1_pred)*100:+.1f}%)")

    # 方法 2: per-channel 缩放
    D = t_pred.shape[2]  # 48 channels
    alphas = torch.zeros(D)
    t_channel_scaled = t_pred.clone()
    for c in range(D):
        pred_c = t_pred[0, :, c].flatten()
        gt_c = t_gt[0, :, c].flatten()
        denom = (pred_c * pred_c).sum()
        if denom > 1e-8:
            alphas[c] = (pred_c * gt_c).sum() / denom
        else:
            alphas[c] = 1.0
        t_channel_scaled[0, :, c] = t_pred[0, :, c] * alphas[c]

    renders_channel = render_multiview(model, t_channel_scaled, render_cameras, render_size, device, view_indices)
    l1_channel = mean_l1_across_views(renders_channel, renders_gt)
    print(f"  Per-channel 缩放 L1: {l1_channel:.6f} ({(1-l1_channel/l1_pred)*100:+.1f}%)")

    # 方法 3: per-channel affine (scale + bias)
    betas = torch.zeros(D)
    t_affine = t_pred.clone()
    for c in range(D):
        pred_c = t_pred[0, :, c].flatten()
        gt_c = t_gt[0, :, c].flatten()
        # α, β = argmin ||α * pred + β - gt||²
        n = pred_c.shape[0]
        A = torch.stack([pred_c, torch.ones(n)], dim=1)
        solution = torch.linalg.lstsq(A, gt_c).solution
        alphas[c] = solution[0]
        betas[c] = solution[1]
        t_affine[0, :, c] = t_pred[0, :, c] * alphas[c] + betas[c]

    renders_affine = render_multiview(model, t_affine, render_cameras, render_size, device, view_indices)
    l1_affine = mean_l1_across_views(renders_affine, renders_gt)
    print(f"  Per-channel affine L1: {l1_affine:.6f} ({(1-l1_affine/l1_pred)*100:+.1f}%)")

    # 统计
    print(f"\n  Alpha 分布: mean={alphas.mean():.4f}, std={alphas.std():.4f}, "
          f"min={alphas.min():.4f}, max={alphas.max():.4f}")

    # 可视化
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    # 行1: 对比柱状图 + alpha 分布
    ax = fig.add_subplot(gs[0, 0:2])
    methods = ['T_pred\n(原始)', 'Global\nScaling', 'Per-Ch\nScaling', 'Per-Ch\nAffine', 'T_GT\n(oracle)']
    values = [l1_pred, l1_global, l1_channel, l1_affine, 0]
    colors = ['#F44336', '#FF9800', '#2196F3', '#4CAF50', '#9E9E9E']
    bars = ax.bar(methods, values, color=colors, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', fontsize=10)
    ax.set_ylabel('Mean L1 Error vs T_GT render', fontsize=12)
    ax.set_title('Calibration Methods Comparison', fontsize=13)
    ax.grid(axis='y', alpha=0.3)

    # 改善率表
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    table_data = [
        ['Method', 'L1', 'Improvement'],
        ['T_pred (baseline)', f'{l1_pred:.5f}', '-'],
        ['Global scaling', f'{l1_global:.5f}', f'{(1-l1_global/l1_pred)*100:+.1f}%'],
        ['Per-ch scaling', f'{l1_channel:.5f}', f'{(1-l1_channel/l1_pred)*100:+.1f}%'],
        ['Per-ch affine', f'{l1_affine:.5f}', f'{(1-l1_affine/l1_pred)*100:+.1f}%'],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.35, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)
    table[0, 0].set_text_props(fontweight='bold')
    table[0, 1].set_text_props(fontweight='bold')
    table[0, 2].set_text_props(fontweight='bold')
    ax.set_title('Improvement Summary', fontsize=13, pad=15)

    # Per-channel alpha 分布
    ax = fig.add_subplot(gs[0, 3])
    ax.hist(alphas.numpy(), bins=30, color='steelblue', alpha=0.8, edgecolor='black')
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='α=1 (no change)')
    ax.axvline(x=alphas.mean().item(), color='green', linestyle='--', linewidth=2, label=f'mean={alphas.mean():.3f}')
    ax.set_xlabel('Optimal α per channel', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Per-Channel Scaling Distribution', fontsize=13)
    ax.legend(fontsize=10)

    # 行2: 逐视角对比
    per_view_l1 = {'pred': [], 'global': [], 'channel': [], 'affine': []}
    for i in range(len(view_indices)):
        per_view_l1['pred'].append((renders_pred[i] - renders_gt[i]).abs().mean().item())
        per_view_l1['global'].append((renders_global[i] - renders_gt[i]).abs().mean().item())
        per_view_l1['channel'].append((renders_channel[i] - renders_gt[i]).abs().mean().item())
        per_view_l1['affine'].append((renders_affine[i] - renders_gt[i]).abs().mean().item())

    ax = fig.add_subplot(gs[1, 0:3])
    x_pos = np.arange(len(view_indices))
    w = 0.2
    ax.bar(x_pos - 1.5*w, per_view_l1['pred'], w, label='T_pred', color='#F44336', alpha=0.7)
    ax.bar(x_pos - 0.5*w, per_view_l1['global'], w, label='Global', color='#FF9800', alpha=0.7)
    ax.bar(x_pos + 0.5*w, per_view_l1['channel'], w, label='Per-ch scale', color='#2196F3', alpha=0.7)
    ax.bar(x_pos + 1.5*w, per_view_l1['affine'], w, label='Per-ch affine', color='#4CAF50', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'V{vi}' for vi in view_indices], fontsize=9)
    ax.set_ylabel('L1 Error', fontsize=12)
    ax.set_title('Per-View Error Comparison', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # 残差结构分析: T_pred vs T_gt 的 per-channel correlation
    ax = fig.add_subplot(gs[1, 3])
    correlations = []
    for c in range(D):
        pred_c = t_pred[0, :, c].flatten()
        gt_c = t_gt[0, :, c].flatten()
        corr = torch.corrcoef(torch.stack([pred_c, gt_c]))[0, 1].item()
        correlations.append(corr if not np.isnan(corr) else 0)
    correlations = np.array(correlations)
    ax.bar(range(D), correlations, color='steelblue', alpha=0.7)
    ax.axhline(y=correlations.mean(), color='red', linestyle='--', label=f'mean={correlations.mean():.3f}')
    ax.set_xlabel('Channel Index', fontsize=11)
    ax.set_ylabel('Pearson r (pred vs GT)', fontsize=11)
    ax.set_title('Per-Channel Correlation', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.2, 1.1)

    # 行3: 样本视角渲染对比 (选2个视角)
    sample_vi_indices = [0, len(view_indices)//2]
    for si, svi in enumerate(sample_vi_indices):
        ax = fig.add_subplot(gs[2, si*2:si*2+2])
        imgs = [
            renders_pred[svi].permute(1, 2, 0).clamp(0, 1).numpy(),
            renders_channel[svi].permute(1, 2, 0).clamp(0, 1).numpy(),
            renders_affine[svi].permute(1, 2, 0).clamp(0, 1).numpy(),
            renders_gt[svi].permute(1, 2, 0).clamp(0, 1).numpy(),
        ]
        combined = np.concatenate(imgs, axis=1)
        ax.imshow(combined)
        labels = ['Pred', 'Ch-Scale', 'Ch-Affine', 'GT']
        w_img = imgs[0].shape[1]
        for li, label in enumerate(labels):
            ax.text(li * w_img + 5, 15, label, color='yellow', fontsize=10, fontweight='bold')
        ax.set_title(f'View {view_indices[svi]}', fontsize=11)
        ax.axis('off')

    plt.suptitle(f'Decoder Sensitivity & Calibration Analysis\n{uid}', fontsize=16, fontweight='bold')
    save_path = os.path.join(output_dir, f'{uid}_calibration.png')
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path}")

    return {
        'l1_pred': l1_pred,
        'l1_global': l1_global,
        'l1_channel': l1_channel,
        'l1_affine': l1_affine,
        'alpha_global': alpha_global.item(),
        'alphas_mean': alphas.mean().item(),
        'correlations_mean': correlations.mean(),
    }


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplane_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="zxhezexin/openlrm-mix-base-1.1")
    parser.add_argument("--infer_config", type=str, default="./configs/infer-b.yaml")
    parser.add_argument("--output_dir", type=str, default="./exps/decoder_analysis")
    parser.add_argument("--num_views", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.infer_config)
    render_size = cfg.render_size

    # 加载
    print("[1/3] 加载数据...")
    data = torch.load(args.triplane_path, map_location='cpu')
    t_pred, t_gt = data['t_pred'], data['t_gt']
    uid = os.path.basename(args.triplane_path).replace('_triplane.pt', '')
    print(f"  UID: {uid}")

    print("[2/3] 加载模型...")
    from openlrm.models import model_dict
    hf_model_cls = wrap_model_hub(model_dict['lrm'])
    model = hf_model_cls.from_pretrained(args.model_name).to(device)
    model.eval()

    print("[3/3] 加载相机...")
    images, poses, intrinsics = load_render_data(args.data_dir)
    normalized_poses = camera_normalization_objaverse('auto', poses)
    intrinsics_batch = intrinsics.unsqueeze(0).repeat(poses.shape[0], 1, 1)
    render_cameras = build_camera_standard(normalized_poses, intrinsics_batch)

    os.makedirs(args.output_dir, exist_ok=True)

    total_views = poses.shape[0]
    step = max(1, total_views // args.num_views)
    view_indices = list(range(0, total_views, step))[:args.num_views]
    print(f"  分析视角: {view_indices}")

    # 实验 A
    sensitivities, residual_energy, effective_impact = experiment_a(
        model, t_gt, render_cameras, render_size, device, view_indices, args.output_dir, uid
    )

    # 实验 C
    results_c = experiment_c(
        model, t_pred, t_gt, render_cameras, render_size, device, view_indices, args.output_dir, uid
    )

    print("\n" + "="*60)
    print("  完成！")
    print("="*60)


if __name__ == "__main__":
    main()
