"""
渲染图像频率分析

用 T_pred 和 T_GT 分别从多视角渲染图像，对渲染结果做 2D FFT，
比较两者在各频段的差异。这比直接分析 triplane 特征更有物理意义。

用法：
    python scripts/rendered_frequency_analysis.py \
        --triplane_path ./exps/gt_triplane/<uid>_triplane.pt \
        --data_dir ./data/rendered/<uid> \
        --model_name zxhezexin/openlrm-mix-base-1.1 \
        --infer_config ./configs/infer-b.yaml \
        --output_dir ./exps/rendered_freq
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


def render_view(model, triplane, camera, render_size, device):
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
    return out['images_rgb'].squeeze().cpu()  # (3, H, W)


def image_freq_analysis(img):
    """对单张 RGB 图像做频率分析，返回灰度幅度谱"""
    # RGB → 灰度
    gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    fft2 = torch.fft.fft2(gray)
    fft_shift = torch.fft.fftshift(fft2)
    mag = fft_shift.abs()
    return mag


def radial_profile(mag_2d):
    H, W = mag_2d.shape
    cy, cx = H // 2, W // 2
    y = torch.arange(H).float() - cy
    x = torch.arange(W).float() - cx
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    r = torch.sqrt(xx**2 + yy**2)
    r_max = int(min(cy, cx))
    profile = torch.zeros(r_max)
    r_int = r.long()
    for ri in range(r_max):
        mask = r_int == ri
        if mask.any():
            profile[ri] = mag_2d[mask].mean()
    return profile.numpy()


def freq_band_energy(mag, r_low, r_high):
    H, W = mag.shape
    cy, cx = H // 2, W // 2
    y = torch.arange(H).float() - cy
    x = torch.arange(W).float() - cx
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    r_norm = torch.sqrt(xx**2 + yy**2) / min(cy, cx)
    mask = (r_norm >= r_low) & (r_norm < r_high)
    return mag[mask].sum().item()


def low_high_pass_images(img, cutoff=0.2):
    """对 RGB 图像做低通/高通滤波，返回分离后的图像"""
    result = {}
    for name, lo, hi in [('low', 0.0, cutoff), ('high', cutoff, 1.0)]:
        filtered = torch.zeros_like(img)
        for c in range(3):
            fft2 = torch.fft.fft2(img[c])
            fft_shift = torch.fft.fftshift(fft2)
            H, W = img.shape[1], img.shape[2]
            cy, cx = H // 2, W // 2
            y = torch.arange(H).float() - cy
            x = torch.arange(W).float() - cx
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            r_norm = torch.sqrt(xx**2 + yy**2) / min(cy, cx)
            mask = (r_norm >= lo) & (r_norm < hi)
            fft_filtered = fft_shift * mask.float()
            filtered[c] = torch.fft.ifft2(torch.fft.ifftshift(fft_filtered)).real
        result[name] = filtered
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplane_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="zxhezexin/openlrm-mix-base-1.1")
    parser.add_argument("--infer_config", type=str, default="./configs/infer-b.yaml")
    parser.add_argument("--output_dir", type=str, default="./exps/rendered_freq")
    parser.add_argument("--num_views", type=int, default=8,
                        help="用多少个均匀采样的视角做分析")
    parser.add_argument("--freq_cutoff", type=float, default=0.2,
                        help="低/高频分界线（归一化频率）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.infer_config)
    render_size = cfg.render_size

    # 加载数据
    print("[1/4] 加载数据...")
    data = torch.load(args.triplane_path, map_location='cpu')
    t_pred, t_gt = data['t_pred'], data['t_gt']
    source_view = data.get('source_view', 0)
    uid = os.path.basename(args.triplane_path).replace('_triplane.pt', '')
    print(f"  UID: {uid}")

    print("[2/4] 加载模型...")
    from openlrm.models import model_dict
    hf_model_cls = wrap_model_hub(model_dict['lrm'])
    model = hf_model_cls.from_pretrained(args.model_name).to(device)
    model.eval()

    print("[3/4] 加载相机...")
    images, poses, intrinsics = load_render_data(args.data_dir)
    normalized_poses = camera_normalization_objaverse('auto', poses)
    intrinsics_batch = intrinsics.unsqueeze(0).repeat(poses.shape[0], 1, 1)
    render_cameras = build_camera_standard(normalized_poses, intrinsics_batch)

    os.makedirs(args.output_dir, exist_ok=True)

    # 均匀选取视角
    total_views = poses.shape[0]
    step = max(1, total_views // args.num_views)
    view_indices = list(range(0, total_views, step))[:args.num_views]
    if source_view not in view_indices:
        view_indices[0] = source_view
    print(f"  分析视角: {view_indices}")

    # 逐视角渲染并分析
    print("[4/4] 逐视角渲染 + 频率分析...")
    bands = [('Low', 0.0, args.freq_cutoff), ('High', args.freq_cutoff, 1.0)]

    all_pred_profiles = []
    all_gt_profiles = []
    all_residual_profiles = []
    per_view_band_energy = {'pred': [], 'gt': [], 'residual': []}
    per_view_l1_low = []
    per_view_l1_high = []
    per_view_l1_total = []
    sample_views = []  # 存几个视角的图像用于可视化

    for vi in view_indices:
        cam = render_cameras[vi:vi+1]
        rgb_pred = render_view(model, t_pred, cam, render_size, device)
        rgb_gt = render_view(model, t_gt, cam, render_size, device)
        rgb_diff = rgb_pred - rgb_gt

        # 频谱
        mag_pred = image_freq_analysis(rgb_pred)
        mag_gt = image_freq_analysis(rgb_gt)
        mag_diff = image_freq_analysis(rgb_diff)

        # 径向曲线
        all_pred_profiles.append(radial_profile(mag_pred))
        all_gt_profiles.append(radial_profile(mag_gt))
        all_residual_profiles.append(radial_profile(mag_diff))

        # 频段能量
        for tag, mag in [('pred', mag_pred), ('gt', mag_gt), ('residual', mag_diff)]:
            energies = {}
            total = mag.sum().item()
            for bname, lo, hi in bands:
                energies[bname] = freq_band_energy(mag, lo, hi) / (total + 1e-8)
            per_view_band_energy[tag].append(energies)

        # 低频/高频分离后的 L1 误差
        pred_parts = low_high_pass_images(rgb_pred, args.freq_cutoff)
        gt_parts = low_high_pass_images(rgb_gt, args.freq_cutoff)
        l1_low = (pred_parts['low'] - gt_parts['low']).abs().mean().item()
        l1_high = (pred_parts['high'] - gt_parts['high']).abs().mean().item()
        l1_total = (rgb_pred - rgb_gt).abs().mean().item()
        per_view_l1_low.append(l1_low)
        per_view_l1_high.append(l1_high)
        per_view_l1_total.append(l1_total)

        is_source = (vi == source_view)
        print(f"  View {vi:2d}{'*' if is_source else ' '}: L1_total={l1_total:.4f}, L1_low={l1_low:.4f}, L1_high={l1_high:.4f}")

        # 存储采样视角
        if len(sample_views) < 4:
            sample_views.append({
                'vi': vi, 'is_source': is_source,
                'rgb_pred': rgb_pred, 'rgb_gt': rgb_gt,
                'pred_low': pred_parts['low'], 'pred_high': pred_parts['high'],
                'gt_low': gt_parts['low'], 'gt_high': gt_parts['high'],
            })

    # ── 汇总统计 ──
    mean_l1_low = np.mean(per_view_l1_low)
    mean_l1_high = np.mean(per_view_l1_high)
    mean_l1_total = np.mean(per_view_l1_total)
    print(f"\n{'='*50}")
    print(f"  渲染图像频率分析汇总 (cutoff={args.freq_cutoff})")
    print(f"{'='*50}")
    print(f"  平均 L1 total: {mean_l1_total:.6f}")
    print(f"  平均 L1 low:   {mean_l1_low:.6f} ({mean_l1_low/mean_l1_total:.1%} of total)")
    print(f"  平均 L1 high:  {mean_l1_high:.6f} ({mean_l1_high/mean_l1_total:.1%} of total)")
    print(f"  High/Low ratio: {mean_l1_high / (mean_l1_low + 1e-8):.2f}x")
    print(f"{'='*50}")

    # ── 绘图 ──
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

    # 行1：径向频谱 (平均)
    avg_pred = np.mean(all_pred_profiles, axis=0)
    avg_gt = np.mean(all_gt_profiles, axis=0)
    avg_res = np.mean(all_residual_profiles, axis=0)
    freqs = np.arange(len(avg_pred))

    ax = fig.add_subplot(gs[0, 0:2])
    ax.semilogy(freqs, avg_pred + 1, 'b-', linewidth=2, label='T_pred render')
    ax.semilogy(freqs, avg_gt + 1, 'g-', linewidth=2, label='T_GT render')
    ax.semilogy(freqs, avg_res + 1, 'r-', linewidth=2, label='Residual')
    cutoff_freq = int(args.freq_cutoff * len(freqs))
    ax.axvline(x=cutoff_freq, color='gray', linestyle='--', alpha=0.7, label=f'Cutoff={args.freq_cutoff}')
    ax.set_xlabel('Radial Frequency', fontsize=12)
    ax.set_ylabel('Magnitude (log)', fontsize=12)
    ax.set_title('Average Radial Spectrum (all views)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Pred/GT ratio
    ax = fig.add_subplot(gs[0, 2:4])
    ratio = (avg_pred + 1e-6) / (avg_gt + 1e-6)
    ax.plot(freqs, ratio, 'k-', linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=cutoff_freq, color='gray', linestyle='--', alpha=0.7, label=f'Cutoff={args.freq_cutoff}')
    ax.fill_between(freqs[:cutoff_freq], 0, ratio[:cutoff_freq], alpha=0.2, color='blue', label='Low freq')
    ax.fill_between(freqs[cutoff_freq:], 0, ratio[cutoff_freq:], alpha=0.2, color='red', label='High freq')
    avg_low_ratio = ratio[:cutoff_freq].mean()
    avg_high_ratio = ratio[cutoff_freq:].mean()
    ax.text(0.05, 0.95, f'Low freq avg ratio: {avg_low_ratio:.3f}\nHigh freq avg ratio: {avg_high_ratio:.3f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Radial Frequency', fontsize=12)
    ax.set_ylabel('T_pred / T_GT Ratio', fontsize=12)
    ax.set_title('Spectral Ratio: Pred vs GT (rendered images)', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.5)
    ax.grid(True, alpha=0.3)

    # 行2: 逐视角 L1 低频 vs 高频
    ax = fig.add_subplot(gs[1, 0:2])
    x_pos = np.arange(len(view_indices))
    width = 0.35
    ax.bar(x_pos - width/2, per_view_l1_low, width, label=f'Low freq L1', color='#2196F3', alpha=0.8)
    ax.bar(x_pos + width/2, per_view_l1_high, width, label=f'High freq L1', color='#F44336', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'V{vi}{"*" if vi==source_view else ""}' for vi in view_indices], fontsize=9)
    ax.set_ylabel('L1 Error', fontsize=11)
    ax.set_title('Per-View Low vs High Frequency Error', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # 比例饼图
    ax = fig.add_subplot(gs[1, 2])
    sizes = [mean_l1_low, mean_l1_high]
    labels = [f'Low\n{mean_l1_low/mean_l1_total:.1%}', f'High\n{mean_l1_high/mean_l1_total:.1%}']
    colors = ['#2196F3', '#F44336']
    ax.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90,
           textprops={'fontsize': 13, 'fontweight': 'bold'})
    ax.set_title('Error Attribution', fontsize=13)

    # 统计表
    ax = fig.add_subplot(gs[1, 3])
    ax.axis('off')
    table_data = [
        ['Metric', 'Value'],
        ['Total L1', f'{mean_l1_total:.5f}'],
        ['Low freq L1', f'{mean_l1_low:.5f}'],
        ['High freq L1', f'{mean_l1_high:.5f}'],
        ['Low %', f'{mean_l1_low/mean_l1_total:.1%}'],
        ['High %', f'{mean_l1_high/mean_l1_total:.1%}'],
        ['High/Low ratio', f'{mean_l1_high/(mean_l1_low+1e-8):.2f}x'],
        ['Cutoff', f'{args.freq_cutoff}'],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.45, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    table[0, 0].set_text_props(fontweight='bold')
    table[0, 1].set_text_props(fontweight='bold')
    ax.set_title('Summary Statistics', fontsize=13, pad=15)

    # 行3-4: 样本视角的低频/高频分离可视化
    for si, sv in enumerate(sample_views[:4]):
        # 行3: pred low vs gt low | pred high vs gt high
        row = 2
        col = si

        ax = fig.add_subplot(gs[row, col])
        # 拼图: 上半=low freq diff, 下半=high freq diff
        diff_low = (sv['pred_low'] - sv['gt_low']).abs().mean(dim=0)
        diff_high = (sv['pred_high'] - sv['gt_high']).abs().mean(dim=0)

        combined = torch.cat([diff_low, diff_high], dim=0)  # (2H, W)
        im = ax.imshow(combined.numpy(), cmap='hot', vmin=0, vmax=0.15)
        H = diff_low.shape[0]
        ax.axhline(y=H - 0.5, color='white', linewidth=2)
        ax.text(5, H // 2, 'Low', color='white', fontsize=10, fontweight='bold')
        ax.text(5, H + H // 2, 'High', color='white', fontsize=10, fontweight='bold')
        tag = '*SRC' if sv['is_source'] else ''
        ax.set_title(f'View {sv["vi"]} {tag} Error Map', fontsize=10)
        ax.axis('off')

        # 行4: pred vs gt 原图对比
        ax = fig.add_subplot(gs[row + 1, col])
        pred_np = sv['rgb_pred'].permute(1, 2, 0).clamp(0, 1).numpy()
        gt_np = sv['rgb_gt'].permute(1, 2, 0).clamp(0, 1).numpy()
        combined_img = np.concatenate([pred_np, gt_np], axis=1)
        ax.imshow(combined_img)
        ax.text(5, 15, 'Pred', color='yellow', fontsize=10, fontweight='bold')
        ax.text(render_size + 5, 15, 'GT', color='yellow', fontsize=10, fontweight='bold')
        ax.set_title(f'View {sv["vi"]} Pred|GT', fontsize=10)
        ax.axis('off')

    plt.suptitle(f'Rendered Image Frequency Analysis\n{uid}', fontsize=16, fontweight='bold')
    save_path = os.path.join(args.output_dir, f'{uid}_rendered_freq.png')
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\n保存分析图: {save_path}")
    print("完成！")


if __name__ == "__main__":
    main()
