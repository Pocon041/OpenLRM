"""
Triplane 残差频率分解分析

对 T_pred - T_GT 做 2D FFT，将误差分解为低频/中频/高频分量，
回答：encoder 是形状没对（低频误差大）还是细节丢了（高频误差大）？

用法：
    python scripts/frequency_analysis.py \
        --triplane_dir ./exps/gt_triplane \
        --output_dir ./exps/frequency_analysis
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_triplane_pair(path):
    """加载一对 T_pred / T_GT"""
    data = torch.load(path, map_location='cpu')
    return data['t_pred'], data['t_gt']


def freq_band_energy(fft_mag, h, w, r_low, r_high):
    """计算频率环 [r_low, r_high) 内的能量占比"""
    cy, cx = h // 2, w // 2
    y = torch.arange(h).float() - cy
    x = torch.arange(w).float() - cx
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    r = torch.sqrt(xx**2 + yy**2)
    r_max = min(cy, cx)
    # 归一化到 [0, 1]
    r_norm = r / r_max
    mask = (r_norm >= r_low) & (r_norm < r_high)
    return fft_mag[mask].sum().item()


def analyze_single(t_pred, t_gt, uid):
    """对单个样本做频率分析"""
    residual = (t_pred - t_gt)  # (1, 3, D, H, W)
    residual = residual[0]       # (3, D, H, W)

    plane_names = ['XY', 'XZ', 'YZ']
    # 频率带定义 (归一化半径)
    bands = [
        ('Low  (0-0.15)', 0.0, 0.15),
        ('Mid  (0.15-0.4)', 0.15, 0.4),
        ('High (0.4-1.0)', 0.4, 1.0),
    ]

    results = {}
    all_spectra = {}

    for pi in range(3):
        plane = residual[pi]  # (D, H, W) — D 是通道维
        # 对每个通道做 2D FFT，然后取平均幅度谱
        H, W = plane.shape[1], plane.shape[2]
        avg_mag = torch.zeros(H, W)

        for ci in range(plane.shape[0]):
            channel = plane[ci]  # (H, W)
            fft2 = torch.fft.fft2(channel)
            fft_shift = torch.fft.fftshift(fft2)
            mag = fft_shift.abs()
            avg_mag += mag

        avg_mag /= plane.shape[0]
        all_spectra[plane_names[pi]] = avg_mag

        # 计算各频段能量
        total_energy = avg_mag.sum().item()
        band_energies = {}
        for name, r_low, r_high in bands:
            energy = freq_band_energy(avg_mag, H, W, r_low, r_high)
            band_energies[name] = energy / (total_energy + 1e-8)

        results[plane_names[pi]] = band_energies

    # 同样分析 T_pred 和 T_GT 本身的频谱（用于对比）
    pred_spectra = {}
    gt_spectra = {}
    for pi in range(3):
        H, W = t_pred.shape[3], t_pred.shape[4]
        pred_avg = torch.zeros(H, W)
        gt_avg = torch.zeros(H, W)
        for ci in range(t_pred.shape[2]):
            pred_fft = torch.fft.fftshift(torch.fft.fft2(t_pred[0, pi, ci]))
            gt_fft = torch.fft.fftshift(torch.fft.fft2(t_gt[0, pi, ci]))
            pred_avg += pred_fft.abs()
            gt_avg += gt_fft.abs()
        pred_avg /= t_pred.shape[2]
        gt_avg /= t_pred.shape[2]
        pred_spectra[plane_names[pi]] = pred_avg
        gt_spectra[plane_names[pi]] = gt_avg

    return results, all_spectra, pred_spectra, gt_spectra


def radial_profile(mag_2d):
    """计算 2D 幅度谱的径向平均（频率 vs 能量曲线）"""
    H, W = mag_2d.shape
    cy, cx = H // 2, W // 2
    y = torch.arange(H).float() - cy
    x = torch.arange(W).float() - cx
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    r = torch.sqrt(xx**2 + yy**2)

    r_max = int(min(cy, cx))
    profile = torch.zeros(r_max)
    counts = torch.zeros(r_max)

    r_int = r.long()
    for ri in range(r_max):
        mask = r_int == ri
        if mask.any():
            profile[ri] = mag_2d[mask].mean()
            counts[ri] = mask.sum()

    return profile.numpy()


def plot_frequency_analysis(uid, results, spectra, pred_spectra, gt_spectra, output_dir):
    """生成综合频率分析图"""
    plane_names = ['XY', 'XZ', 'YZ']

    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    # ── 行1：2D 幅度谱 (残差 / T_pred / T_GT) ──
    for pi, pname in enumerate(plane_names):
        ax = fig.add_subplot(gs[0, pi])
        mag = spectra[pname]
        im = ax.imshow(
            torch.log1p(mag).numpy(),
            cmap='inferno', interpolation='nearest',
        )
        ax.set_title(f'Residual Spectrum - {pname}', fontsize=11)
        ax.set_xlabel('freq u')
        ax.set_ylabel('freq v')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 行1第4列：频段能量柱状图
    ax = fig.add_subplot(gs[0, 3])
    band_names = list(results['XY'].keys())
    x_pos = np.arange(len(band_names))
    width = 0.25
    for i, pname in enumerate(plane_names):
        values = [results[pname][b] for b in band_names]
        ax.bar(x_pos + i * width, values, width, label=pname, alpha=0.8)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(['Low', 'Mid', 'High'], fontsize=10)
    ax.set_ylabel('Energy Fraction', fontsize=11)
    ax.set_title('Residual Energy by Frequency Band', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)

    # ── 行2：径向频率曲线 (T_pred vs T_GT vs Residual) ──
    for pi, pname in enumerate(plane_names):
        ax = fig.add_subplot(gs[1, pi])
        rp_residual = radial_profile(spectra[pname])
        rp_pred = radial_profile(pred_spectra[pname])
        rp_gt = radial_profile(gt_spectra[pname])

        freqs = np.arange(len(rp_residual))
        ax.semilogy(freqs, rp_pred + 1, 'b-', alpha=0.7, linewidth=1.5, label='T_pred')
        ax.semilogy(freqs, rp_gt + 1, 'g-', alpha=0.7, linewidth=1.5, label='T_GT')
        ax.semilogy(freqs, rp_residual + 1, 'r-', alpha=0.9, linewidth=2, label='Residual')
        ax.set_xlabel('Radial Frequency', fontsize=10)
        ax.set_ylabel('Magnitude (log)', fontsize=10)
        ax.set_title(f'Radial Spectrum - {pname}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # 行2第4列：T_pred vs T_GT 的高频能量比
    ax = fig.add_subplot(gs[1, 3])
    for pi, pname in enumerate(plane_names):
        rp_pred = radial_profile(pred_spectra[pname])
        rp_gt = radial_profile(gt_spectra[pname])
        ratio = (rp_pred + 1e-6) / (rp_gt + 1e-6)
        freqs = np.arange(len(ratio))
        ax.plot(freqs, ratio, linewidth=1.5, label=pname, alpha=0.8)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Radial Frequency', fontsize=10)
    ax.set_ylabel('T_pred / T_GT Magnitude Ratio', fontsize=10)
    ax.set_title('Pred vs GT Spectral Ratio', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2)

    # ── 行3：低频/高频残差空间可视化 ──
    for pi, pname in enumerate(plane_names):
        ax = fig.add_subplot(gs[2, pi])

        # 取残差的一个平面，做低频/高频分离后可视化
        residual_plane = spectra[pname]  # 这是 avg magnitude
        H, W = residual_plane.shape
        cy, cx = H // 2, W // 2
        y = torch.arange(H).float() - cy
        x = torch.arange(W).float() - cx
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        r_norm = torch.sqrt(xx**2 + yy**2) / min(cy, cx)

        low_energy = residual_plane[r_norm < 0.15].sum().item()
        high_energy = residual_plane[r_norm >= 0.4].sum().item()
        total = residual_plane.sum().item()

        sizes = [low_energy, total - low_energy - high_energy, high_energy]
        labels = [f'Low\n{low_energy/total:.1%}',
                  f'Mid\n{(total-low_energy-high_energy)/total:.1%}',
                  f'High\n{high_energy/total:.1%}']
        colors = ['#2196F3', '#FF9800', '#F44336']
        ax.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90,
               textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax.set_title(f'{pname} Plane Energy Split', fontsize=11)

    # 行3第4列：汇总统计表
    ax = fig.add_subplot(gs[2, 3])
    ax.axis('off')
    table_data = [['Plane', 'Low %', 'Mid %', 'High %']]
    for pname in plane_names:
        vals = results[pname]
        row = [pname]
        for b in vals:
            row.append(f'{vals[b]:.1%}')
        table_data.append(row)
    # 三个平面平均
    avg_row = ['AVG']
    for bi, b in enumerate(results['XY'].keys()):
        avg_val = np.mean([results[p][b] for p in plane_names])
        avg_row.append(f'{avg_val:.1%}')
    table_data.append(avg_row)

    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.2, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.0)
    for j in range(4):
        table[0, j].set_text_props(fontweight='bold')
    table[len(table_data)-1, 0].set_text_props(fontweight='bold')
    ax.set_title('Energy Distribution Summary', fontsize=11, pad=20)

    plt.suptitle(f'Triplane Residual Frequency Analysis\n{uid}', fontsize=16, fontweight='bold')
    save_path = os.path.join(output_dir, f'{uid}_frequency_analysis.png')
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  保存频率分析图: {save_path}")


def plot_multi_comparison(all_results, output_dir):
    """多样本频率对比图"""
    if len(all_results) < 2:
        return

    uids = list(all_results.keys())
    plane_names = ['XY', 'XZ', 'YZ']
    band_keys = list(all_results[uids[0]]['XY'].keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for bi, bname in enumerate(band_keys):
        ax = axes[bi]
        x_pos = np.arange(len(uids))
        width = 0.25
        for pi, pname in enumerate(plane_names):
            values = [all_results[uid][pname][bname] for uid in uids]
            ax.bar(x_pos + pi * width, values, width, label=pname, alpha=0.8)
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels([u[:8] + '...' for u in uids], fontsize=9, rotation=15)
        ax.set_ylabel('Energy Fraction', fontsize=11)
        ax.set_title(bname, fontsize=12)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Cross-Sample Frequency Band Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'cross_sample_frequency.png')
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  保存跨样本对比图: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Triplane 残差频率分析")
    parser.add_argument("--triplane_dir", type=str, default="./exps/gt_triplane",
                        help="包含 *_triplane.pt 文件的目录")
    parser.add_argument("--output_dir", type=str, default="./exps/frequency_analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pt_files = sorted(glob.glob(os.path.join(args.triplane_dir, '*_triplane.pt')))
    if not pt_files:
        print(f"错误: 在 {args.triplane_dir} 中未找到 *_triplane.pt 文件")
        return

    print(f"找到 {len(pt_files)} 个 triplane 文件\n")

    all_results = {}

    for pt_path in pt_files:
        uid = os.path.basename(pt_path).replace('_triplane.pt', '')
        print(f"━━━ 分析 {uid} ━━━")

        t_pred, t_gt = load_triplane_pair(pt_path)
        print(f"  T_pred: {t_pred.shape}, T_GT: {t_gt.shape}")

        # 基本残差统计
        residual = (t_pred - t_gt).abs()
        print(f"  残差 L1 mean: {residual.mean():.4f}, max: {residual.max():.4f}")

        results, spectra, pred_spectra, gt_spectra = analyze_single(t_pred, t_gt, uid)
        all_results[uid] = results

        # 打印频段能量
        for pname in ['XY', 'XZ', 'YZ']:
            vals = results[pname]
            parts = [f"{k}: {v:.1%}" for k, v in vals.items()]
            print(f"  {pname}: {', '.join(parts)}")

        plot_frequency_analysis(uid, results, spectra, pred_spectra, gt_spectra, args.output_dir)
        print()

    # 跨样本对比
    if len(all_results) > 1:
        plot_multi_comparison(all_results, args.output_dir)

    print("完成！")


if __name__ == "__main__":
    main()
