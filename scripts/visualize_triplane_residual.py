"""
Triplane 残差可视化脚本

从 optimize_gt_triplane.py 的输出加载 T_pred 和 T_GT，
在 3D 空间中采样点计算逐点误差，生成误差热力图。

输出：
    1. 三个正交平面上的 2D 残差热力图
    2. 3D 空间中的误差点云 (.ply)，颜色编码误差大小

用法：
    python scripts/visualize_triplane_residual.py \
        --triplane_path ./exps/gt_triplane/<uid>_triplane.pt \
        --model_name zxhezexin/openlrm-mix-base-1.1 \
        --output_dir ./exps/residual_vis \
        --grid_size 128
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from accelerate import PartialState
PartialState()
torch._dynamo.config.suppress_errors = True

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openlrm.utils.hf_hub import wrap_model_hub


def query_triplane_features(model, triplane, grid_size, device):
    """
    在 3D 空间均匀采样点，查询 triplane 特征得到 RGB 和 sigma。

    返回：
        points: (G^3, 3) 空间坐标
        rgb: (G^3, 3) 颜色
        sigma: (G^3, 1) 密度
    """
    # 在 [-1, 1]^3 空间均匀采样
    coords = torch.linspace(-1, 1, grid_size, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing='ij')
    points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(1, -1, 3)  # (1, G^3, 3)

    triplane = triplane.to(device)
    with torch.no_grad():
        out = model.synthesizer.forward_points(triplane, points, chunk_size=2**18)

    return points.squeeze(0).cpu(), out['rgb'].squeeze(0).cpu(), out['sigma'].squeeze(0).cpu()


def visualize_triplane_planes(t_pred, t_gt, output_dir, uid):
    """可视化三个正交平面上的特征残差热力图"""
    residual = (t_pred - t_gt).abs()  # (1, 3, D, H, W)
    # 对通道维度取均值，得到每个平面的空间残差
    plane_residuals = residual[0].mean(dim=1)  # (3, H, W)

    plane_names = ['XY平面', 'XZ平面', 'YZ平面']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i in range(3):
        im = axes[i].imshow(
            plane_residuals[i].numpy(),
            cmap='hot', interpolation='nearest',
            vmin=0, vmax=plane_residuals.max().item(),
        )
        axes[i].set_title(f'{plane_names[i]} 特征残差', fontsize=14)
        axes[i].set_xlabel('空间坐标 u')
        axes[i].set_ylabel('空间坐标 v')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.suptitle(f'Triplane 特征残差热力图 (T_pred vs T_GT)\n{uid}', fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{uid}_triplane_residual_heatmap.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存 2D 热力图: {save_path}")


def visualize_3d_error_cloud(
    model, t_pred, t_gt, output_dir, uid,
    grid_size=128, sigma_threshold=1.0, device='cuda',
):
    """
    在 3D 空间中采样点，用两个 triplane 分别查询 RGB/sigma，
    仅保留 T_GT 中 sigma 较高的点（即物体表面附近），
    用颜色编码逐点 RGB 误差，输出 .ply 点云。
    """
    print(f"  采样 {grid_size}^3 = {grid_size**3} 个 3D 点...")

    # 查询 T_GT 的 sigma 和 RGB
    points, rgb_gt, sigma_gt = query_triplane_features(model, t_gt, grid_size, device)
    # 查询 T_pred 的 RGB
    _, rgb_pred, sigma_pred = query_triplane_features(model, t_pred, grid_size, device)

    # 过滤出物体表面附近的点（sigma > threshold）
    sigma_vals = sigma_gt.squeeze(-1)
    surface_mask = sigma_vals > sigma_threshold
    num_surface = surface_mask.sum().item()
    print(f"  表面点数量: {num_surface} / {grid_size**3} (sigma > {sigma_threshold})")

    if num_surface == 0:
        print("  警告: 没有检测到表面点。请降低 sigma_threshold 重试。")
        # 自动降低阈值
        sigma_threshold = sigma_vals.quantile(0.9).item()
        surface_mask = sigma_vals > sigma_threshold
        num_surface = surface_mask.sum().item()
        print(f"  自动调整阈值至 {sigma_threshold:.2f}, 表面点: {num_surface}")

    surface_points = points[surface_mask].numpy()  # (P, 3)
    surface_rgb_gt = rgb_gt[surface_mask].numpy()
    surface_rgb_pred = rgb_pred[surface_mask].numpy()

    # 计算逐点 RGB 误差
    per_point_error = np.linalg.norm(surface_rgb_pred - surface_rgb_gt, axis=-1)  # (P,)

    # 归一化误差到 [0, 1] 用于颜色映射
    if per_point_error.max() > 0:
        error_normalized = per_point_error / per_point_error.max()
    else:
        error_normalized = per_point_error

    # 用 hot colormap 编码误差: 蓝(低) -> 红(高)
    cmap = plt.cm.get_cmap('coolwarm')
    error_colors = (cmap(error_normalized)[:, :3] * 255).astype(np.uint8)

    def save_ply(path, verts, colors):
        with open(path, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(verts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for v, c in zip(verts, colors):
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

    # 保存误差点云
    error_ply_path = os.path.join(output_dir, f'{uid}_error_cloud.ply')
    save_ply(error_ply_path, surface_points, error_colors)
    print(f"  保存 3D 误差点云: {error_ply_path}")

    # 同时保存 GT 外观点云作为参考
    gt_colors = (np.clip(surface_rgb_gt, 0, 1) * 255).astype(np.uint8)
    gt_ply_path = os.path.join(output_dir, f'{uid}_gt_cloud.ply')
    save_ply(gt_ply_path, surface_points, gt_colors)
    print(f"  保存 GT 参考点云: {gt_ply_path}")

    # 保存 pred 外观点云作为参考
    pred_colors = (np.clip(surface_rgb_pred, 0, 1) * 255).astype(np.uint8)
    pred_ply_path = os.path.join(output_dir, f'{uid}_pred_cloud.ply')
    save_ply(pred_ply_path, surface_points, pred_colors)
    print(f"  保存 Pred 参考点云: {pred_ply_path}")

    # 打印误差统计
    print(f"\n===== 3D 空间逐点 RGB 误差统计 =====")
    print(f"  表面点数: {num_surface}")
    print(f"  误差 mean: {per_point_error.mean():.6f}")
    print(f"  误差 std:  {per_point_error.std():.6f}")
    print(f"  误差 max:  {per_point_error.max():.6f}")
    print(f"  误差 P90:  {np.percentile(per_point_error, 90):.6f}")
    print(f"  误差 P99:  {np.percentile(per_point_error, 99):.6f}")

    return per_point_error


def visualize_error_histogram(per_point_error, output_dir, uid):
    """误差分布直方图"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(per_point_error, bins=100, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(per_point_error.mean(), color='red', linestyle='--', label=f'Mean={per_point_error.mean():.4f}')
    ax.axvline(np.percentile(per_point_error, 90), color='orange', linestyle='--', label=f'P90={np.percentile(per_point_error, 90):.4f}')
    ax.set_xlabel('Per-point RGB Error (L2)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'3D Surface Error Distribution\n{uid}', fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{uid}_error_histogram.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存误差直方图: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Triplane 残差可视化")
    parser.add_argument("--triplane_path", type=str, required=True,
                        help="optimize_gt_triplane.py 的输出 .pt 文件路径")
    parser.add_argument("--model_name", type=str, default="zxhezexin/openlrm-mix-base-1.1")
    parser.add_argument("--output_dir", type=str, default="./exps/residual_vis")
    parser.add_argument("--grid_size", type=int, default=128,
                        help="3D 采样分辨率 (grid_size^3 个点)")
    parser.add_argument("--sigma_threshold", type=float, default=1.0,
                        help="表面点 sigma 阈值")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 triplane 数据
    print("[1/4] 加载 triplane 数据...")
    data = torch.load(args.triplane_path, map_location='cpu')
    t_pred = data['t_pred']
    t_gt = data['t_gt']
    uid = os.path.basename(args.triplane_path).replace('_triplane.pt', '')
    print(f"  UID: {uid}")
    print(f"  T_pred 形状: {t_pred.shape}, T_GT 形状: {t_gt.shape}")

    # 加载模型（只需要 synthesizer 部分）
    print("[2/4] 加载模型 decoder...")
    from openlrm.models import model_dict
    hf_model_cls = wrap_model_hub(model_dict['lrm'])
    model = hf_model_cls.from_pretrained(args.model_name).to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    # 2D 热力图
    print("[3/4] 生成 Triplane 2D 残差热力图...")
    visualize_triplane_planes(t_pred, t_gt, args.output_dir, uid)

    # 3D 误差点云
    print("[4/4] 生成 3D 误差点云...")
    per_point_error = visualize_3d_error_cloud(
        model=model,
        t_pred=t_pred,
        t_gt=t_gt,
        output_dir=args.output_dir,
        uid=uid,
        grid_size=args.grid_size,
        sigma_threshold=args.sigma_threshold,
        device=device,
    )
    visualize_error_histogram(per_point_error, args.output_dir, uid)

    print("\n完成！")


if __name__ == "__main__":
    main()
