"""
可见性感知误差分析脚本

验证假设：encoder 在输入视角可见区域误差低，不可见（遮挡）区域误差高。

方法：
    1. 从 T_GT 渲染源视角深度图
    2. 将 3D 表面点投影到源相机，对比深度判定可见/遮挡
    3. 分别统计两组的误差分布

用法：
    python scripts/visibility_aware_analysis.py \
        --triplane_path ./exps/gt_triplane/<uid>_triplane.pt \
        --data_dir ./data/rendered/<uid> \
        --model_name zxhezexin/openlrm-mix-base-1.1 \
        --infer_config ./configs/infer-b.yaml \
        --output_dir ./exps/visibility_analysis
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
    build_camera_principle,
    camera_normalization_objaverse,
    compose_extrinsic_RT,
    create_intrinsics,
)
from openlrm.utils.hf_hub import wrap_model_hub


# ─────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────

def load_render_data(data_dir, num_views=32):
    """加载渲染数据"""
    rgba_dir = os.path.join(data_dir, 'rgba')
    pose_dir = os.path.join(data_dir, 'pose')
    intrinsics_path = os.path.join(data_dir, 'intrinsics.npy')

    intrinsics = torch.from_numpy(np.load(intrinsics_path)).float()

    images = []
    poses = []
    for i in range(num_views):
        img_path = os.path.join(rgba_dir, f'{i:03d}.png')
        img = np.array(Image.open(img_path)).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        rgb = img_tensor[:3] * img_tensor[3:4] + (1 - img_tensor[3:4])
        images.append(rgb)

        pose = np.load(os.path.join(pose_dir, f'{i:03d}.npy'))
        poses.append(torch.from_numpy(pose).float())

    images = torch.stack(images, dim=0)
    poses = torch.stack(poses, dim=0)
    return images, poses, intrinsics


def query_triplane_features(model, triplane, grid_size, device):
    """在 3D 空间均匀采样，查询 triplane 特征"""
    coords = torch.linspace(-1, 1, grid_size, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing='ij')
    points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(1, -1, 3)

    triplane = triplane.to(device)
    with torch.no_grad():
        out = model.synthesizer.forward_points(triplane, points, chunk_size=2**18)

    return points.squeeze(0).cpu(), out['rgb'].squeeze(0).cpu(), out['sigma'].squeeze(0).cpu()


def render_from_triplane(model, triplane, camera, render_size, device):
    """用给定 triplane 和相机参数渲染 RGB + depth"""
    triplane = triplane.to(device)
    camera = camera.to(device)
    render_anchors = torch.zeros(1, 1, 2, device=device)
    render_resolutions = torch.ones(1, 1, 1, device=device) * render_size
    render_bg_colors = torch.ones(1, 1, 1, device=device)

    with torch.no_grad():
        out = model.synthesizer(
            planes=triplane,
            cameras=camera.unsqueeze(0),  # (1, 1, D_cam)
            anchors=render_anchors,
            resolutions=render_resolutions,
            bg_colors=render_bg_colors,
            region_size=render_size,
        )
    rgb = out['images_rgb'].squeeze()    # (3, H, W)
    depth = out['images_depth'].squeeze()  # (1, H, W) or (H, W)
    return rgb.cpu(), depth.cpu()


def build_source_camera(poses, intrinsics, source_view, device):
    """构建源视角的归一化相机参数"""
    normalized_poses = camera_normalization_objaverse('auto', poses)
    intrinsics_batch = intrinsics.unsqueeze(0).repeat(poses.shape[0], 1, 1)
    render_cameras = build_camera_standard(normalized_poses, intrinsics_batch)
    return render_cameras[source_view:source_view+1].to(device)  # (1, D_cam)


def project_points_to_camera(points, cam2world, intrinsics_3x3, render_size):
    """
    将 3D 点投影到相机图像平面。

    NeRF 约定：相机看向 -z 方向，因此相机前方的点 z_cam < 0，
    深度 = -z_cam（正值）。

    参数：
        points: (P, 3) 世界坐标
        cam2world: (4, 4) 相机外参 (camera-to-world)
        intrinsics_3x3: (3, 3) 归一化相机内参
        render_size: 渲染分辨率

    返回：
        pixel_coords: (P, 2) 像素坐标 (u, v)
        depths: (P,) 正深度值（相机前方为正）
    """
    # world-to-camera
    world2cam = torch.inverse(cam2world)

    # 变换到相机坐标系
    points_h = torch.cat([points, torch.ones(len(points), 1)], dim=1)  # (P, 4)
    points_cam = (world2cam @ points_h.T).T[:, :3]  # (P, 3)

    # NeRF 约定：相机看向 -z，所以 depth = -z_cam
    depths = -points_cam[:, 2]  # 正值 = 在相机前方

    # 投影到归一化像素坐标，再乘以 render_size
    # u_norm = fx * X / (-Z) + cx = fx * X / depth + cx
    # v_norm = fy * Y / (-Z) + cy = fy * Y / depth + cy
    fx = intrinsics_3x3[0, 0]
    fy = intrinsics_3x3[1, 1]
    cx = intrinsics_3x3[0, 2]
    cy = intrinsics_3x3[1, 2]

    safe_depth = depths.clamp(min=1e-6)
    u_norm = fx * points_cam[:, 0] / safe_depth + cx
    v_norm = fy * points_cam[:, 1] / safe_depth + cy

    pixel_u = u_norm * render_size
    pixel_v = v_norm * render_size

    pixel_coords = torch.stack([pixel_u, pixel_v], dim=-1)
    return pixel_coords, depths


def compute_visibility_depthbuffer(
    surface_points, cam_params, depth_map, render_size, depth_tolerance=0.05,
):
    """
    通过深度缓冲判断每个表面点是否从源视角可见。

    原理：将 3D 点投影到源相机，比较其深度与渲染深度图。
    如果深度接近（差值 < tolerance），则可见；否则被遮挡。
    """
    points = torch.from_numpy(surface_points).float()

    # 解析相机参数
    cam2world = cam_params[:16].reshape(4, 4)
    intrinsics_flat = cam_params[16:25].reshape(3, 3)

    # 投影
    pixel_coords, point_depths = project_points_to_camera(
        points, cam2world, intrinsics_flat, render_size,
    )

    u = pixel_coords[:, 0]
    v = pixel_coords[:, 1]
    in_front = point_depths > 0
    in_frame = (u >= 0) & (u < render_size) & (v >= 0) & (v < render_size) & in_front

    # 调试信息
    print(f"    [debug] 深度范围: point_depths [{point_depths.min():.3f}, {point_depths.max():.3f}]")
    print(f"    [debug] 在相机前方: {in_front.sum()}/{len(points)}")
    print(f"    [debug] 像素 u 范围: [{u[in_front].min():.1f}, {u[in_front].max():.1f}]" if in_front.any() else "    [debug] 无前方点")
    print(f"    [debug] 像素 v 范围: [{v[in_front].min():.1f}, {v[in_front].max():.1f}]" if in_front.any() else "")
    print(f"    [debug] 在画面内: {in_frame.sum()}/{len(points)}")

    visible_mask = np.zeros(len(surface_points), dtype=bool)

    if in_frame.sum() == 0:
        print("    [debug] 警告: 深度缓冲法无画面内点，回退到法线法")
        return None  # 返回 None 表示需要回退

    # 取整像素坐标
    u_int = u[in_frame].long().clamp(0, render_size - 1)
    v_int = v[in_frame].long().clamp(0, render_size - 1)

    # 查询深度图
    if depth_map.dim() == 3:
        depth_map = depth_map.squeeze(0)
    rendered_depth = depth_map[v_int, u_int]

    # 深度对比（使用相对容差）
    depth_diff = (point_depths[in_frame] - rendered_depth).abs()
    rel_tolerance = depth_tolerance * rendered_depth  # 相对容差
    is_visible = depth_diff < rel_tolerance

    print(f"    [debug] 深度差分布: mean={depth_diff.mean():.4f}, median={depth_diff.median():.4f}, max={depth_diff.max():.4f}")
    print(f"    [debug] 通过深度测试: {is_visible.sum()}/{in_frame.sum()}")

    # 写回
    in_frame_indices = torch.where(in_frame)[0]
    visible_mask[in_frame_indices[is_visible].numpy()] = True

    return visible_mask


def compute_visibility_normal(
    model, triplane, surface_points, cam_params, device, eps=0.02,
):
    """
    通过表面法线 vs 视线方向判断可见性。

    原理：用 sigma 场的梯度估计法线，如果法线朝向相机则可见。
    比深度缓冲法更鲁棒，不依赖投影坐标系。
    """
    cam2world = cam_params[:16].reshape(4, 4)
    camera_pos = cam2world[:3, 3].numpy()  # 相机世界坐标

    pts = torch.from_numpy(surface_points).float().to(device)
    triplane = triplane.to(device)

    # 用有限差分估计 sigma 梯度 → 表面法线
    grads = []
    for dim in range(3):
        pts_plus = pts.clone()
        pts_minus = pts.clone()
        pts_plus[:, dim] += eps
        pts_minus[:, dim] -= eps

        with torch.no_grad():
            sigma_plus = model.synthesizer.forward_points(
                triplane, pts_plus.unsqueeze(0), chunk_size=2**18
            )['sigma'].squeeze(0).squeeze(-1)
            sigma_minus = model.synthesizer.forward_points(
                triplane, pts_minus.unsqueeze(0), chunk_size=2**18
            )['sigma'].squeeze(0).squeeze(-1)

        grads.append((sigma_plus - sigma_minus) / (2 * eps))

    normals = torch.stack(grads, dim=-1).cpu().numpy()  # (P, 3)
    # 归一化
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normals = normals / norms

    # 视线方向：从表面点指向相机
    view_dirs = camera_pos[None, :] - surface_points  # (P, 3)
    view_norms = np.linalg.norm(view_dirs, axis=-1, keepdims=True)
    view_dirs = view_dirs / np.clip(view_norms, 1e-8, None)

    # dot(normal, view_dir) > 0 → 面朝相机 → 可见
    cos_angle = np.sum(normals * view_dirs, axis=-1)
    visible_mask = cos_angle > 0

    print(f"    [normal法] cos_angle 分布: mean={cos_angle.mean():.4f}, min={cos_angle.min():.4f}, max={cos_angle.max():.4f}")
    print(f"    [normal法] 可见: {visible_mask.sum()}, 遮挡: {(~visible_mask).sum()}")

    return visible_mask


# ─────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────

def plot_visibility_error_comparison(
    error_visible, error_occluded, output_dir, uid,
):
    """绘制可见 vs 遮挡区域的误差分布对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- 子图 1: 重叠直方图 ---
    ax = axes[0]
    bins = np.linspace(0, max(error_visible.max(), error_occluded.max()) * 1.05, 80)
    ax.hist(error_visible, bins=bins, alpha=0.6, color='#2196F3', label=f'Visible (n={len(error_visible)})', density=True)
    ax.hist(error_occluded, bins=bins, alpha=0.6, color='#F44336', label=f'Occluded (n={len(error_occluded)})', density=True)
    ax.axvline(error_visible.mean(), color='#1565C0', linestyle='--', linewidth=2,
               label=f'Vis. Mean={error_visible.mean():.4f}')
    ax.axvline(error_occluded.mean(), color='#C62828', linestyle='--', linewidth=2,
               label=f'Occ. Mean={error_occluded.mean():.4f}')
    ax.set_xlabel('Per-point RGB Error (L2)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Error Distribution: Visible vs Occluded', fontsize=13)
    ax.legend(fontsize=10)

    # --- 子图 2: 箱线图 ---
    ax = axes[1]
    bp = ax.boxplot(
        [error_visible, error_occluded],
        labels=['Visible', 'Occluded'],
        patch_artist=True,
        widths=0.5,
    )
    bp['boxes'][0].set_facecolor('#BBDEFB')
    bp['boxes'][1].set_facecolor('#FFCDD2')
    ax.set_ylabel('Per-point RGB Error (L2)', fontsize=12)
    ax.set_title('Error Boxplot', fontsize=13)

    # --- 子图 3: 统计表格 ---
    ax = axes[2]
    ax.axis('off')
    stats = [
        ['', 'Visible', 'Occluded', 'Ratio (Occ/Vis)'],
        ['Count', f'{len(error_visible)}', f'{len(error_occluded)}', '-'],
        ['Mean', f'{error_visible.mean():.4f}', f'{error_occluded.mean():.4f}',
         f'{error_occluded.mean() / (error_visible.mean() + 1e-8):.2f}x'],
        ['Median', f'{np.median(error_visible):.4f}', f'{np.median(error_occluded):.4f}',
         f'{np.median(error_occluded) / (np.median(error_visible) + 1e-8):.2f}x'],
        ['Std', f'{error_visible.std():.4f}', f'{error_occluded.std():.4f}', '-'],
        ['P90', f'{np.percentile(error_visible, 90):.4f}', f'{np.percentile(error_occluded, 90):.4f}',
         f'{np.percentile(error_occluded, 90) / (np.percentile(error_visible, 90) + 1e-8):.2f}x'],
        ['Max', f'{error_visible.max():.4f}', f'{error_occluded.max():.4f}', '-'],
    ]
    table = ax.table(
        cellText=stats, loc='center', cellLoc='center',
        colWidths=[0.2, 0.25, 0.25, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    # 表头加粗
    for j in range(4):
        table[0, j].set_text_props(fontweight='bold')
    ax.set_title('Error Statistics', fontsize=13, pad=20)

    plt.suptitle(f'Visibility-Aware Error Analysis\n{uid}', fontsize=15, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{uid}_visibility_error.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存可见性分析图: {save_path}")


def plot_visibility_cloud(surface_points, visible_mask, per_point_error, output_dir, uid):
    """保存可见性标注的点云 (蓝=可见, 红=遮挡) 和误差着色点云"""

    def save_ply(path, verts, colors):
        with open(path, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(verts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for v, c in zip(verts, colors):
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

    # 可见性点云：蓝=可见, 红=遮挡
    vis_colors = np.zeros((len(surface_points), 3), dtype=np.uint8)
    vis_colors[visible_mask] = [0, 100, 255]   # 蓝
    vis_colors[~visible_mask] = [255, 50, 50]   # 红
    vis_ply_path = os.path.join(output_dir, f'{uid}_visibility_mask.ply')
    save_ply(vis_ply_path, surface_points, vis_colors)
    print(f"  保存可见性点云: {vis_ply_path} (蓝=可见, 红=遮挡)")

    # 可见区域误差点云
    cmap = plt.cm.get_cmap('coolwarm')
    if per_point_error.max() > 0:
        normed = per_point_error / per_point_error.max()
    else:
        normed = per_point_error
    error_colors = (cmap(normed)[:, :3] * 255).astype(np.uint8)
    error_ply_path = os.path.join(output_dir, f'{uid}_vis_error_cloud.ply')
    save_ply(error_ply_path, surface_points, error_colors)
    print(f"  保存误差点云: {error_ply_path}")


def plot_per_view_error(
    model, t_pred, t_gt, poses, intrinsics,
    render_size, source_view, output_dir, uid, device,
):
    """逐视角渲染 T_pred vs T_GT，画 error vs view-angle 曲线"""
    num_views = poses.shape[0]
    normalized_poses = camera_normalization_objaverse('auto', poses)
    intrinsics_batch = intrinsics.unsqueeze(0).repeat(num_views, 1, 1)
    render_cameras = build_camera_standard(normalized_poses, intrinsics_batch)

    # 源视角的相机位置
    source_c2w = render_cameras[source_view, :16].reshape(4, 4)
    source_pos = source_c2w[:3, 3]

    per_view_l1 = []
    angles = []

    for vi in range(num_views):
        cam = render_cameras[vi:vi+1].to(device)

        rgb_pred, _ = render_from_triplane(model, t_pred, cam, render_size, device)
        rgb_gt, _ = render_from_triplane(model, t_gt, cam, render_size, device)

        l1 = (rgb_pred - rgb_gt).abs().mean().item()
        per_view_l1.append(l1)

        # 计算该视角与源视角的夹角
        vi_c2w = render_cameras[vi, :16].reshape(4, 4)
        vi_pos = vi_c2w[:3, 3]
        cos_angle = F.cosine_similarity(source_pos.unsqueeze(0), vi_pos.unsqueeze(0)).item()
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        angles.append(angle_deg)

    per_view_l1 = np.array(per_view_l1)
    angles = np.array(angles)

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 子图 1: error vs view index (标记源视角)
    ax = axes[0]
    colors = ['#F44336' if i != source_view else '#4CAF50' for i in range(num_views)]
    ax.bar(range(num_views), per_view_l1, color=colors, alpha=0.8)
    ax.set_xlabel('View Index', fontsize=12)
    ax.set_ylabel('L1 Error (T_pred vs T_GT render)', fontsize=12)
    ax.set_title('Per-View Rendering Error', fontsize=13)
    ax.annotate('Source View', xy=(source_view, per_view_l1[source_view]),
                fontsize=10, ha='center',
                xytext=(source_view, per_view_l1[source_view] + per_view_l1.max() * 0.1),
                arrowprops=dict(arrowstyle='->', color='green'))

    # 子图 2: error vs angle to source view
    ax = axes[1]
    ax.scatter(angles, per_view_l1, c=angles, cmap='coolwarm', s=60, edgecolors='black', linewidths=0.5)
    # 拟合趋势线
    z = np.polyfit(angles, per_view_l1, 1)
    p = np.poly1d(z)
    angle_sorted = np.sort(angles)
    ax.plot(angle_sorted, p(angle_sorted), 'k--', alpha=0.5, label=f'Linear fit (slope={z[0]:.5f})')
    ax.set_xlabel('Angle to Source View (degrees)', fontsize=12)
    ax.set_ylabel('L1 Error', fontsize=12)
    ax.set_title('Error vs Angular Distance from Source', fontsize=13)
    ax.legend(fontsize=10)

    # 计算相关系数
    corr = np.corrcoef(angles, per_view_l1)[0, 1]
    ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Per-View Error Analysis\n{uid}', fontsize=15, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{uid}_per_view_error.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存逐视角误差图: {save_path}")

    # 保存最差的4个视角的对比图
    worst_views = np.argsort(per_view_l1)[-4:][::-1]
    best_views = np.argsort(per_view_l1)[:4]
    save_view_comparison(model, t_pred, t_gt, render_cameras,
                         worst_views, best_views, render_size, output_dir, uid, device)

    return per_view_l1, angles


def save_view_comparison(
    model, t_pred, t_gt, render_cameras,
    worst_views, best_views, render_size, output_dir, uid, device,
):
    """保存最好/最差视角的 pred vs gt 对比图"""
    fig, axes = plt.subplots(4, 6, figsize=(24, 16))

    for row, (views, label) in enumerate([
        (best_views, 'BEST'),
        (worst_views, 'WORST'),
    ]):
        for col, vi in enumerate(views):
            cam = render_cameras[vi:vi+1].to(device)
            rgb_pred, _ = render_from_triplane(model, t_pred, cam, render_size, device)
            rgb_gt, _ = render_from_triplane(model, t_gt, cam, render_size, device)
            diff = (rgb_pred - rgb_gt).abs()
            l1 = diff.mean().item()

            r = row * 2
            # pred
            axes[r, col].imshow(rgb_pred.permute(1, 2, 0).clamp(0, 1).numpy())
            axes[r, col].set_title(f'{label} #{col+1} View {vi}\nT_pred', fontsize=9)
            axes[r, col].axis('off')
            # gt
            axes[r+1, col].imshow(rgb_gt.permute(1, 2, 0).clamp(0, 1).numpy())
            axes[r+1, col].set_title(f'T_GT  (L1={l1:.4f})', fontsize=9)
            axes[r+1, col].axis('off')

        # 第5/6列放 diff map
        for col, vi in enumerate(views[:2]):
            cam = render_cameras[vi:vi+1].to(device)
            rgb_pred, _ = render_from_triplane(model, t_pred, cam, render_size, device)
            rgb_gt, _ = render_from_triplane(model, t_gt, cam, render_size, device)
            diff = (rgb_pred - rgb_gt).abs().mean(dim=0)  # (H, W)

            r = row * 2
            axes[r, 4 + col].imshow(diff.numpy(), cmap='hot', vmin=0, vmax=0.3)
            axes[r, 4 + col].set_title(f'Error Map View {vi}', fontsize=9)
            axes[r, 4 + col].axis('off')
            axes[r+1, 4 + col].axis('off')

    plt.suptitle(f'Best vs Worst View Comparison\n{uid}', fontsize=15, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{uid}_view_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存视角对比图: {save_path}")


# ─────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="可见性感知误差分析")
    parser.add_argument("--triplane_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="渲染数据目录 (包含 rgba/ pose/ intrinsics.npy)")
    parser.add_argument("--model_name", type=str, default="zxhezexin/openlrm-mix-base-1.1")
    parser.add_argument("--infer_config", type=str, default="./configs/infer-b.yaml")
    parser.add_argument("--output_dir", type=str, default="./exps/visibility_analysis")
    parser.add_argument("--grid_size", type=int, default=128)
    parser.add_argument("--sigma_threshold", type=float, default=1.0)
    parser.add_argument("--depth_tolerance", type=float, default=0.05,
                        help="深度缓冲可见性判定的容差")
    parser.add_argument("--source_view", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.infer_config)

    # ── 加载数据 ──
    print("[1/6] 加载 triplane 数据...")
    data = torch.load(args.triplane_path, map_location='cpu')
    t_pred = data['t_pred']
    t_gt = data['t_gt']
    source_view = data.get('source_view', args.source_view)
    uid = os.path.basename(args.triplane_path).replace('_triplane.pt', '')
    print(f"  UID: {uid}, source_view: {source_view}")

    print("[2/6] 加载模型...")
    from openlrm.models import model_dict
    hf_model_cls = wrap_model_hub(model_dict['lrm'])
    model = hf_model_cls.from_pretrained(args.model_name).to(device)
    model.eval()

    print("[3/6] 加载渲染数据 (相机参数)...")
    images, poses, intrinsics = load_render_data(args.data_dir)
    render_size = cfg.render_size

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 采样表面点并计算误差 ──
    print("[4/6] 采样 3D 表面点并计算逐点误差...")
    points, rgb_gt, sigma_gt = query_triplane_features(model, t_gt, args.grid_size, device)
    _, rgb_pred, _ = query_triplane_features(model, t_pred, args.grid_size, device)

    sigma_vals = sigma_gt.squeeze(-1)
    surface_mask = sigma_vals > args.sigma_threshold
    num_surface = surface_mask.sum().item()
    print(f"  表面点: {num_surface} / {args.grid_size**3}")

    if num_surface == 0:
        args.sigma_threshold = sigma_vals.quantile(0.9).item()
        surface_mask = sigma_vals > args.sigma_threshold
        num_surface = surface_mask.sum().item()
        print(f"  自动降低阈值至 {args.sigma_threshold:.2f}, 表面点: {num_surface}")

    surface_points = points[surface_mask].numpy()
    surface_rgb_gt = rgb_gt[surface_mask].numpy()
    surface_rgb_pred = rgb_pred[surface_mask].numpy()
    per_point_error = np.linalg.norm(surface_rgb_pred - surface_rgb_gt, axis=-1)

    # ── 可见性判定 ──
    print("[5/6] 计算可见性...")
    source_cam = build_source_camera(poses, intrinsics, source_view, device)
    _, depth_map = render_from_triplane(model, t_gt, source_cam, render_size, device)
    print(f"  深度图范围: [{depth_map.min():.3f}, {depth_map.max():.3f}]")

    # 方法 1：深度缓冲法
    print("  尝试深度缓冲法...")
    visible_mask = compute_visibility_depthbuffer(
        surface_points, source_cam.squeeze(0).cpu(),
        depth_map, render_size,
        depth_tolerance=args.depth_tolerance,
    )

    # 方法 2：法线法（作为备用或验证）
    if visible_mask is None or visible_mask.sum() == 0:
        print("  深度缓冲法失败，使用法线法...")
        visible_mask = compute_visibility_normal(
            model, t_gt, surface_points,
            source_cam.squeeze(0).cpu(), device,
        )
    else:
        # 同时跑法线法作为交叉验证
        print("  同时运行法线法交叉验证...")
        visible_mask_normal = compute_visibility_normal(
            model, t_gt, surface_points,
            source_cam.squeeze(0).cpu(), device,
        )
        agreement = (visible_mask == visible_mask_normal).mean()
        print(f"  两种方法一致率: {agreement:.1%}")

    num_visible = visible_mask.sum()
    num_occluded = (~visible_mask).sum()
    print(f"  最终: 可见点 {num_visible}, 遮挡点 {num_occluded}")

    error_visible = per_point_error[visible_mask]
    error_occluded = per_point_error[~visible_mask]

    # ── 打印统计 ──
    print(f"\n{'='*50}")
    print(f"  可见性感知误差统计")
    print(f"{'='*50}")
    if len(error_visible) > 0:
        print(f"  可见区域 ({num_visible} 点):")
        print(f"    Mean: {error_visible.mean():.6f}")
        print(f"    Std:  {error_visible.std():.6f}")
        print(f"    P90:  {np.percentile(error_visible, 90):.6f}")
    else:
        print(f"  可见区域: 无点（请检查可见性判定）")
    if len(error_occluded) > 0:
        print(f"  遮挡区域 ({num_occluded} 点):")
        print(f"    Mean: {error_occluded.mean():.6f}")
        print(f"    Std:  {error_occluded.std():.6f}")
        print(f"    P90:  {np.percentile(error_occluded, 90):.6f}")
    else:
        print(f"  遮挡区域: 无点")
    if len(error_visible) > 0 and len(error_occluded) > 0:
        print(f"  遮挡/可见 Mean 比值: {error_occluded.mean() / (error_visible.mean() + 1e-8):.2f}x")
    print(f"{'='*50}")

    # ── 生成可视化 ──
    print("[6/6] 生成可视化...")
    plot_visibility_error_comparison(error_visible, error_occluded, args.output_dir, uid)
    plot_visibility_cloud(surface_points, visible_mask, per_point_error, args.output_dir, uid)
    plot_per_view_error(
        model, t_pred, t_gt, poses, intrinsics,
        render_size, source_view, args.output_dir, uid, device,
    )

    print("\n完成！")


if __name__ == "__main__":
    main()
