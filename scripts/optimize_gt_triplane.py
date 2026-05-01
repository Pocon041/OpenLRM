"""
GT Triplane 优化脚本

原理：冻结预训练模型的 OSGDecoder + ImportanceRenderer，
将 triplane 特征张量作为可学习参数，用多视角渲染图做监督优化。
优化完成后保存 T_GT 和 T_pred 用于后续残差分析。

用法：
    python scripts/optimize_gt_triplane.py \
        --data_dir <渲染数据目录，包含 rgba/ pose/ intrinsics.npy> \
        --model_name zxhezexin/openlrm-mix-base-1.1 \
        --infer_config ./configs/infer-b.yaml \
        --output_dir ./exps/gt_triplane \
        --num_iters 2000 \
        --lr 0.01
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
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


def load_render_data(data_dir, num_views=32):
    """加载渲染数据：RGBA 图像、相机位姿、内参"""
    rgba_dir = os.path.join(data_dir, 'rgba')
    pose_dir = os.path.join(data_dir, 'pose')
    intrinsics_path = os.path.join(data_dir, 'intrinsics.npy')

    intrinsics = torch.from_numpy(np.load(intrinsics_path)).float()

    images = []
    poses = []
    for i in range(num_views):
        # 加载 RGBA 图像
        img_path = os.path.join(rgba_dir, f'{i:03d}.png')
        img = np.array(Image.open(img_path)).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # (4, H, W)
        # RGBA -> RGB (白色背景合成)
        rgb = img_tensor[:3] * img_tensor[3:4] + (1 - img_tensor[3:4])
        images.append(rgb)

        # 加载位姿 (3x4 C2W)
        pose = np.load(os.path.join(pose_dir, f'{i:03d}.npy'))
        poses.append(torch.from_numpy(pose).float())

    images = torch.stack(images, dim=0)  # (V, 3, H, W)
    poses = torch.stack(poses, dim=0)    # (V, 3, 4)
    return images, poses, intrinsics


def get_predicted_triplane(model, source_image, source_cam_dist, source_size, device):
    """通过模型单图前向传播获取 T_pred"""
    # 准备输入图像
    image = source_image.unsqueeze(0).to(device)  # (1, 3, H, W)
    image = F.interpolate(image, size=(source_size, source_size), mode='bicubic', align_corners=True)
    image = torch.clamp(image, 0, 1)

    # 构建源相机 (模型推理时使用的默认相机)
    canonical_camera_extrinsics = torch.tensor([[
        [1, 0, 0, 0],
        [0, 0, -1, -source_cam_dist],
        [0, 1, 0, 0],
    ]], dtype=torch.float32, device=device)
    canonical_camera_intrinsics = create_intrinsics(
        f=0.75, c=0.5, device=device,
    ).unsqueeze(0)
    source_camera = build_camera_principle(canonical_camera_extrinsics, canonical_camera_intrinsics)

    with torch.no_grad():
        planes = model.forward_planes(image, source_camera)
    return planes  # (1, 3, D, H, W)


def optimize_gt_triplane(
    model, images, poses, intrinsics,
    render_size, num_iters, lr, device,
    normed_dist_to_center='auto',
    init_triplane=None,
):
    """
    冻结 decoder + renderer，优化 triplane 特征张量以拟合多视角 GT 图像。

    参数：
        model: 预训练的 ModelLRM
        images: (V, 3, H_render, W_render) GT 渲染图
        poses: (V, 3, 4) 相机 C2W 矩阵
        intrinsics: (3, 2) 相机内参
        render_size: 渲染分辨率
        num_iters: 优化迭代次数
        lr: 学习率
        device: 设备
    """
    num_views = images.shape[0]

    # 相机归一化（与训练时一致）
    normalized_poses = camera_normalization_objaverse(normed_dist_to_center, poses)

    # 构建渲染相机参数
    intrinsics_batch = intrinsics.unsqueeze(0).repeat(num_views, 1, 1)
    render_cameras = build_camera_standard(normalized_poses, intrinsics_batch)
    render_cameras = render_cameras.unsqueeze(0).to(device)  # (1, V, D_cam)

    # GT 图像下采样到渲染分辨率
    gt_images = F.interpolate(images, size=(render_size, render_size), mode='bicubic', align_corners=True)
    gt_images = torch.clamp(gt_images, 0, 1).to(device)  # (V, 3, H, W)

    # 初始化可学习的 triplane
    triplane_dim = model.triplane_dim
    triplane_res = model.triplane_high_res
    if init_triplane is not None:
        print("  从提供的 triplane 初始化（T_pred init 模式）")
        triplane_param = init_triplane.clone().to(device)
    else:
        print("  从随机噪声初始化")
        triplane_param = torch.randn(
            1, 3, triplane_dim, triplane_res, triplane_res,
            device=device, dtype=torch.float32,
        ) * 0.1
    triplane_param = torch.nn.Parameter(triplane_param)

    optimizer = torch.optim.Adam([triplane_param], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters, eta_min=lr * 0.01)

    # 冻结模型所有参数（只优化 triplane_param）
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # 渲染参数
    render_anchors = torch.zeros(1, 1, 2, device=device)
    render_resolutions = torch.ones(1, 1, 1, device=device) * render_size
    render_bg_colors = torch.ones(1, 1, 1, device=device)

    pbar = tqdm(range(num_iters), desc="优化 GT Triplane")
    for it in pbar:
        optimizer.zero_grad()

        # 每次迭代随机选几个视角渲染（节省显存）
        batch_views = min(4, num_views)
        view_indices = torch.randperm(num_views)[:batch_views]

        total_loss = 0.0
        for vi in view_indices:
            cam = render_cameras[:, vi:vi+1, :]  # (1, 1, D_cam)
            anchor = render_anchors
            res = render_resolutions
            bg = render_bg_colors

            render_out = model.synthesizer(
                planes=triplane_param,
                cameras=cam,
                anchors=anchor,
                resolutions=res,
                bg_colors=bg,
                region_size=render_size,
            )
            rendered_rgb = render_out['images_rgb'].squeeze(0).squeeze(0)  # (3, H, W)
            gt_rgb = gt_images[vi]  # (3, H, W)

            # L1 + L2 混合损失
            loss = F.l1_loss(rendered_rgb, gt_rgb) + F.mse_loss(rendered_rgb, gt_rgb)
            total_loss = total_loss + loss

        total_loss = total_loss / batch_views
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if it % 100 == 0 or it == num_iters - 1:
            pbar.set_postfix(loss=f"{total_loss.item():.5f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

    return triplane_param.detach()


def main():
    parser = argparse.ArgumentParser(description="GT Triplane 优化")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="渲染数据目录，包含 rgba/ pose/ intrinsics.npy")
    parser.add_argument("--model_name", type=str, default="zxhezexin/openlrm-mix-base-1.1")
    parser.add_argument("--infer_config", type=str, default="./configs/infer-b.yaml")
    parser.add_argument("--output_dir", type=str, default="./exps/gt_triplane")
    parser.add_argument("--num_views", type=int, default=32)
    parser.add_argument("--num_iters", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--source_view", type=int, default=0,
                        help="用哪个视角作为单图输入获取 T_pred")
    parser.add_argument("--init_from_pred", action="store_true",
                        help="从 T_pred 初始化 T_GT 优化（而非随机初始化）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.infer_config)

    # 加载模型
    print("[1/4] 加载预训练模型...")
    from openlrm.models import model_dict
    hf_model_cls = wrap_model_hub(model_dict['lrm'])
    model = hf_model_cls.from_pretrained(args.model_name).to(device)
    model.eval()

    # 加载渲染数据
    print("[2/4] 加载渲染数据...")
    images, poses, intrinsics = load_render_data(args.data_dir, num_views=args.num_views)
    print(f"  加载了 {images.shape[0]} 个视角, 图像分辨率: {images.shape[2]}x{images.shape[3]}")

    # 获取 T_pred（单图预测）
    print("[3/4] 获取单图预测的 T_pred...")
    source_image = images[args.source_view]  # (3, H, W)
    t_pred = get_predicted_triplane(
        model, source_image,
        source_cam_dist=cfg.source_cam_dist,
        source_size=cfg.source_size,
        device=device,
    )
    print(f"  T_pred 形状: {t_pred.shape}")

    # 优化 GT Triplane
    init_mode = "pred" if args.init_from_pred else "random"
    print(f"[4/4] 优化 GT Triplane（多视角监督, init={init_mode}）...")
    t_gt = optimize_gt_triplane(
        model=model,
        images=images,
        poses=poses,
        intrinsics=intrinsics,
        render_size=cfg.render_size,
        num_iters=args.num_iters,
        lr=args.lr,
        device=device,
        init_triplane=t_pred if args.init_from_pred else None,
    )
    print(f"  T_GT 形状: {t_gt.shape}")

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    uid = os.path.basename(args.data_dir.rstrip('/'))
    suffix = "_triplane_predinit.pt" if args.init_from_pred else "_triplane.pt"
    save_path = os.path.join(args.output_dir, f"{uid}{suffix}")
    torch.save({
        't_pred': t_pred.cpu(),
        't_gt': t_gt.cpu(),
        'source_view': args.source_view,
        'data_dir': args.data_dir,
        'model_name': args.model_name,
        'init_mode': init_mode,
    }, save_path)
    print(f"已保存至: {save_path}")

    # 打印基本统计量
    residual = (t_pred.cpu() - t_gt.cpu()).abs()
    print(f"\n===== 残差统计 =====")
    print(f"  全局均值: {residual.mean().item():.6f}")
    print(f"  全局最大: {residual.max().item():.6f}")
    for plane_idx in range(3):
        plane_res = residual[0, plane_idx]
        print(f"  平面 {plane_idx} (XY/XZ/YZ): mean={plane_res.mean().item():.6f}, max={plane_res.max().item():.6f}")


if __name__ == "__main__":
    main()
