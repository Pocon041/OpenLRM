"""
Proof-of-Concept: Frequency-Weighted Rendering Loss Fine-tuning

验证频率加权渲染 loss 是否能让 encoder 输出更好的低频结构。支持 train/val split。

对比两种微调方案（从同一预训练 checkpoint 出发）：
  A. Standard loss: pixel MSE
  B. Freq-weighted loss: 对渲染图像 FFT 后低频误差加大权重

评估：微调后 encoder 在每个样本上的渲染质量（训练视角 + 新视角）

用法：
    python scripts/finetune_freq_loss.py --output_dir ./exps/finetune_freq
"""

import os
import sys
import argparse
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def get_source_camera(source_cam_dist, device):
    canonical_camera_extrinsics = torch.tensor([[
        [1, 0, 0, 0],
        [0, 0, -1, -source_cam_dist],
        [0, 1, 0, 0],
    ]], dtype=torch.float32, device=device)
    canonical_camera_intrinsics = create_intrinsics(f=0.75, c=0.5, device=device).unsqueeze(0)
    return build_camera_principle(canonical_camera_extrinsics, canonical_camera_intrinsics)


def freq_weighted_mse(pred, gt, low_freq_weight=5.0, cutoff_ratio=0.3):
    """
    频率加权 MSE loss：对渲染图像做 FFT，低频误差加大权重

    pred, gt: (N, C, H, W)
    """
    err = pred - gt  # (N, C, H, W)
    N, C, H, W = err.shape

    # FFT
    err_fft = torch.fft.fft2(err)

    # 构建频率权重掩码
    fy = torch.fft.fftfreq(H, device=err.device).unsqueeze(1)
    fx = torch.fft.fftfreq(W, device=err.device).unsqueeze(0)
    freq_radius = (fx**2 + fy**2).sqrt()  # (H, W)

    # 低频区域权重放大
    weight = torch.ones_like(freq_radius)
    low_mask = freq_radius < cutoff_ratio
    weight[low_mask] = low_freq_weight

    # 加权频域误差 → 反变换回空域 → MSE
    weighted_fft = err_fft * weight.unsqueeze(0).unsqueeze(0)
    weighted_err = torch.fft.ifft2(weighted_fft).real

    return weighted_err.pow(2).mean()


def standard_mse(pred, gt):
    return F.mse_loss(pred, gt)


def render_views(model, source_image, source_camera, render_cameras, render_size, device):
    """单图推理 + 渲染多个视角"""
    planes = model.forward_planes(source_image, source_camera)
    rendered = []
    for vi in range(render_cameras.shape[0]):
        cam = render_cameras[vi:vi+1].to(device)
        out = model.synthesizer(
            planes=planes,
            cameras=cam.unsqueeze(0),
            anchors=torch.zeros(1, 1, 2, device=device),
            resolutions=torch.ones(1, 1, 1, device=device) * render_size,
            bg_colors=torch.ones(1, 1, 1, device=device),
            region_size=render_size,
        )
        rendered.append(out['images_rgb'].squeeze(0).squeeze(0))  # (C, H, W)
    return torch.stack(rendered), planes  # (M, C, H, W), (1, 3, D, H, W)


def evaluate_model(model, all_data, source_cam, render_size, device, eval_views):
    """评估模型在所有样本的 eval views 上的 L1"""
    model.eval()
    all_l1s = []
    with torch.no_grad():
        for uid, (images, cameras) in all_data.items():
            src_img = images[0:1].to(device)
            src_img_resized = F.interpolate(src_img, size=(source_size_global, source_size_global),
                                            mode='bicubic', align_corners=True).clamp(0, 1)
            for evi in eval_views:
                cam = cameras[evi:evi+1].to(device)
                planes = model.forward_planes(src_img_resized, source_cam)
                out = model.synthesizer(
                    planes=planes,
                    cameras=cam.unsqueeze(0),
                    anchors=torch.zeros(1, 1, 2, device=device),
                    resolutions=torch.ones(1, 1, 1, device=device) * render_size,
                    bg_colors=torch.ones(1, 1, 1, device=device),
                    region_size=render_size,
                )
                pred = out['images_rgb'].squeeze()
                gt = F.interpolate(images[evi:evi+1], size=(render_size, render_size),
                                   mode='bilinear', align_corners=False).squeeze().to(device)
                all_l1s.append((pred - gt).abs().mean().item())
    return np.mean(all_l1s)


source_size_global = 224  # will be set in main


def main():
    global source_size_global

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/rendered")
    parser.add_argument("--model_name", type=str, default="zxhezexin/openlrm-mix-base-1.1")
    parser.add_argument("--infer_config", type=str, default="./configs/infer-b.yaml")
    parser.add_argument("--output_dir", type=str, default="./exps/finetune_freq")
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--low_freq_weight", type=float, default=5.0)
    parser.add_argument("--cutoff_ratio", type=float, default=0.3)
    parser.add_argument("--num_train_views", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.infer_config)
    render_size = cfg.render_size
    source_size_global = cfg.source_size

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 train/val split
    meta_dir = os.path.join(args.data_root, 'meta')
    if os.path.exists(os.path.join(meta_dir, 'train_uids.json')):
        with open(os.path.join(meta_dir, 'train_uids.json')) as f:
            train_uids = json.load(f)
        with open(os.path.join(meta_dir, 'val_uids.json')) as f:
            val_uids = json.load(f)
        uids = train_uids + val_uids
        print(f"Train: {len(train_uids)}, Val: {len(val_uids)}, Total: {len(uids)}")
    else:
        uids = [d for d in os.listdir(args.data_root)
                if os.path.isdir(os.path.join(args.data_root, d)) and d != 'meta']
        train_uids = uids
        val_uids = uids
        print(f"No split found, using all {len(uids)} for both train and val")

    # 加载所有渲染数据
    print("[1/5] 加载数据...")
    all_data = {}
    for uid in uids:
        images, poses, intrinsics = load_render_data(os.path.join(args.data_root, uid))
        normalized_poses = camera_normalization_objaverse('auto', poses)
        intrinsics_batch = intrinsics.unsqueeze(0).repeat(poses.shape[0], 1, 1)
        render_cameras = build_camera_standard(normalized_poses, intrinsics_batch)
        all_data[uid] = (images, render_cameras)

    # 训练/评估视角
    total_views = min(32, min(len(imgs) for imgs, _ in all_data.values()))
    train_step = max(1, total_views // args.num_train_views)
    train_views = list(range(0, total_views, train_step))[:args.num_train_views]
    remaining = sorted(set(range(total_views)) - set(train_views))
    eval_views = remaining[::max(1, len(remaining) // 6)][:6]
    print(f"  训练视角: {train_views}")
    print(f"  评估视角: {eval_views}")

    source_cam = get_source_camera(cfg.source_cam_dist, device)

    # 加载两份模型
    print("[2/5] 加载模型（2份）...")
    from openlrm.models import model_dict
    hf_model_cls = wrap_model_hub(model_dict['lrm'])

    model_std = hf_model_cls.from_pretrained(args.model_name).to(device)
    model_freq = hf_model_cls.from_pretrained(args.model_name).to(device)

    # 评估 baseline
    val_data = {uid: all_data[uid] for uid in val_uids if uid in all_data}
    train_data = {uid: all_data[uid] for uid in train_uids if uid in all_data}
    baseline_l1 = evaluate_model(model_std, val_data, source_cam, render_size, device, eval_views)
    print(f"  Baseline eval L1: {baseline_l1:.6f}")

    # 设置微调：冻结 encoder (DINOv2), 只训练 transformer + upsampler
    for model in [model_std, model_freq]:
        model.train()
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        # pos_embed, transformer, upsampler 可训练
        model.pos_embed.requires_grad_(True)
        for p in model.transformer.parameters():
            p.requires_grad_(True)
        for p in model.upsampler.parameters():
            p.requires_grad_(True)

    trainable_std = [p for p in model_std.parameters() if p.requires_grad]
    trainable_freq = [p for p in model_freq.parameters() if p.requires_grad]
    print(f"  可训练参数: {sum(p.numel() for p in trainable_std):,}")

    opt_std = torch.optim.AdamW(trainable_std, lr=args.lr, weight_decay=0.01)
    opt_freq = torch.optim.AdamW(trainable_freq, lr=args.lr, weight_decay=0.01)

    # ─── 微调循环 ───
    print(f"[3/5] 微调 ({args.num_steps} steps)...")
    uid_list = list(train_data.keys())
    hist_std = {'step': [], 'train_loss': [], 'eval_l1': []}
    hist_freq = {'step': [], 'train_loss': [], 'eval_l1': []}

    pbar = tqdm(range(args.num_steps), desc='Fine-tuning')
    for step in pbar:
        # 随机选样本和训练视角
        uid = uid_list[step % len(uid_list)]
        images, cameras = all_data[uid]
        vi = train_views[step % len(train_views)]

        src_img = images[0:1].to(device)
        src_img = F.interpolate(src_img, size=(source_size_global, source_size_global),
                                mode='bicubic', align_corners=True).clamp(0, 1)
        gt_img = F.interpolate(images[vi:vi+1], size=(render_size, render_size),
                               mode='bilinear', align_corners=False).to(device)
        cam = cameras[vi:vi+1].to(device)

        # ── Standard MSE ──
        opt_std.zero_grad()
        planes_s = model_std.forward_planes(src_img, source_cam)
        out_s = model_std.synthesizer(
            planes=planes_s, cameras=cam.unsqueeze(0),
            anchors=torch.zeros(1, 1, 2, device=device),
            resolutions=torch.ones(1, 1, 1, device=device) * render_size,
            bg_colors=torch.ones(1, 1, 1, device=device),
            region_size=render_size,
        )
        pred_s = out_s['images_rgb'].squeeze(0)
        loss_s = standard_mse(pred_s, gt_img)
        loss_s.backward()
        torch.nn.utils.clip_grad_norm_(trainable_std, 1.0)
        opt_std.step()

        # ── Freq-weighted MSE ──
        opt_freq.zero_grad()
        planes_f = model_freq.forward_planes(src_img, source_cam)
        out_f = model_freq.synthesizer(
            planes=planes_f, cameras=cam.unsqueeze(0),
            anchors=torch.zeros(1, 1, 2, device=device),
            resolutions=torch.ones(1, 1, 1, device=device) * render_size,
            bg_colors=torch.ones(1, 1, 1, device=device),
            region_size=render_size,
        )
        pred_f = out_f['images_rgb'].squeeze(0)
        loss_f = freq_weighted_mse(pred_f, gt_img,
                                   low_freq_weight=args.low_freq_weight,
                                   cutoff_ratio=args.cutoff_ratio)
        loss_f.backward()
        torch.nn.utils.clip_grad_norm_(trainable_freq, 1.0)
        opt_freq.step()

        pbar.set_postfix(std=f'{loss_s.item():.5f}', freq=f'{loss_f.item():.5f}')

        # 定期评估
        if step % 20 == 0 or step == args.num_steps - 1:
            eval_std = evaluate_model(model_std, val_data, source_cam, render_size, device, eval_views)
            eval_freq = evaluate_model(model_freq, val_data, source_cam, render_size, device, eval_views)
            model_std.train()
            model_freq.train()
            for p in model_std.encoder.parameters():
                p.requires_grad_(False)
            for p in model_freq.encoder.parameters():
                p.requires_grad_(False)

            hist_std['step'].append(step)
            hist_std['train_loss'].append(loss_s.item())
            hist_std['eval_l1'].append(eval_std)
            hist_freq['step'].append(step)
            hist_freq['train_loss'].append(loss_f.item())
            hist_freq['eval_l1'].append(eval_freq)

    # ─── 最终评估 + 渲染对比 ───
    print("[4/5] 最终渲染对比...")
    model_std.eval()
    model_freq.eval()

    final_std_l1 = evaluate_model(model_std, val_data, source_cam, render_size, device, eval_views)
    final_freq_l1 = evaluate_model(model_freq, val_data, source_cam, render_size, device, eval_views)

    print(f"\n  {'方法':<25} {'Eval L1':>10} {'改善':>10}")
    print(f"  {'':-<25} {'':-<10} {'':-<10}")
    print(f"  {'Baseline (no finetune)':<25} {baseline_l1:>10.6f} {'':>10}")
    print(f"  {'Standard MSE finetune':<25} {final_std_l1:>10.6f} {(1-final_std_l1/baseline_l1)*100:>+9.1f}%")
    print(f"  {'Freq-weighted finetune':<25} {final_freq_l1:>10.6f} {(1-final_freq_l1/baseline_l1)*100:>+9.1f}%")

    # 选一个样本做渲染对比
    vis_uid = uid_list[0]
    vis_images, vis_cameras = all_data[vis_uid]
    vis_views = eval_views[:4]

    all_renders = {'Baseline': [], 'Std-FT': [], 'Freq-FT': [], 'GT': []}

    # 需要原始模型做 baseline
    model_base = hf_model_cls.from_pretrained(args.model_name).to(device)
    model_base.eval()

    with torch.no_grad():
        src_img = vis_images[0:1].to(device)
        src_img = F.interpolate(src_img, size=(source_size_global, source_size_global),
                                mode='bicubic', align_corners=True).clamp(0, 1)
        for vi in vis_views:
            cam = vis_cameras[vi:vi+1].to(device)
            gt = F.interpolate(vis_images[vi:vi+1], size=(render_size, render_size),
                               mode='bilinear', align_corners=False).squeeze(0)
            all_renders['GT'].append(gt.cpu())

            for model_obj, key in [(model_base, 'Baseline'), (model_std, 'Std-FT'), (model_freq, 'Freq-FT')]:
                planes = model_obj.forward_planes(src_img, source_cam)
                out = model_obj.synthesizer(
                    planes=planes, cameras=cam.unsqueeze(0),
                    anchors=torch.zeros(1, 1, 2, device=device),
                    resolutions=torch.ones(1, 1, 1, device=device) * render_size,
                    bg_colors=torch.ones(1, 1, 1, device=device),
                    region_size=render_size,
                )
                all_renders[key].append(out['images_rgb'].squeeze().cpu())

    del model_base
    torch.cuda.empty_cache()

    # ─── 图 1：渲染对比 ───
    print("[5/5] 保存可视化...")
    n_views = len(vis_views)
    fig, axes = plt.subplots(4, n_views, figsize=(4*n_views, 16))
    method_names = ['Baseline', 'Std-FT', 'Freq-FT', 'GT']
    for row, m in enumerate(method_names):
        for col in range(n_views):
            img = all_renders[m][col]
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0)
            axes[row, col].imshow(img.clamp(0, 1).numpy())
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(f'View {vis_views[col]}', fontsize=11)
        # 在每行左侧标注方法名
        axes[row, 0].text(-0.15, 0.5, m, transform=axes[row, 0].transAxes,
                          fontsize=14, fontweight='bold', va='center', ha='right')
    plt.suptitle(f'Fine-tuning Comparison: Standard vs Freq-Weighted\n{vis_uid}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save1 = os.path.join(args.output_dir, f'render_comparison.png')
    plt.savefig(save1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save1}")

    # ─── 图 2：收敛曲线 ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(hist_std['step'], hist_std['train_loss'], 'b-', label='Standard MSE', linewidth=2)
    ax.plot(hist_freq['step'], hist_freq['train_loss'], 'r-', label='Freq-Weighted', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Train Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(hist_std['step'], hist_std['eval_l1'], 'b-', label='Standard MSE', linewidth=2)
    ax.plot(hist_freq['step'], hist_freq['eval_l1'], 'r-', label='Freq-Weighted', linewidth=2)
    ax.axhline(baseline_l1, color='gray', linestyle='--', alpha=0.7, label=f'Baseline ({baseline_l1:.4f})')
    ax.set_xlabel('Step')
    ax.set_ylabel('Eval L1 (novel views)')
    ax.set_title('Novel View Quality')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('Fine-tuning Convergence: Standard vs Freq-Weighted',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save2 = os.path.join(args.output_dir, f'convergence.png')
    plt.savefig(save2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save2}")

    # 保存数值结果
    results = {
        'baseline_l1': baseline_l1,
        'std_ft_l1': final_std_l1,
        'freq_ft_l1': final_freq_l1,
        'std_improvement': (1 - final_std_l1 / baseline_l1) * 100,
        'freq_improvement': (1 - final_freq_l1 / baseline_l1) * 100,
        'num_samples': len(uids),
        'num_steps': args.num_steps,
        'lr': args.lr,
        'low_freq_weight': args.low_freq_weight,
        'cutoff_ratio': args.cutoff_ratio,
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  完成！")


if __name__ == "__main__":
    main()
