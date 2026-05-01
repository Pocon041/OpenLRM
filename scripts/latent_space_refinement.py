"""
Encoder Latent Space Refinement

核心思想：不直接修改 triplane，而是优化 encoder transformer 输出的 token 序列。
upsampler (ConvTranspose2d) 将 tokens 映射为 triplane，天然约束输出在
encoder 学到的 triplane 流形上，避免直接优化 triplane 时的伪影问题。

对比方案：
  1. T_pred（baseline，无优化）
  2. Token-space 优化（proposed）
  3. Triplane 直接优化 + proximity（baseline）

用法：
    python scripts/latent_space_refinement.py \
        --data_dir ./data/rendered/<uid> \
        --output_dir ./exps/latent_refine
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


def get_initial_tokens(model, source_image, source_cam_dist, source_size, device):
    """获取 encoder 输出的初始 tokens（upsampler 之前）"""
    image = source_image.unsqueeze(0).to(device)
    image = F.interpolate(image, size=(source_size, source_size), mode='bicubic', align_corners=True)
    image = torch.clamp(image, 0, 1)
    source_camera = get_source_camera(source_cam_dist, device)

    with torch.no_grad():
        image_feats = model.encoder(image)
        camera_embeddings = model.camera_embedder(source_camera)
        tokens = model.forward_transformer(image_feats, camera_embeddings)

    return tokens  # (1, 3*H*H, D)


def tokens_to_triplane(model, tokens):
    """tokens → triplane via upsampler"""
    return model.reshape_upsample(tokens)


def optimize_tokens(model, tokens_init, images, cameras, render_size, device,
                    num_iters=500, lr=0.005, eval_cameras=None, eval_images=None,
                    eval_every=10, lambda_lpips=0.0, lpips_fn=None):
    """在 token 空间做优化，通过 upsampler 映射到 triplane"""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # 可学习的 tokens
    tokens_param = torch.nn.Parameter(tokens_init.clone().to(device))
    optimizer = torch.optim.Adam([tokens_param], lr=lr)

    num_train_views = images.shape[0]
    history = {'iter': [], 'train_loss': [], 'eval_l1': []}

    pbar = tqdm(range(num_iters), desc='Optimize (token-space)')
    for it in pbar:
        optimizer.zero_grad()

        vi = it % num_train_views
        gt_img = images[vi:vi+1].to(device)
        gt_img = F.interpolate(gt_img, size=(render_size, render_size),
                               mode='bilinear', align_corners=False)
        cam = cameras[vi:vi+1].to(device)

        # tokens → triplane → render
        triplane = tokens_to_triplane(model, tokens_param)
        out = model.synthesizer(
            planes=triplane,
            cameras=cam.unsqueeze(0),
            anchors=torch.zeros(1, 1, 2, device=device),
            resolutions=torch.ones(1, 1, 1, device=device) * render_size,
            bg_colors=torch.ones(1, 1, 1, device=device),
            region_size=render_size,
        )
        pred_img = out['images_rgb'].squeeze(0)
        loss_mse = F.mse_loss(pred_img, gt_img)

        loss_percep = torch.tensor(0.0, device=device)
        if lambda_lpips > 0 and lpips_fn is not None:
            # LPIPS expects [N, M, C, H, W]
            loss_percep = lpips_fn(
                pred_img.unsqueeze(0), gt_img.unsqueeze(0), is_training=False
            )

        loss = loss_mse + lambda_lpips * loss_percep
        loss.backward()
        optimizer.step()

        pbar.set_postfix(mse=f'{loss_mse.item():.5f}', lpips=f'{loss_percep.item():.4f}')

        if it % eval_every == 0 or it == num_iters - 1:
            tp_eval = tokens_to_triplane(model, tokens_param).detach()
            if eval_cameras is not None and len(eval_cameras) > 0:
                eval_l1s = []
                for evi in range(len(eval_cameras)):
                    r = render_single(model, tp_eval, eval_cameras[evi:evi+1], render_size, device)
                    gt = eval_images[evi]
                    gt_r = F.interpolate(gt.unsqueeze(0), size=(render_size, render_size),
                                         mode='bilinear', align_corners=False).squeeze(0)
                    eval_l1s.append((r - gt_r).abs().mean().item())
                eval_l1 = np.mean(eval_l1s)
            else:
                eval_l1 = loss.item()

            history['iter'].append(it)
            history['train_loss'].append(loss.item())
            history['eval_l1'].append(eval_l1)

    final_tp = tokens_to_triplane(model, tokens_param).detach().cpu()
    final_tokens = tokens_param.detach().cpu()
    return final_tp, final_tokens, history


def optimize_triplane_prox(model, triplane_init, images, cameras, render_size, device,
                           num_iters=500, lr=0.005, lambda_prox=1.0,
                           eval_cameras=None, eval_images=None, eval_every=10):
    """Triplane 直接优化 + proximity prior（对照组）"""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    triplane_ref = triplane_init.clone().to(device)
    tp_param = torch.nn.Parameter(triplane_init.clone().to(device))
    optimizer = torch.optim.Adam([tp_param], lr=lr)

    num_train_views = images.shape[0]
    history = {'iter': [], 'train_loss': [], 'eval_l1': []}

    pbar = tqdm(range(num_iters), desc='Optimize (triplane+prox)')
    for it in pbar:
        optimizer.zero_grad()

        vi = it % num_train_views
        gt_img = images[vi:vi+1].to(device)
        gt_img = F.interpolate(gt_img, size=(render_size, render_size),
                               mode='bilinear', align_corners=False)
        cam = cameras[vi:vi+1].to(device)

        out = model.synthesizer(
            planes=tp_param,
            cameras=cam.unsqueeze(0),
            anchors=torch.zeros(1, 1, 2, device=device),
            resolutions=torch.ones(1, 1, 1, device=device) * render_size,
            bg_colors=torch.ones(1, 1, 1, device=device),
            region_size=render_size,
        )
        pred_img = out['images_rgb'].squeeze(0)
        loss_rgb = F.mse_loss(pred_img, gt_img)
        loss_prox = lambda_prox * (tp_param - triplane_ref).pow(2).mean()
        loss = loss_rgb + loss_prox
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=f'{loss_rgb.item():.5f}', prox=f'{loss_prox.item():.5f}')

        if it % eval_every == 0 or it == num_iters - 1:
            if eval_cameras is not None and len(eval_cameras) > 0:
                eval_l1s = []
                for evi in range(len(eval_cameras)):
                    r = render_single(model, tp_param.detach(), eval_cameras[evi:evi+1], render_size, device)
                    gt = eval_images[evi]
                    gt_r = F.interpolate(gt.unsqueeze(0), size=(render_size, render_size),
                                         mode='bilinear', align_corners=False).squeeze(0)
                    eval_l1s.append((r - gt_r).abs().mean().item())
                eval_l1 = np.mean(eval_l1s)
            else:
                eval_l1 = loss_rgb.item()

            history['iter'].append(it)
            history['train_loss'].append(loss_rgb.item())
            history['eval_l1'].append(eval_l1)

    final_tp = tp_param.detach().cpu()
    return final_tp, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="zxhezexin/openlrm-mix-base-1.1")
    parser.add_argument("--infer_config", type=str, default="./configs/infer-b.yaml")
    parser.add_argument("--output_dir", type=str, default="./exps/latent_refine")
    parser.add_argument("--num_iters", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--source_view", type=int, default=0)
    parser.add_argument("--num_train_views", type=int, default=8)
    parser.add_argument("--lambda_prox", type=float, default=1.0,
                        help="Proximity prior for triplane baseline")
    parser.add_argument("--lambda_lpips", type=float, default=1.0,
                        help="LPIPS loss weight for perceptual variant")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.infer_config)
    render_size = cfg.render_size

    uid = os.path.basename(args.data_dir.rstrip('/'))
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print("[1/6] 加载模型...")
    from openlrm.models import model_dict
    hf_model_cls = wrap_model_hub(model_dict['lrm'])
    model = hf_model_cls.from_pretrained(args.model_name).to(device)
    model.eval()

    # 加载数据
    print("[2/6] 加载渲染数据...")
    images, poses, intrinsics = load_render_data(args.data_dir)
    normalized_poses = camera_normalization_objaverse('auto', poses)
    intrinsics_batch = intrinsics.unsqueeze(0).repeat(poses.shape[0], 1, 1)
    render_cameras = build_camera_standard(normalized_poses, intrinsics_batch)

    # 获取初始 tokens 和 T_pred
    print("[3/6] 获取初始 tokens...")
    tokens_init = get_initial_tokens(
        model, images[args.source_view],
        source_cam_dist=cfg.source_cam_dist,
        source_size=cfg.source_size,
        device=device,
    )
    t_pred = tokens_to_triplane(model, tokens_init).detach().cpu()
    print(f"  Tokens shape: {tokens_init.shape}  ({tokens_init.numel()} params)")
    print(f"  Triplane shape: {t_pred.shape}  ({t_pred.numel()} values)")

    # 选择训练/评估视角
    total = poses.shape[0]
    train_step = max(1, total // args.num_train_views)
    train_views = list(range(0, total, train_step))[:args.num_train_views]
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

    # 初始化 LPIPS
    print("[4/7] 初始化 LPIPS...")
    from openlrm.losses.perceptual import LPIPSLoss
    lpips_fn = LPIPSLoss(device=device, prefech=False)

    # Token-space MSE only
    print(f"[5/7] Token-space MSE-only ({args.num_iters} iters)...")
    tp_token_mse, _, hist_token_mse = optimize_tokens(
        model, tokens_init, train_images, train_cameras, render_size, device,
        num_iters=args.num_iters, lr=args.lr,
        eval_cameras=eval_cameras, eval_images=eval_images_list,
        lambda_lpips=0.0,
    )

    # Token-space MSE + LPIPS
    print(f"[6/7] Token-space MSE+LPIPS ({args.num_iters} iters, λ_lpips={args.lambda_lpips})...")
    tp_token_lpips, _, hist_token_lpips = optimize_tokens(
        model, tokens_init, train_images, train_cameras, render_size, device,
        num_iters=args.num_iters, lr=args.lr,
        eval_cameras=eval_cameras, eval_images=eval_images_list,
        lambda_lpips=args.lambda_lpips, lpips_fn=lpips_fn,
    )

    # ─── 渲染所有方案 ───
    print("[7/7] 渲染对比...")
    vis_views = eval_views[:6]
    all_renders = {
        'Pred': [], 'Token+MSE': [], 'Token+LPIPS': [], 'GT': []
    }
    for vi in vis_views:
        cam = render_cameras[vi:vi+1]
        all_renders['Pred'].append(render_single(model, t_pred, cam, render_size, device))
        all_renders['Token+MSE'].append(render_single(model, tp_token_mse, cam, render_size, device))
        all_renders['Token+LPIPS'].append(render_single(model, tp_token_lpips, cam, render_size, device))
        gt_resized = F.interpolate(images[vi].unsqueeze(0), size=(render_size, render_size),
                                   mode='bilinear', align_corners=False).squeeze(0)
        all_renders['GT'].append(gt_resized)

    # 统计
    methods = ['Pred', 'Token+MSE', 'Token+LPIPS']
    l1s = {}
    print(f"\n  {'方法':<20} {'Eval L1':>10} {'改善':>10}")
    print(f"  {'':-<20} {'':-<10} {'':-<10}")
    for m in methods:
        l1 = np.mean([(r - g).abs().mean().item()
                       for r, g in zip(all_renders[m], all_renders['GT'])])
        l1s[m] = l1
        if m == 'Pred':
            print(f"  {m:<20} {l1:>10.6f} {'':>10}")
        else:
            imp = (1 - l1 / l1s['Pred']) * 100
            print(f"  {m:<20} {l1:>10.6f} {imp:>+9.1f}%")

    # Triplane 偏移量
    delta_mse = (tp_token_mse - t_pred).abs().mean().item()
    delta_lpips = (tp_token_lpips - t_pred).abs().mean().item()
    print(f"\n  Triplane 偏移 (from T_pred):")
    print(f"    Token+MSE:    {delta_mse:.4f}")
    print(f"    Token+LPIPS:  {delta_lpips:.4f}")

    # ─── 图 1：渲染对比 ───
    n_views = len(vis_views)
    fig = plt.figure(figsize=(4 * n_views, 16))
    gs = GridSpec(4, n_views, figure=fig, hspace=0.05, wspace=0.05)

    method_names = ['Pred (Encoder)', 'Token+MSE', 'Token+MSE+LPIPS (Ours)', 'GT']
    method_keys = ['Pred', 'Token+MSE', 'Token+LPIPS', 'GT']

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

    plt.suptitle(f'Latent Space Refinement Results\n{uid}',
                 fontsize=14, fontweight='bold', y=1.01)
    save1 = os.path.join(args.output_dir, f'{uid}_render_comparison.png')
    plt.savefig(save1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  保存: {save1}")

    # ─── 图 2：误差热力图 ───
    fig = plt.figure(figsize=(4 * n_views, 12))
    gs = GridSpec(3, n_views, figure=fig, hspace=0.08, wspace=0.05)

    err_methods = ['Pred', 'Token+MSE', 'Token+LPIPS']
    err_labels = ['Pred Error', 'Token+MSE Error', 'Token+LPIPS Error']

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
            im = ax.imshow(all_errs[idx], cmap='hot', vmin=0, vmax=vmax)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(label, fontsize=11, fontweight='bold', rotation=90, labelpad=10)
            if row == 0:
                ax.set_title(f'View {vis_views[col]}', fontsize=11)
            idx += 1

    plt.colorbar(im, ax=fig.axes, fraction=0.02, pad=0.02, label='L1 Error')
    plt.suptitle(f'Error Maps: Token+MSE vs Token+LPIPS\n{uid}', fontsize=14, fontweight='bold')
    save2 = os.path.join(args.output_dir, f'{uid}_error_maps.png')
    plt.savefig(save2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save2}")

    # ─── 图 3：收敛曲线 ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(hist_token_mse['iter'], hist_token_mse['train_loss'], 'b-', label='Token+MSE', linewidth=2)
    ax.plot(hist_token_lpips['iter'], hist_token_lpips['train_loss'], 'r-', label='Token+LPIPS', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Train Loss (MSE)')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')

    ax = axes[1]
    ax.plot(hist_token_mse['iter'], hist_token_mse['eval_l1'], 'b-', label='Token+MSE', linewidth=2)
    ax.plot(hist_token_lpips['iter'], hist_token_lpips['eval_l1'], 'r-', label='Token+LPIPS', linewidth=2)
    pred_l1 = l1s['Pred']
    ax.axhline(pred_l1, color='gray', linestyle='--', alpha=0.7, label=f'Pred ({pred_l1:.4f})')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Eval L1 (novel views)')
    ax.set_title('Novel View Quality')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(f'Convergence: Token+MSE vs Token+LPIPS\n{uid}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save3 = os.path.join(args.output_dir, f'{uid}_convergence.png')
    plt.savefig(save3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save3}")

    print("\n  完成！")


if __name__ == "__main__":
    main()
