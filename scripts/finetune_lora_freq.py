"""
LoRA + Frequency-Weighted Loss Fine-tuning

双重正则化策略：
  - LoRA (Low-Rank Adaptation): 限制参数更新在低秩子空间，防止过拟合
  - Freq-Weighted Loss: 引导优化方向聚焦低频结构

对比方案：
  A. LoRA + Standard MSE
  B. LoRA + Freq-Weighted MSE
  C. (参考) Full-FT + Freq-Weighted MSE

用法：
    python scripts/finetune_lora_freq.py --output_dir ./exps/lora_freq
"""

import os
import sys
import argparse
import json
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


# ─── LoRA via Forward Hooks ───
# nn.MultiheadAttention uses F.linear internally, not self.out_proj(x),
# so module replacement doesn't work. We use forward hooks instead:
# each hook adds a low-rank residual to the attention output.

class LoRAAdapter(nn.Module):
    """Low-rank adapter: output += B @ A @ input * scaling"""
    def __init__(self, dim, rank=4, alpha=1.0):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.lora_A.weight, std=0.01)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scaling


def inject_lora(model, rank=4, alpha=1.0):
    """
    Inject LoRA adapters via forward hooks on attention modules.
    Hook adds low-rank residual to attention output (attn_output).
    Returns list of LoRA parameters for optimizer.
    """
    adapters = nn.ModuleList()
    hooks = []

    for i, layer in enumerate(model.transformer.layers):
        for attn_name in ['cross_attn', 'self_attn']:
            attn = getattr(layer, attn_name, None)
            if attn is None:
                continue

            adapter = LoRAAdapter(attn.embed_dim, rank=rank, alpha=alpha)
            adapters.append(adapter)

            def make_hook(adpt):
                def hook_fn(module, input, output):
                    # MHA returns (attn_output, attn_weights)
                    attn_out, attn_weights = output
                    attn_out = attn_out + adpt(attn_out)
                    return (attn_out, attn_weights)
                return hook_fn

            h = attn.register_forward_hook(make_hook(adapter))
            hooks.append(h)

    # Move adapters to same device as model
    device = next(model.parameters()).device
    adapters = adapters.to(device)

    # Store adapters on model so they're tracked
    model._lora_adapters = adapters
    model._lora_hooks = hooks

    lora_params = list(adapters.parameters())
    print(f"  Injected {len(adapters)} LoRA adapters (rank={rank})")
    return lora_params


# ─── Data & Loss Functions ───

def load_render_data(data_dir, num_views=32):
    rgba_dir = os.path.join(data_dir, 'rgba')
    pose_dir = os.path.join(data_dir, 'pose')
    intrinsics = torch.from_numpy(np.load(os.path.join(data_dir, 'intrinsics.npy'))).float()
    images, poses = [], []
    for i in range(num_views):
        img_path = os.path.join(rgba_dir, f'{i:03d}.png')
        if not os.path.exists(img_path):
            break
        img = np.array(Image.open(img_path)).astype(np.float32) / 255.0
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
    err = pred - gt
    N, C, H, W = err.shape
    err_fft = torch.fft.fft2(err)
    fy = torch.fft.fftfreq(H, device=err.device).unsqueeze(1)
    fx = torch.fft.fftfreq(W, device=err.device).unsqueeze(0)
    freq_radius = (fx**2 + fy**2).sqrt()
    weight = torch.ones_like(freq_radius)
    weight[freq_radius < cutoff_ratio] = low_freq_weight
    weighted_fft = err_fft * weight.unsqueeze(0).unsqueeze(0)
    weighted_err = torch.fft.ifft2(weighted_fft).real
    return weighted_err.pow(2).mean()


def render_one_view(model, src_img, source_cam, cam, render_size, device):
    planes = model.forward_planes(src_img, source_cam)
    out = model.synthesizer(
        planes=planes, cameras=cam.unsqueeze(0),
        anchors=torch.zeros(1, 1, 2, device=device),
        resolutions=torch.ones(1, 1, 1, device=device) * render_size,
        bg_colors=torch.ones(1, 1, 1, device=device),
        region_size=render_size,
    )
    return out['images_rgb']


def evaluate_model(model, data, source_cam, render_size, source_size, device, eval_views):
    """评估：返回 dict of metric lists"""
    import lpips
    from skimage.metrics import structural_similarity as ssim_fn

    if not hasattr(evaluate_model, '_lpips'):
        evaluate_model._lpips = lpips.LPIPS(net='alex').to(device).eval()
    lpips_model = evaluate_model._lpips

    model.eval()
    metrics = {'l1': [], 'psnr': [], 'ssim': [], 'lpips': []}

    with torch.no_grad():
        for uid, (images, cameras) in data.items():
            src_img = images[0:1].to(device)
            src_img_r = F.interpolate(src_img, size=(source_size, source_size),
                                      mode='bicubic', align_corners=True).clamp(0, 1)
            for vi in eval_views:
                cam = cameras[vi:vi+1].to(device)
                pred = render_one_view(model, src_img_r, source_cam, cam, render_size, device).squeeze()
                gt = F.interpolate(images[vi:vi+1], size=(render_size, render_size),
                                   mode='bilinear', align_corners=False).squeeze().to(device)

                # L1
                metrics['l1'].append((pred - gt).abs().mean().item())
                # PSNR
                mse = F.mse_loss(pred, gt).item()
                metrics['psnr'].append(-10 * np.log10(mse + 1e-10))
                # SSIM
                p_np = pred.cpu().permute(1, 2, 0).clamp(0, 1).numpy()
                g_np = gt.cpu().permute(1, 2, 0).clamp(0, 1).numpy()
                metrics['ssim'].append(ssim_fn(p_np, g_np, channel_axis=2, data_range=1.0))
                # LPIPS
                metrics['lpips'].append(
                    lpips_model(pred.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1).item()
                )

    return {k: np.mean(v) for k, v in metrics.items()}


# ─── Main ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/rendered")
    parser.add_argument("--model_name", type=str, default="zxhezexin/openlrm-mix-base-1.1")
    parser.add_argument("--infer_config", type=str, default="./configs/infer-b.yaml")
    parser.add_argument("--output_dir", type=str, default="./exps/lora_freq")
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--low_freq_weight", type=float, default=5.0)
    parser.add_argument("--cutoff_ratio", type=float, default=0.3)
    parser.add_argument("--num_train_views", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.infer_config)
    render_size = cfg.render_size
    source_size = cfg.source_size

    os.makedirs(args.output_dir, exist_ok=True)

    # ─── 加载数据 ───
    meta_dir = os.path.join(args.data_root, 'meta')
    if os.path.exists(os.path.join(meta_dir, 'train_uids.json')):
        with open(os.path.join(meta_dir, 'train_uids.json')) as f:
            train_uids = json.load(f)
        with open(os.path.join(meta_dir, 'val_uids.json')) as f:
            val_uids = json.load(f)
    else:
        uids = [d for d in os.listdir(args.data_root)
                if os.path.isdir(os.path.join(args.data_root, d)) and d != 'meta']
        train_uids = uids
        val_uids = uids

    print(f"[数据] Train: {len(train_uids)}, Val: {len(val_uids)}")

    all_data = {}
    for uid in train_uids + val_uids:
        dp = os.path.join(args.data_root, uid)
        if not os.path.exists(os.path.join(dp, 'intrinsics.npy')):
            continue
        images, poses, intrinsics = load_render_data(dp)
        normalized_poses = camera_normalization_objaverse('auto', poses)
        intrinsics_batch = intrinsics.unsqueeze(0).repeat(poses.shape[0], 1, 1)
        render_cameras = build_camera_standard(normalized_poses, intrinsics_batch)
        all_data[uid] = (images, render_cameras)

    train_data = {uid: all_data[uid] for uid in train_uids if uid in all_data}
    val_data = {uid: all_data[uid] for uid in val_uids if uid in all_data}
    print(f"[数据] 实际加载 Train: {len(train_data)}, Val: {len(val_data)}")

    # 视角设置
    total_views = min(32, min(len(imgs) for imgs, _ in all_data.values()))
    train_step = max(1, total_views // args.num_train_views)
    train_views = list(range(0, total_views, train_step))[:args.num_train_views]
    remaining = sorted(set(range(total_views)) - set(train_views))
    eval_views = remaining[::max(1, len(remaining) // 6)][:6]
    print(f"[视角] Train: {train_views}, Eval: {eval_views}")

    source_cam = get_source_camera(cfg.source_cam_dist, device)

    # ─── 加载模型 ───
    print("[模型] 加载 3 份模型...")
    from openlrm.models import model_dict
    hf_model_cls = wrap_model_hub(model_dict['lrm'])

    # Model A: LoRA + Standard MSE
    model_lora_std = hf_model_cls.from_pretrained(args.model_name).to(device)
    # Model B: LoRA + Freq-Weighted
    model_lora_freq = hf_model_cls.from_pretrained(args.model_name).to(device)

    # 注入 LoRA into out_proj layers (inject_lora freezes original weights as buffers)
    print(f"[LoRA] Rank={args.lora_rank}, Alpha={args.lora_alpha}")
    lora_params_std = inject_lora(model_lora_std, rank=args.lora_rank, alpha=args.lora_alpha)
    lora_params_freq = inject_lora(model_lora_freq, rank=args.lora_rank, alpha=args.lora_alpha)

    # Freeze everything except LoRA params
    # LoRA params already have requires_grad=True (nn.Parameter default)
    # All other model params need requires_grad=False
    lora_param_set_std = set(id(p) for p in lora_params_std)
    lora_param_set_freq = set(id(p) for p in lora_params_freq)
    for p in model_lora_std.parameters():
        if id(p) not in lora_param_set_std:
            p.requires_grad_(False)
    for p in model_lora_freq.parameters():
        if id(p) not in lora_param_set_freq:
            p.requires_grad_(False)

    num_lora = sum(p.numel() for p in lora_params_std)
    total_params = sum(p.numel() for p in model_lora_std.parameters())
    trainable = sum(p.numel() for p in model_lora_std.parameters() if p.requires_grad)
    print(f"[LoRA] 可训练参数: {trainable:,} / {total_params:,} "
          f"(压缩比 {total_params/trainable:.0f}x)")

    # Baseline 评估
    print("[评估] Baseline...")
    baseline_metrics = evaluate_model(model_lora_std, val_data, source_cam,
                                       render_size, source_size, device, eval_views)
    print(f"  Baseline: L1={baseline_metrics['l1']:.4f} PSNR={baseline_metrics['psnr']:.1f} "
          f"SSIM={baseline_metrics['ssim']:.4f} LPIPS={baseline_metrics['lpips']:.4f}")

    # ─── 优化器 ───
    opt_std = torch.optim.AdamW(lora_params_std, lr=args.lr, weight_decay=0.01)
    opt_freq = torch.optim.AdamW(lora_params_freq, lr=args.lr, weight_decay=0.01)

    # ─── 微调循环 ───
    print(f"[微调] {args.num_steps} steps...")
    uid_list = list(train_data.keys())
    hist = {'step': [], 'std_loss': [], 'freq_loss': [],
            'std_l1': [], 'freq_l1': [], 'std_psnr': [], 'freq_psnr': [],
            'std_ssim': [], 'freq_ssim': [], 'std_lpips': [], 'freq_lpips': []}

    eval_interval = max(20, args.num_steps // 25)

    pbar = tqdm(range(args.num_steps), desc='LoRA Fine-tuning')
    for step in pbar:
        uid = uid_list[step % len(uid_list)]
        images, cameras = train_data[uid]
        vi = train_views[step % len(train_views)]

        src_img = images[0:1].to(device)
        src_img = F.interpolate(src_img, size=(source_size, source_size),
                                mode='bicubic', align_corners=True).clamp(0, 1)
        gt_img = F.interpolate(images[vi:vi+1], size=(render_size, render_size),
                               mode='bilinear', align_corners=False).to(device)
        cam = cameras[vi:vi+1].to(device)

        # ── LoRA + Standard MSE ──
        model_lora_std.train()
        opt_std.zero_grad()
        pred_s = render_one_view(model_lora_std, src_img, source_cam, cam, render_size, device).squeeze(0)
        loss_s = F.mse_loss(pred_s, gt_img)
        loss_s.backward()
        opt_std.step()

        # ── LoRA + Freq-Weighted ──
        model_lora_freq.train()
        opt_freq.zero_grad()
        pred_f = render_one_view(model_lora_freq, src_img, source_cam, cam, render_size, device).squeeze(0)
        loss_f = freq_weighted_mse(pred_f, gt_img,
                                   low_freq_weight=args.low_freq_weight,
                                   cutoff_ratio=args.cutoff_ratio)
        loss_f.backward()
        opt_freq.step()

        pbar.set_postfix(std=f'{loss_s.item():.5f}', freq=f'{loss_f.item():.5f}')

        # 定期评估
        if step % eval_interval == 0 or step == args.num_steps - 1:
            m_std = evaluate_model(model_lora_std, val_data, source_cam,
                                   render_size, source_size, device, eval_views)
            m_freq = evaluate_model(model_lora_freq, val_data, source_cam,
                                    render_size, source_size, device, eval_views)

            hist['step'].append(step)
            hist['std_loss'].append(loss_s.item())
            hist['freq_loss'].append(loss_f.item())
            for mk in ['l1', 'psnr', 'ssim', 'lpips']:
                hist[f'std_{mk}'].append(m_std[mk])
                hist[f'freq_{mk}'].append(m_freq[mk])

    # ─── 最终评估 ───
    print("\n[最终评估]")
    final_std = evaluate_model(model_lora_std, val_data, source_cam,
                                render_size, source_size, device, eval_views)
    final_freq = evaluate_model(model_lora_freq, val_data, source_cam,
                                 render_size, source_size, device, eval_views)

    print(f"\n{'方法':<28} {'L1↓':>8} {'PSNR↑':>8} {'SSIM↑':>8} {'LPIPS↓':>8}")
    print("-" * 65)
    b = baseline_metrics
    print(f"{'Baseline':<28} {b['l1']:>8.4f} {b['psnr']:>8.1f} {b['ssim']:>8.4f} {b['lpips']:>8.4f}")
    print(f"{'LoRA + Std MSE':<28} {final_std['l1']:>8.4f} {final_std['psnr']:>8.1f} "
          f"{final_std['ssim']:>8.4f} {final_std['lpips']:>8.4f}")
    print(f"{'LoRA + Freq-Weighted':<28} {final_freq['l1']:>8.4f} {final_freq['psnr']:>8.1f} "
          f"{final_freq['ssim']:>8.4f} {final_freq['lpips']:>8.4f}")

    print(f"\n改善 vs Baseline:")
    for mk, hb in [('l1', False), ('psnr', True), ('ssim', True), ('lpips', False)]:
        bv = b[mk]
        sv = final_std[mk]
        fv = final_freq[mk]
        if hb:
            print(f"  {mk.upper():<6} Std: {sv-bv:+.4f}  Freq: {fv-bv:+.4f}")
        else:
            print(f"  {mk.upper():<6} Std: {(1-sv/bv)*100:+.1f}%  Freq: {(1-fv/bv)*100:+.1f}%")

    # ─── 可视化：收敛曲线（4 指标）───
    print("[可视化] 收敛曲线...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metric_info = [
        ('l1', 'L1 (↓)', False),
        ('psnr', 'PSNR (↑)', True),
        ('ssim', 'SSIM (↑)', True),
        ('lpips', 'LPIPS (↓)', False),
    ]
    for idx, (mk, label, higher_better) in enumerate(metric_info):
        ax = axes[idx // 2][idx % 2]
        ax.plot(hist['step'], hist[f'std_{mk}'], 'b-', label='LoRA+Std', linewidth=2)
        ax.plot(hist['step'], hist[f'freq_{mk}'], 'r-', label='LoRA+Freq', linewidth=2)
        ax.axhline(b[mk], color='gray', linestyle='--', alpha=0.7,
                   label=f'Baseline ({b[mk]:.4f})')
        ax.set_xlabel('Step')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(f'LoRA Fine-tuning Convergence (rank={args.lora_rank})\n'
                 f'Train: {len(train_data)}, Val: {len(val_data)}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'convergence_4metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ─── 可视化：渲染对比 ───
    print("[可视化] 渲染对比...")
    model_base = hf_model_cls.from_pretrained(args.model_name).to(device).eval()
    model_lora_std.eval()
    model_lora_freq.eval()

    vis_uids = list(val_data.keys())[:5]
    vis_views = eval_views[:4]

    for uid in vis_uids:
        images, cameras = val_data[uid]
        src_img = images[0:1].to(device)
        src_r = F.interpolate(src_img, size=(source_size, source_size),
                              mode='bicubic', align_corners=True).clamp(0, 1)

        methods = [('Baseline', model_base), ('LoRA+Std', model_lora_std),
                   ('LoRA+Freq', model_lora_freq), ('GT', None)]
        n_methods = len(methods)
        fig, axes = plt.subplots(n_methods, len(vis_views) + 1,
                                 figsize=(4*(len(vis_views)+1), 4*n_methods))

        with torch.no_grad():
            # Input column
            inp = images[0].permute(1, 2, 0).numpy()
            for row in range(n_methods):
                axes[row, 0].imshow(inp)
                axes[row, 0].axis('off')
                if row == 0:
                    axes[row, 0].set_title('Input', fontsize=11, fontweight='bold')

            for col, vi in enumerate(vis_views):
                cam = cameras[vi:vi+1].to(device)
                gt = F.interpolate(images[vi:vi+1], size=(render_size, render_size),
                                   mode='bilinear', align_corners=False).squeeze(0)

                for row, (m_name, m_obj) in enumerate(methods):
                    if m_obj is not None:
                        pred = render_one_view(m_obj, src_r, source_cam, cam,
                                             render_size, device).squeeze().cpu()
                    else:
                        pred = gt
                    img = pred.permute(1, 2, 0).clamp(0, 1).numpy()
                    axes[row, col + 1].imshow(img)
                    axes[row, col + 1].axis('off')
                    if row == 0:
                        axes[row, col + 1].set_title(f'View {vi}', fontsize=11)

            for row, (m_name, _) in enumerate(methods):
                axes[row, 0].text(-0.15, 0.5, m_name, transform=axes[row, 0].transAxes,
                                  fontsize=13, fontweight='bold', va='center', ha='right')

        plt.suptitle(f'{uid[:12]}...', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'vis_{uid[:12]}.png'), dpi=120, bbox_inches='tight')
        plt.close()

    del model_base
    torch.cuda.empty_cache()

    # ─── 保存结果 ───
    results = {
        'config': {
            'lora_rank': args.lora_rank,
            'lora_alpha': args.lora_alpha,
            'lr': args.lr,
            'num_steps': args.num_steps,
            'low_freq_weight': args.low_freq_weight,
            'lora_params': num_lora,
            'num_train': len(train_data),
            'num_val': len(val_data),
        },
        'baseline': baseline_metrics,
        'lora_std': final_std,
        'lora_freq': final_freq,
        'history': hist,
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n保存到 {args.output_dir}")
    print("完成！")


if __name__ == "__main__":
    main()
