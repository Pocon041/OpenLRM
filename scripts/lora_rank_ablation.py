"""
LoRA Rank 消融实验：rank={2, 4, 8, 16, 32}
每个 rank 跑 LoRA+Std 和 LoRA+Freq，记录 4 指标，画参数效率曲线。

用法：
    python scripts/lora_rank_ablation.py
"""

import os, sys, json, torch, numpy as np
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
    build_camera_standard, build_camera_principle,
    create_intrinsics, camera_normalization_objaverse,
)
from openlrm.utils.hf_hub import wrap_model_hub

# ─── Reuse LoRA + eval from finetune_lora_freq.py ───

class LoRAAdapter(nn.Module):
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
    adapters = nn.ModuleList()
    hooks = []
    for layer in model.transformer.layers:
        for attn_name in ['cross_attn', 'self_attn']:
            attn = getattr(layer, attn_name, None)
            if attn is None:
                continue
            adapter = LoRAAdapter(attn.embed_dim, rank=rank, alpha=alpha)
            adapters.append(adapter)

            def make_hook(adpt):
                def hook_fn(module, input, output):
                    attn_out, attn_weights = output
                    return (attn_out + adpt(attn_out), attn_weights)
                return hook_fn

            h = attn.register_forward_hook(make_hook(adapter))
            hooks.append(h)

    device = next(model.parameters()).device
    adapters = adapters.to(device)
    model._lora_adapters = adapters
    model._lora_hooks = hooks
    return list(adapters.parameters()), hooks


def remove_lora(model):
    if hasattr(model, '_lora_hooks'):
        for h in model._lora_hooks:
            h.remove()
        del model._lora_hooks
    if hasattr(model, '_lora_adapters'):
        del model._lora_adapters


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
        t = torch.from_numpy(img).permute(2, 0, 1)
        rgb = t[:3] * t[3:4] + (1 - t[3:4])
        images.append(rgb)
        poses.append(torch.from_numpy(np.load(os.path.join(pose_dir, f'{i:03d}.npy'))).float())
    return torch.stack(images), torch.stack(poses), intrinsics


def get_source_camera(dist, device):
    ext = torch.tensor([[[1,0,0,0],[0,0,-1,-dist],[0,1,0,0]]], dtype=torch.float32, device=device)
    intr = create_intrinsics(f=0.75, c=0.5, device=device).unsqueeze(0)
    return build_camera_principle(ext, intr)


def freq_weighted_mse(pred, gt, low_freq_weight=5.0, cutoff=0.3):
    err = pred - gt
    N, C, H, W = err.shape
    fft = torch.fft.fft2(err)
    fy = torch.fft.fftfreq(H, device=err.device).unsqueeze(1)
    fx = torch.fft.fftfreq(W, device=err.device).unsqueeze(0)
    r = (fx**2 + fy**2).sqrt()
    w = torch.ones_like(r)
    w[r < cutoff] = low_freq_weight
    return torch.fft.ifft2(fft * w.unsqueeze(0).unsqueeze(0)).real.pow(2).mean()


def render_one(model, src, scam, cam, rs, dev):
    planes = model.forward_planes(src, scam)
    out = model.synthesizer(
        planes=planes, cameras=cam.unsqueeze(0),
        anchors=torch.zeros(1,1,2, device=dev),
        resolutions=torch.ones(1,1,1, device=dev)*rs,
        bg_colors=torch.ones(1,1,1, device=dev),
        region_size=rs,
    )
    return out['images_rgb']


def evaluate(model, data, scam, rs, ss, dev, eviews):
    import lpips as lp
    from skimage.metrics import structural_similarity as ssim_fn
    if not hasattr(evaluate, '_lp'):
        evaluate._lp = lp.LPIPS(net='alex').to(dev).eval()
    lpips_m = evaluate._lp
    model.eval()
    m = {'l1': [], 'psnr': [], 'ssim': [], 'lpips': []}
    with torch.no_grad():
        for uid, (imgs, cams) in data.items():
            si = F.interpolate(imgs[0:1].to(dev), size=(ss,ss), mode='bicubic', align_corners=True).clamp(0,1)
            for vi in eviews:
                c = cams[vi:vi+1].to(dev)
                pred = render_one(model, si, scam, c, rs, dev).squeeze()
                gt = F.interpolate(imgs[vi:vi+1], size=(rs,rs), mode='bilinear', align_corners=False).squeeze().to(dev)
                m['l1'].append((pred-gt).abs().mean().item())
                mse = F.mse_loss(pred, gt).item()
                m['psnr'].append(-10*np.log10(mse+1e-10))
                pn = pred.cpu().permute(1,2,0).clamp(0,1).numpy()
                gn = gt.cpu().permute(1,2,0).clamp(0,1).numpy()
                m['ssim'].append(ssim_fn(pn, gn, channel_axis=2, data_range=1.0))
                m['lpips'].append(lpips_m(pred.unsqueeze(0)*2-1, gt.unsqueeze(0)*2-1).item())
    return {k: np.mean(v) for k, v in m.items()}


def run_one_rank(rank, model_name, train_data, val_data, source_cam, render_size, source_size,
                 device, train_views, eval_views, num_steps=500, lr=5e-4, alpha_ratio=2.0):
    """Run LoRA+Std and LoRA+Freq for one rank value. Returns metrics dict."""
    from openlrm.models import model_dict
    hf_cls = wrap_model_hub(model_dict['lrm'])
    alpha = rank * alpha_ratio

    results = {}
    uid_list = list(train_data.keys())

    for loss_name in ['std', 'freq']:
        model = hf_cls.from_pretrained(model_name).to(device)
        for p in model.parameters():
            p.requires_grad_(False)

        lora_params, hooks = inject_lora(model, rank=rank, alpha=alpha)

        # Mark only LoRA as trainable
        lora_set = set(id(p) for p in lora_params)
        for p in model.parameters():
            if id(p) not in lora_set:
                p.requires_grad_(False)

        num_p = sum(p.numel() for p in lora_params)
        opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)

        for step in range(num_steps):
            uid = uid_list[step % len(uid_list)]
            imgs, cams = train_data[uid]
            vi = train_views[step % len(train_views)]

            si = F.interpolate(imgs[0:1].to(device), size=(source_size, source_size),
                               mode='bicubic', align_corners=True).clamp(0,1)
            gt = F.interpolate(imgs[vi:vi+1], size=(render_size, render_size),
                               mode='bilinear', align_corners=False).to(device)
            cam = cams[vi:vi+1].to(device)

            model.train()
            opt.zero_grad()
            pred = render_one(model, si, source_cam, cam, render_size, device).squeeze(0)

            if loss_name == 'std':
                loss = F.mse_loss(pred, gt)
            else:
                loss = freq_weighted_mse(pred, gt)

            loss.backward()
            opt.step()

        metrics = evaluate(model, val_data, source_cam, render_size, source_size, device, eval_views)
        metrics['num_params'] = num_p
        results[loss_name] = metrics

        # Clean up
        remove_lora(model)
        del model, opt, lora_params
        torch.cuda.empty_cache()

    return results


def main():
    data_root = "./data/rendered"
    model_name = "zxhezexin/openlrm-mix-base-1.1"
    infer_config = "./configs/infer-b.yaml"
    output_dir = "./exps/lora_ablation"
    num_steps = 500
    ranks = [2, 4, 8, 16, 32]

    device = torch.device("cuda")
    cfg = OmegaConf.load(infer_config)
    rs = cfg.render_size
    ss = cfg.source_size
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    meta_dir = os.path.join(data_root, 'meta')
    with open(os.path.join(meta_dir, 'train_uids.json')) as f:
        train_uids = json.load(f)
    with open(os.path.join(meta_dir, 'val_uids.json')) as f:
        val_uids = json.load(f)

    all_data = {}
    for uid in train_uids + val_uids:
        dp = os.path.join(data_root, uid)
        if not os.path.exists(os.path.join(dp, 'intrinsics.npy')):
            continue
        images, poses, intrinsics = load_render_data(dp)
        norm_poses = camera_normalization_objaverse('auto', poses)
        intr_batch = intrinsics.unsqueeze(0).repeat(poses.shape[0], 1, 1)
        all_data[uid] = (images, build_camera_standard(norm_poses, intr_batch))

    train_data = {u: all_data[u] for u in train_uids if u in all_data}
    val_data = {u: all_data[u] for u in val_uids if u in all_data}
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    total_views = min(32, min(len(i) for i, _ in all_data.values()))
    train_views = list(range(0, total_views, max(1, total_views // 4)))[:4]
    remaining = sorted(set(range(total_views)) - set(train_views))
    eval_views = remaining[::max(1, len(remaining) // 6)][:6]

    source_cam = get_source_camera(cfg.source_cam_dist, device)

    # Baseline
    print("Baseline 评估...")
    from openlrm.models import model_dict
    hf_cls = wrap_model_hub(model_dict['lrm'])
    model_b = hf_cls.from_pretrained(model_name).to(device).eval()
    baseline = evaluate(model_b, val_data, source_cam, rs, ss, device, eval_views)
    del model_b; torch.cuda.empty_cache()
    print(f"  Baseline: L1={baseline['l1']:.4f} PSNR={baseline['psnr']:.1f} "
          f"SSIM={baseline['ssim']:.4f} LPIPS={baseline['lpips']:.4f}")

    # Run ablation
    all_results = {'baseline': baseline, 'ranks': {}}

    for rank in ranks:
        print(f"\n{'='*60}")
        print(f"Rank={rank}")
        print(f"{'='*60}")
        results = run_one_rank(rank, model_name, train_data, val_data, source_cam,
                               rs, ss, device, train_views, eval_views, num_steps=num_steps)
        all_results['ranks'][rank] = results

        for loss_name in ['std', 'freq']:
            m = results[loss_name]
            print(f"  LoRA+{'Std':4s} (rank={rank:2d}, params={m['num_params']:>7,}): "
                  f"L1={m['l1']:.4f} PSNR={m['psnr']:.1f} SSIM={m['ssim']:.4f} LPIPS={m['lpips']:.4f}")

    # Save results
    with open(os.path.join(output_dir, 'ablation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x))

    # ─── Plot ───
    print("\n生成图表...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metric_info = [('l1', 'L1 (↓)', False), ('psnr', 'PSNR (↑)', True),
                   ('ssim', 'SSIM (↑)', True), ('lpips', 'LPIPS (↓)', False)]

    for idx, (mk, label, higher_better) in enumerate(metric_info):
        ax = axes[idx // 2][idx % 2]
        params_std, vals_std = [], []
        params_freq, vals_freq = [], []

        for rank in ranks:
            r = all_results['ranks'][rank]
            params_std.append(r['std']['num_params'])
            vals_std.append(r['std'][mk])
            params_freq.append(r['freq']['num_params'])
            vals_freq.append(r['freq'][mk])

        ax.plot(params_std, vals_std, 'b-o', label='LoRA+Std', linewidth=2, markersize=8)
        ax.plot(params_freq, vals_freq, 'r-s', label='LoRA+Freq', linewidth=2, markersize=8)
        ax.axhline(baseline[mk], color='gray', linestyle='--', alpha=0.7,
                   label=f'Baseline ({baseline[mk]:.4f})')

        # Annotate ranks
        for i, rank in enumerate(ranks):
            ax.annotate(f'r={rank}', (params_std[i], vals_std[i]),
                       textcoords="offset points", xytext=(0, 10), fontsize=8, ha='center')

        ax.set_xlabel('Trainable Parameters')
        ax.set_xscale('log')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(f'LoRA Rank Ablation: Parameter Efficiency\n'
                 f'Train: {len(train_data)}, Val: {len(val_data)}, Steps: {num_steps}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rank_ablation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Summary table
    print(f"\n{'Rank':>5} {'Params':>8} | {'--- LoRA+Std ---':^28} | {'--- LoRA+Freq ---':^28}")
    print(f"{'':5} {'':8} | {'L1':>7} {'PSNR':>6} {'SSIM':>6} {'LPIPS':>6} | {'L1':>7} {'PSNR':>6} {'SSIM':>6} {'LPIPS':>6}")
    print("-" * 80)
    for rank in ranks:
        r = all_results['ranks'][rank]
        s, f = r['std'], r['freq']
        print(f"{rank:>5} {s['num_params']:>8,} | "
              f"{s['l1']:.4f} {s['psnr']:6.1f} {s['ssim']:.4f} {s['lpips']:.4f} | "
              f"{f['l1']:.4f} {f['psnr']:6.1f} {f['ssim']:.4f} {f['lpips']:.4f}")

    print(f"\n保存到 {output_dir}")
    print("完成！")


if __name__ == "__main__":
    main()
