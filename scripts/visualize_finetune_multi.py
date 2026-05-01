"""
多样本可视化：对 val set 每个样本渲染 Baseline / Std-FT / Freq-FT / GT 对比
"""
import os, sys, json, torch, numpy as np
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from accelerate import PartialState
PartialState()
torch._dynamo.config.suppress_errors = True

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openlrm.datasets.cam_utils import (
    build_camera_standard, build_camera_principle,
    create_intrinsics, camera_normalization_objaverse,
)
from openlrm.utils.hf_hub import wrap_model_hub


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


def render_one_view(model, src_img, source_cam, cam, render_size, device):
    planes = model.forward_planes(src_img, source_cam)
    out = model.synthesizer(
        planes=planes, cameras=cam.unsqueeze(0),
        anchors=torch.zeros(1, 1, 2, device=device),
        resolutions=torch.ones(1, 1, 1, device=device) * render_size,
        bg_colors=torch.ones(1, 1, 1, device=device),
        region_size=render_size,
    )
    return out['images_rgb'].squeeze().cpu()


def main():
    data_root = "./data/rendered"
    model_name = "zxhezexin/openlrm-mix-base-1.1"
    infer_config = "./configs/infer-b.yaml"
    ft_dir = "./exps/finetune_freq_v2"
    output_dir = ft_dir

    device = torch.device("cuda")
    cfg = OmegaConf.load(infer_config)
    render_size = cfg.render_size
    source_size = cfg.source_size
    source_cam = get_source_camera(cfg.source_cam_dist, device)

    # 加载 val uids
    with open(os.path.join(data_root, 'meta', 'val_uids.json')) as f:
        val_uids = json.load(f)
    with open(os.path.join(data_root, 'meta', 'train_uids.json')) as f:
        train_uids = json.load(f)
    print(f"Val samples: {len(val_uids)}")

    # 加载数据
    all_data = {}
    for uid in val_uids + train_uids[:3]:  # val + 几个 train 样本
        dp = os.path.join(data_root, uid)
        if not os.path.exists(os.path.join(dp, 'intrinsics.npy')):
            continue
        images, poses, intrinsics = load_render_data(dp)
        normalized_poses = camera_normalization_objaverse('auto', poses)
        intrinsics_batch = intrinsics.unsqueeze(0).repeat(poses.shape[0], 1, 1)
        render_cameras = build_camera_standard(normalized_poses, intrinsics_batch)
        all_data[uid] = (images, render_cameras)

    # 训练/评估视角
    total_views = min(32, min(len(imgs) for imgs, _ in all_data.values()))
    train_views = [0, 8, 16, 24]
    eval_views = [3, 7, 11, 15]

    # 加载 3 个模型
    print("加载模型...")
    from openlrm.models import model_dict
    hf_model_cls = wrap_model_hub(model_dict['lrm'])

    model_base = hf_model_cls.from_pretrained(model_name).to(device).eval()

    # 需要重新训练来获取 std/freq 模型，或者直接重跑
    # 这里我们重新加载并快速微调
    # 为了节省时间，直接重跑微调脚本中的训练然后渲染
    # 实际上我们需要保存模型权重...
    # 简单方案：重新快速微调
    model_std = hf_model_cls.from_pretrained(model_name).to(device)
    model_freq = hf_model_cls.from_pretrained(model_name).to(device)

    # 冻结 encoder, 训练 transformer + upsampler
    for model in [model_std, model_freq]:
        model.train()
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        model.pos_embed.requires_grad_(True)
        for p in model.transformer.parameters():
            p.requires_grad_(True)
        for p in model.upsampler.parameters():
            p.requires_grad_(True)

    trainable_std = [p for p in model_std.parameters() if p.requires_grad]
    trainable_freq = [p for p in model_freq.parameters() if p.requires_grad]
    opt_std = torch.optim.AdamW(trainable_std, lr=1e-5, weight_decay=0.01)
    opt_freq = torch.optim.AdamW(trainable_freq, lr=1e-5, weight_decay=0.01)

    # 快速微调 500 步
    from tqdm import tqdm
    train_data = {uid: all_data[uid] for uid in train_uids if uid in all_data}
    uid_list = list(train_data.keys())

    print(f"微调 500 步 (train: {len(uid_list)} 样本)...")
    for step in tqdm(range(500), desc='Fine-tuning'):
        uid = uid_list[step % len(uid_list)]
        images, cameras = all_data[uid]
        vi = train_views[step % len(train_views)]

        src_img = images[0:1].to(device)
        src_img = F.interpolate(src_img, size=(source_size, source_size),
                                mode='bicubic', align_corners=True).clamp(0, 1)
        gt_img = F.interpolate(images[vi:vi+1], size=(render_size, render_size),
                               mode='bilinear', align_corners=False).to(device)
        cam = cameras[vi:vi+1].to(device)

        # Standard MSE
        opt_std.zero_grad()
        planes_s = model_std.forward_planes(src_img, source_cam)
        out_s = model_std.synthesizer(
            planes=planes_s, cameras=cam.unsqueeze(0),
            anchors=torch.zeros(1, 1, 2, device=device),
            resolutions=torch.ones(1, 1, 1, device=device) * render_size,
            bg_colors=torch.ones(1, 1, 1, device=device),
            region_size=render_size,
        )
        loss_s = F.mse_loss(out_s['images_rgb'].squeeze(0), gt_img)
        loss_s.backward()
        torch.nn.utils.clip_grad_norm_(trainable_std, 1.0)
        opt_std.step()

        # Freq-weighted
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
        err = pred_f - gt_img
        err_fft = torch.fft.fft2(err)
        H, W = err.shape[-2:]
        fy = torch.fft.fftfreq(H, device=device).unsqueeze(1)
        fx = torch.fft.fftfreq(W, device=device).unsqueeze(0)
        freq_r = (fx**2 + fy**2).sqrt()
        weight = torch.ones_like(freq_r)
        weight[freq_r < 0.3] = 5.0
        weighted_err = torch.fft.ifft2(err_fft * weight).real
        loss_f = weighted_err.pow(2).mean()
        loss_f.backward()
        torch.nn.utils.clip_grad_norm_(trainable_freq, 1.0)
        opt_freq.step()

    model_std.eval()
    model_freq.eval()

    # 渲染所有 val 样本
    print("渲染 val 样本...")
    vis_uids = val_uids[:5]
    vis_views = eval_views[:4]

    for uid in vis_uids:
        if uid not in all_data:
            continue
        images, cameras = all_data[uid]
        src_img = images[0:1].to(device)
        src_img_r = F.interpolate(src_img, size=(source_size, source_size),
                                  mode='bicubic', align_corners=True).clamp(0, 1)

        fig, axes = plt.subplots(4, len(vis_views) + 1, figsize=(4*(len(vis_views)+1), 16),
                                 gridspec_kw={'width_ratios': [1] + [1]*len(vis_views)})
        method_names = ['Baseline', 'Std-FT', 'Freq-FT', 'GT']
        models_list = [model_base, model_std, model_freq, None]

        # 第一列：输入图像
        input_img = images[0].permute(1, 2, 0).numpy()
        for row in range(4):
            axes[row, 0].imshow(input_img)
            axes[row, 0].axis('off')
            if row == 0:
                axes[row, 0].set_title('Input', fontsize=12, fontweight='bold')

        with torch.no_grad():
            for col_idx, vi in enumerate(vis_views):
                cam = cameras[vi:vi+1].to(device)
                gt = F.interpolate(images[vi:vi+1], size=(render_size, render_size),
                                   mode='bilinear', align_corners=False).squeeze(0)

                for row, (m_name, m_obj) in enumerate(zip(method_names, models_list)):
                    if m_obj is not None:
                        pred = render_one_view(m_obj, src_img_r, source_cam, cam, render_size, device)
                    else:
                        pred = gt
                    img = pred.permute(1, 2, 0).clamp(0, 1).numpy() if pred.shape[0] == 3 else pred.numpy()
                    axes[row, col_idx + 1].imshow(img)
                    axes[row, col_idx + 1].axis('off')
                    if row == 0:
                        axes[row, col_idx + 1].set_title(f'View {vi}', fontsize=11)

            # 行标签
            for row, m in enumerate(method_names):
                axes[row, 0].text(-0.15, 0.5, m, transform=axes[row, 0].transAxes,
                                  fontsize=14, fontweight='bold', va='center', ha='right')

        plt.suptitle(f'Sample: {uid[:12]}...', fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'vis_{uid[:12]}.png')
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  保存: {save_path}")

    # 生成总览大图：每个 val 样本一行，选2个视角
    print("生成总览图...")
    n_samples = len([u for u in vis_uids if u in all_data])
    pick_views = [eval_views[0], eval_views[2]]  # 2 个视角
    n_cols = 2 * 4 + 1  # 2 views × 4 methods + 1 input
    # 布局: 每行一个样本, 列: Input | Base_v1 Std_v1 Freq_v1 GT_v1 | Base_v2 Std_v2 Freq_v2 GT_v2
    fig, axes = plt.subplots(n_samples, 9, figsize=(36, 4*n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Input',
                  f'Base v{pick_views[0]}', f'Std-FT v{pick_views[0]}',
                  f'Freq-FT v{pick_views[0]}', f'GT v{pick_views[0]}',
                  f'Base v{pick_views[1]}', f'Std-FT v{pick_views[1]}',
                  f'Freq-FT v{pick_views[1]}', f'GT v{pick_views[1]}']

    with torch.no_grad():
        for row_idx, uid in enumerate([u for u in vis_uids if u in all_data]):
            images, cameras = all_data[uid]
            src_img = images[0:1].to(device)
            src_img_r = F.interpolate(src_img, size=(source_size, source_size),
                                      mode='bicubic', align_corners=True).clamp(0, 1)

            # Input
            inp = images[0].permute(1, 2, 0).numpy()
            axes[row_idx, 0].imshow(inp)
            axes[row_idx, 0].axis('off')
            axes[row_idx, 0].text(-0.1, 0.5, uid[:8] + '..', transform=axes[row_idx, 0].transAxes,
                                  fontsize=10, va='center', ha='right', fontfamily='monospace')

            col = 1
            for vi in pick_views:
                cam = cameras[vi:vi+1].to(device)
                gt = F.interpolate(images[vi:vi+1], size=(render_size, render_size),
                                   mode='bilinear', align_corners=False).squeeze(0)

                for m_obj in [model_base, model_std, model_freq, None]:
                    if m_obj is not None:
                        pred = render_one_view(m_obj, src_img_r, source_cam, cam, render_size, device)
                    else:
                        pred = gt
                    img = pred.permute(1, 2, 0).clamp(0, 1).numpy() if pred.shape[0] == 3 else pred.numpy()
                    axes[row_idx, col].imshow(img)
                    axes[row_idx, col].axis('off')
                    col += 1

            if row_idx == 0:
                for c, t in enumerate(col_titles):
                    axes[0, c].set_title(t, fontsize=10, fontweight='bold')

    plt.suptitle('Fine-tuning Comparison: All Val Samples\n(Baseline / Std-FT / Freq-FT / GT)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'overview_all_val.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path}")

    # ─── 全指标评估：L1, PSNR, SSIM, LPIPS ───
    import lpips
    from skimage.metrics import structural_similarity as ssim_fn

    lpips_model = lpips.LPIPS(net='alex').to(device).eval()

    def compute_metrics(pred_t, gt_t):
        """pred_t, gt_t: (C, H, W) tensors in [0,1] on device"""
        # L1
        l1 = (pred_t - gt_t).abs().mean().item()
        # PSNR
        mse = F.mse_loss(pred_t, gt_t).item()
        psnr = -10 * np.log10(mse + 1e-10)
        # SSIM (needs numpy HWC)
        p_np = pred_t.cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        g_np = gt_t.cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        ssim_val = ssim_fn(p_np, g_np, channel_axis=2, data_range=1.0)
        # LPIPS
        lpips_val = lpips_model(pred_t.unsqueeze(0) * 2 - 1,
                                gt_t.unsqueeze(0) * 2 - 1).item()
        return {'l1': l1, 'psnr': psnr, 'ssim': ssim_val, 'lpips': lpips_val}

    print("\n" + "="*90)
    print("全指标评估 (L1↓, PSNR↑, SSIM↑, LPIPS↓)")
    print("="*90)

    all_metrics = {method: {m: [] for m in ['l1', 'psnr', 'ssim', 'lpips']}
                   for method in ['base', 'std', 'freq']}
    per_sample = []

    for uid in vis_uids:
        if uid not in all_data:
            continue
        images, cameras = all_data[uid]
        src_img = images[0:1].to(device)
        src_img_r = F.interpolate(src_img, size=(source_size, source_size),
                                  mode='bicubic', align_corners=True).clamp(0, 1)
        sample_metrics = {method: {m: [] for m in ['l1', 'psnr', 'ssim', 'lpips']}
                          for method in ['base', 'std', 'freq']}

        with torch.no_grad():
            for vi in eval_views:
                cam = cameras[vi:vi+1].to(device)
                gt = F.interpolate(images[vi:vi+1], size=(render_size, render_size),
                                   mode='bilinear', align_corners=False).squeeze(0).to(device)
                for key, m_obj in [('base', model_base), ('std', model_std), ('freq', model_freq)]:
                    pred = render_one_view(m_obj, src_img_r, source_cam, cam, render_size, device).to(device)
                    m_vals = compute_metrics(pred, gt)
                    for mk in m_vals:
                        sample_metrics[key][mk].append(m_vals[mk])
                        all_metrics[key][mk].append(m_vals[mk])

        row = {'uid': uid[:12]}
        for method in ['base', 'std', 'freq']:
            for mk in ['l1', 'psnr', 'ssim', 'lpips']:
                row[f'{method}_{mk}'] = np.mean(sample_metrics[method][mk])
        per_sample.append(row)

    # 打印每样本结果
    print(f"\n{'UID':<14} | {'--- Baseline ---':^24} | {'--- Std-FT ---':^24} | {'--- Freq-FT ---':^24}")
    print(f"{'':14} | {'L1':>6} {'PSNR':>6} {'SSIM':>6} {'LPIPS':>6} | {'L1':>6} {'PSNR':>6} {'SSIM':>6} {'LPIPS':>6} | {'L1':>6} {'PSNR':>6} {'SSIM':>6} {'LPIPS':>6}")
    print("-" * 90)
    for r in per_sample:
        print(f"{r['uid']+'.':<14} | "
              f"{r['base_l1']:.4f} {r['base_psnr']:6.1f} {r['base_ssim']:.4f} {r['base_lpips']:.4f} | "
              f"{r['std_l1']:.4f} {r['std_psnr']:6.1f} {r['std_ssim']:.4f} {r['std_lpips']:.4f} | "
              f"{r['freq_l1']:.4f} {r['freq_psnr']:6.1f} {r['freq_ssim']:.4f} {r['freq_lpips']:.4f}")

    # 打印平均结果
    print("-" * 90)
    avg = {}
    for method in ['base', 'std', 'freq']:
        for mk in ['l1', 'psnr', 'ssim', 'lpips']:
            avg[f'{method}_{mk}'] = np.mean(all_metrics[method][mk])
    print(f"{'AVERAGE':<14} | "
          f"{avg['base_l1']:.4f} {avg['base_psnr']:6.1f} {avg['base_ssim']:.4f} {avg['base_lpips']:.4f} | "
          f"{avg['std_l1']:.4f} {avg['std_psnr']:6.1f} {avg['std_ssim']:.4f} {avg['std_lpips']:.4f} | "
          f"{avg['freq_l1']:.4f} {avg['freq_psnr']:6.1f} {avg['freq_ssim']:.4f} {avg['freq_lpips']:.4f}")

    # 打印改善对比
    print(f"\n改善对比 (vs Baseline):")
    print(f"{'指标':<8} {'Std-FT Δ':>12} {'Freq-FT Δ':>12} {'更优':>8}")
    print("-" * 45)
    for mk, higher_better in [('l1', False), ('psnr', True), ('ssim', True), ('lpips', False)]:
        b = avg[f'base_{mk}']
        s = avg[f'std_{mk}']
        f = avg[f'freq_{mk}']
        if higher_better:
            s_delta = s - b
            f_delta = f - b
            winner = 'Freq' if f_delta > s_delta else 'Std'
            print(f"{mk.upper():<8} {s_delta:>+11.4f} {f_delta:>+11.4f} {winner:>8}")
        else:
            s_pct = (1 - s / b) * 100
            f_pct = (1 - f / b) * 100
            winner = 'Freq' if f_pct > s_pct else 'Std'
            print(f"{mk.upper():<8} {s_pct:>+10.1f}% {f_pct:>+10.1f}% {winner:>8}")

    # 保存结果 JSON
    results = {
        'per_sample': per_sample,
        'average': avg,
        'eval_views': eval_views,
        'num_train': len(uid_list),
        'num_val': len(vis_uids),
    }
    results_path = os.path.join(output_dir, 'metrics_full.json')
    with open(results_path, 'w') as fj:
        json.dump(results, fj, indent=2, default=float)
    print(f"\n保存: {results_path}")
    print("完成！")


if __name__ == "__main__":
    main()
