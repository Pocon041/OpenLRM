"""
批量 Triplane 残差分析脚本

对 data/rendered/ 下所有已渲染物体依次执行：
1. GT Triplane 优化
2. 残差可视化

用法：
    python scripts/batch_analyze.py \
        --model_name zxhezexin/openlrm-mix-base-1.1 \
        --infer_config ./configs/infer-b.yaml \
        --num_iters 2000 --lr 0.01 --grid_size 128
"""

import os
import sys
import glob
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="批量 Triplane 残差分析")
    parser.add_argument("--render_dir", type=str, default="./data/rendered")
    parser.add_argument("--model_name", type=str, default="zxhezexin/openlrm-mix-base-1.1")
    parser.add_argument("--infer_config", type=str, default="./configs/infer-b.yaml")
    parser.add_argument("--triplane_dir", type=str, default="./exps/gt_triplane")
    parser.add_argument("--vis_dir", type=str, default="./exps/residual_vis")
    parser.add_argument("--num_iters", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--grid_size", type=int, default=128)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 收集所有已渲染完成的物体目录
    obj_dirs = sorted(glob.glob(os.path.join(args.render_dir, "*")))
    obj_dirs = [d for d in obj_dirs if os.path.isdir(d)]

    valid_dirs = []
    for d in obj_dirs:
        rgba_dir = os.path.join(d, "rgba")
        intrinsics = os.path.join(d, "intrinsics.npy")
        if os.path.isdir(rgba_dir) and os.path.isfile(intrinsics):
            img_count = len([f for f in os.listdir(rgba_dir) if f.endswith(".png")])
            if img_count >= 32:
                valid_dirs.append(d)
            else:
                print(f"[SKIP] {os.path.basename(d)}: only {img_count} images")
        else:
            print(f"[SKIP] {os.path.basename(d)}: incomplete data")

    print(f"\n Found {len(valid_dirs)} valid objects to process\n")

    for idx, data_dir in enumerate(valid_dirs):
        obj_id = os.path.basename(data_dir)
        triplane_path = os.path.join(args.triplane_dir, f"{obj_id}_triplane.pt")

        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(valid_dirs)}] {obj_id}")
        print(f"{'='*60}")

        # Step 1: GT Triplane 优化
        if args.skip_existing and os.path.isfile(triplane_path):
            print(f"  Triplane already exists, skipping optimization")
        else:
            print(f"  Step 1: Optimizing GT Triplane...")
            cmd = [
                sys.executable, os.path.join(script_dir, "optimize_gt_triplane.py"),
                "--data_dir", data_dir,
                "--model_name", args.model_name,
                "--infer_config", args.infer_config,
                "--output_dir", args.triplane_dir,
                "--num_iters", str(args.num_iters),
                "--lr", str(args.lr),
            ]
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                print(f"  [ERROR] Optimization failed for {obj_id}, skipping")
                continue

        # Step 2: 残差可视化
        vis_check = os.path.join(args.vis_dir, f"{obj_id}_error_histogram.png")
        if args.skip_existing and os.path.isfile(vis_check):
            print(f"  Visualization already exists, skipping")
        else:
            print(f"  Step 2: Generating visualizations...")
            cmd = [
                sys.executable, os.path.join(script_dir, "visualize_triplane_residual.py"),
                "--triplane_path", triplane_path,
                "--model_name", args.model_name,
                "--output_dir", args.vis_dir,
                "--grid_size", str(args.grid_size),
            ]
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                print(f"  [ERROR] Visualization failed for {obj_id}")

    print(f"\n{'='*60}")
    print(f"All done! Results in:")
    print(f"  Triplanes: {args.triplane_dir}")
    print(f"  Visualizations: {args.vis_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
