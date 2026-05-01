"""
下载 Objaverse 子集 + 用 Blender 渲染 32 视角
使用 hf-mirror.com 镜像下载

用法:
    python scripts/render_objaverse_subset.py --num_objects 50 --output_dir ./data/rendered
"""

import os
import sys
import json
import gzip
import argparse
import random
import subprocess
import time
import urllib.request


HF_MIRROR = "https://hf-mirror.com"
OBJAVERSE_REPO = "datasets/allenai/objaverse/resolve/main"


def load_object_paths():
    """从本地缓存加载 object-paths.json.gz"""
    cache_path = os.path.expanduser("~/.objaverse/hf-objaverse-v1/object-paths.json.gz")
    if not os.path.exists(cache_path):
        # 从镜像下载
        url = f"{HF_MIRROR}/{OBJAVERSE_REPO}/object-paths.json.gz"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"  下载 object-paths.json.gz ...")
        urllib.request.urlretrieve(url, cache_path)
    with gzip.open(cache_path, 'rt') as f:
        return json.load(f)


def download_glb(uid, object_path, output_dir):
    """用 hf-mirror 下载单个 GLB 文件"""
    local_dir = os.path.expanduser("~/.objaverse/hf-objaverse-v1")
    local_path = os.path.join(local_dir, object_path)
    if os.path.exists(local_path):
        return local_path

    url = f"{HF_MIRROR}/{OBJAVERSE_REPO}/{object_path}"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    tmp_path = local_path + ".tmp"
    try:
        urllib.request.urlretrieve(url, tmp_path)
        os.rename(tmp_path, local_path)
        return local_path
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_objects", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="./data/rendered")
    parser.add_argument("--num_views", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blender_script", type=str,
                        default="./scripts/data/objaverse/blender_script.py")
    parser.add_argument("--blender_bin", type=str,
                        default="/opt/blender-4.2.0-linux-x64/blender")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取已有的 UID（跳过）
    existing_uids = set()
    if args.skip_existing:
        for d in os.listdir(args.output_dir):
            full = os.path.join(args.output_dir, d)
            if os.path.isdir(full) and os.path.exists(os.path.join(full, 'intrinsics.npy')):
                existing_uids.add(d)
        print(f"已有 {len(existing_uids)} 个已渲染样本")

    # 从本地缓存加载物体索引
    print("加载物体索引...")
    object_paths = load_object_paths()
    all_uids = list(object_paths.keys())
    print(f"  索引中共 {len(all_uids)} 个物体")

    # 排除已有的，随机选择
    candidate_uids = [u for u in all_uids if u not in existing_uids]
    random.shuffle(candidate_uids)

    need = args.num_objects - len(existing_uids)
    if need <= 0:
        print(f"已有 {len(existing_uids)} 个样本，满足 {args.num_objects} 的需求")
        all_valid = sorted(existing_uids)[:args.num_objects]
        save_uid_list(all_valid, args.output_dir)
        return

    # 多选一些以防部分失败
    selected_uids = candidate_uids[:need + 20]
    print(f"需要新渲染 {need} 个，候选 {len(selected_uids)} 个 UID")

    # 逐个下载 + 渲染
    objects = {}
    for i, uid in enumerate(selected_uids):
        if len(objects) >= need:
            break
        obj_path = object_paths[uid]
        try:
            local = download_glb(uid, obj_path, args.output_dir)
            # 验证文件存在且大小合理
            if not os.path.exists(local) or os.path.getsize(local) < 100:
                print(f"  跳过 {uid}: 文件无效")
                continue
            objects[uid] = local
        except Exception as e:
            print(f"  下载失败 {uid}: {e}")
            continue
        if (i + 1) % 10 == 0:
            print(f"  已下载 {len(objects)}/{need}")
    print(f"下载完成: {len(objects)} 个")

    # 渲染
    from tqdm import tqdm
    blender_script = os.path.abspath(args.blender_script)
    success = 0
    fail = 0
    render_times = []

    pbar = tqdm(objects.items(), total=len(objects), desc="渲染", unit="obj")
    for i, (uid, obj_path) in enumerate(pbar):
        out_check = os.path.join(args.output_dir, uid, 'intrinsics.npy')
        if os.path.exists(out_check):
            success += 1
            pbar.set_postfix(ok=success, fail=fail, last="skip")
            continue

        t0 = time.time()

        cmd = [
            args.blender_bin, "--background", "--python", blender_script, "--",
            "--object_path", obj_path,
            "--output_dir", args.output_dir,
            "--num_images", str(args.num_views),
            "--resolution", str(args.resolution),
            "--engine", "CYCLES",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            dt = time.time() - t0

            if os.path.exists(out_check):
                success += 1
                render_times.append(dt)
                avg_t = sum(render_times) / len(render_times)
                pbar.set_postfix(ok=success, fail=fail, last=f"{dt:.0f}s", avg=f"{avg_t:.0f}s")
            else:
                fail += 1
                pbar.set_postfix(ok=success, fail=fail, last="FAIL")
                tqdm.write(f"  ✗ {uid[:8]}.. 失败: {result.stderr[-150:] if result.stderr else 'unknown'}")
        except subprocess.TimeoutExpired:
            fail += 1
            pbar.set_postfix(ok=success, fail=fail, last="TIMEOUT")
        except Exception as e:
            fail += 1
            pbar.set_postfix(ok=success, fail=fail, last="ERR")

    pbar.close()
    if render_times:
        print(f"\n渲染统计: {success} 成功, {fail} 失败")
        print(f"  平均: {sum(render_times)/len(render_times):.1f}s/个, "
              f"总计: {sum(render_times)/60:.1f}min")
    else:
        print(f"\n渲染完成: {success} 成功, {fail} 失败")

    # 收集所有有效 UID
    all_uids = sorted([
        d for d in os.listdir(args.output_dir)
        if os.path.isdir(os.path.join(args.output_dir, d))
        and os.path.exists(os.path.join(args.output_dir, d, 'intrinsics.npy'))
    ])
    print(f"总共 {len(all_uids)} 个有效样本")
    save_uid_list(all_uids, args.output_dir)


def save_uid_list(uids, output_dir):
    """保存 UID 列表和 train/val split"""
    # 80/20 split
    random.shuffle(uids)
    n_train = max(1, int(len(uids) * 0.8))
    train_uids = sorted(uids[:n_train])
    val_uids = sorted(uids[n_train:])

    meta_dir = os.path.join(output_dir, 'meta')
    os.makedirs(meta_dir, exist_ok=True)

    with open(os.path.join(meta_dir, 'all_uids.json'), 'w') as f:
        json.dump(sorted(uids), f, indent=2)
    with open(os.path.join(meta_dir, 'train_uids.json'), 'w') as f:
        json.dump(train_uids, f, indent=2)
    with open(os.path.join(meta_dir, 'val_uids.json'), 'w') as f:
        json.dump(val_uids, f, indent=2)

    print(f"  保存: {meta_dir}/all_uids.json ({len(uids)} total)")
    print(f"  保存: {meta_dir}/train_uids.json ({len(train_uids)} train)")
    print(f"  保存: {meta_dir}/val_uids.json ({len(val_uids)} val)")


if __name__ == "__main__":
    main()
