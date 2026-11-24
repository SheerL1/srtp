#!/usr/bin/env python3
"""生成假数据，用于测试 ChangeDataset 接口。
目录结构、tile_id/文件名、index.csv 列与《数据接口规范 v0.1》完全一致。
用法: python make_fake_samples.py --out ./fake_dataset
"""
import argparse
import csv
import numpy as np
from pathlib import Path


def make_fake_tile(channels, size=256):
    I1 = np.random.rand(channels, size, size).astype(np.float32)
    I2 = np.random.rand(channels, size, size).astype(np.float32)
    mask = np.random.choice([0, 1], size=(1, size, size), p=[0.85, 0.15]).astype(np.float32)
    return I1, I2, mask


def generate_dataset(out_root, dataset, channels, scene, n_train, n_val, n_test, seed):
    np.random.seed(seed)
    for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        split_dir = out_root / dataset / split
        tiles_dir = split_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        index_path = split_dir / "index.csv"
        with open(index_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["tile_id", "I1_file", "I2_file", "mask_file", "scene", "split"],
            )
            writer.writeheader()
            # 模拟 sourceImageId：城市/原图目录 ID，如 city_00、city_01
            n_src = max(1, (n + 9) // 10)
            for idx in range(n):
                src = "city_%02d" % (idx // 10)
                row = (idx // 10) % 10
                col = idx % 10
                tile_id = "%s_%s_%s_%03d_%03d" % (dataset, split, src, row, col)
                I1, I2, mask = make_fake_tile(channels)
                I1_file = tile_id + "_I1.npy"
                I2_file = tile_id + "_I2.npy"
                mask_file = tile_id + "_mask.npy"
                np.save(tiles_dir / I1_file, I1)
                np.save(tiles_dir / I2_file, I2)
                np.save(tiles_dir / mask_file, mask)
                writer.writerow({
                    "tile_id": tile_id, "I1_file": I1_file, "I2_file": I2_file,
                    "mask_file": mask_file, "scene": scene, "split": split,
                })
        print("  %s/%s: %d tiles -> %s" % (dataset, split, n, index_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./fake_dataset")
    parser.add_argument("--n-train", type=int, default=5)
    parser.add_argument("--n-val", type=int, default=3)
    parser.add_argument("--n-test", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.out)
    configs = [
        ("LEVIR-CD", 3, "urban"),
        ("CDD", 3, "phenology"),
        ("OSCD", 4, "multispectral"),
    ]
    for dataset, channels, scene in configs:
        print("[%s] generating..." % dataset)
        generate_dataset(out, dataset, channels, scene,
                         args.n_train, args.n_val, args.n_test, args.seed)
    print("done -> %s" % out.resolve())


if __name__ == "__main__":
    main()
