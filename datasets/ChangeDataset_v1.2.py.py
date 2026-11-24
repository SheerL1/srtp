#!/usr/bin/env python3
"""
ChangeDataset.py —— 统一变化检测数据集读取接口实现

引用该脚本可以从预处理数据集中获得需要的I1、I2、mask文件

依赖：numpy（loader 本体不引入 torch；ndarray→tensor 由调用方负责，见规范第 4 节）。
"""
from __future__ import annotations
import csv
import numpy as np
from pathlib import Path
from typing import Optional, Callable

# 不要改动默认的datasets文件夹目录结构，参见接口定义契约的目录结构图
#如果一定要改动，使得数据获取脚本和数据集分离，不要改动data的目录结构，使BASE_DIR指向data的实际上级路径

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = BASE_DIR / "data"

# 各数据集差异由预处理脚本内部抹平（规范第 9 节差异表）。

DATASET_REGISTRY = {
    "LEVIR-CD": {
        "root": str(DEFAULT_DATA_ROOT),
        "channels": 3,
        "bands": ["R", "G", "B"],
        "resolution": 0.5,
        "raw_scale": "8bit [0,255]",
        "mask_encoding": "0=未变,1=变化",
        "license": "引用 LEVIR-CD 原论文",
        "scene": "urban",
    },
    "CDD": {
        "root": str(DEFAULT_DATA_ROOT),
        "channels": 3,
        "bands": ["R", "G", "B"],
        "resolution": 3.0,
        "raw_scale": "8bit [0,255]",
        "mask_encoding": "0=未变,1=变化,255=忽略",
        "license": "引用 CDD 原论文",
        "scene": "phenology",
    },
    "OSCD": {
        "root": str(DEFAULT_DATA_ROOT),
        "channels": 4,
        "bands": ["B2", "B3", "B4", "B8"],
        "resolution": 10.0,
        "raw_scale": "反射率×10000",
        "mask_encoding": "0=未变,1=变化,含云掩膜",
        "license": "引用 OSCD 原论文",
        "scene": "multispectral",
    },
}

VALID_SPLITS = ("train", "val", "test")
VALID_SCENES = ("urban", "phenology", "multispectral")
INDEX_COLUMNS = ["tile_id", "I1_file", "I2_file", "mask_file", "scene", "split"]


class ChangeDataset:
    """统一变化检测数据集读取接口（规范 v0.1 第 3 节契约的实现）。"""

    def __init__(self, dataset: str, split: str,
                 transform: Optional[Callable] = None) -> None:
        if dataset not in DATASET_REGISTRY:
            raise ValueError(f"未知数据集: {dataset}，可选 {list(DATASET_REGISTRY)}")
        if split not in VALID_SPLITS:
            raise ValueError(f"未知 split: {split}，可选 {VALID_SPLITS}")

        self.dataset = dataset
        self.split = split
        self.transform = transform
        self._conf = DATASET_REGISTRY[dataset]
        # 该 split 目录：root / 数据集名 / split 名（规范第 7 节目录树）
        self.split_dir = Path(self._conf["root"]) / dataset / split

        self._index = self._load_index()#关于_load_index()方法的疑问见下文，作用为返回一个元素为字典的列表
                                        #这里返回的列表元素数等于数据组数
        self._assert_unique_tile_id()#这个方法同样也是由下文定义的，用于判断是否混入重复数据（tile）

    # ---------- 第 3 节 __len__ ----------
    def __len__(self) -> int:
        return len(self._index)#由上知，此处返回的是数据组数

    # ---------- 第 3 节 __getitem__ + 第 4 节输出 + 第 11 节校验 ----------
    #调用该方法即可获得一个包含所需数据的字典，如何调用参见接口契约
    def __getitem__(self, index: int) -> dict:
        n = len(self._index)
        if n == 0:
            raise IndexError("数据集为空")
        # 支持 Python 负索引约定
        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise IndexError(f"索引越界: {index}（样本数 {n}）")

        rec = self._index[index]#将对应顺序的字典赋给rec
        I1 = self._load_npy(rec["I1_path"])#该方法参见下文，将该路径下的.npy文件赋给I1变量
        I2 = self._load_npy(rec["I2_path"])#同上
        mask = self._load_npy(rec["mask_path"])#同上

        # 第 5 节第 3 点 + 第 3 节 transform 契约：三元组进、三元组出
        if self.transform is not None:
            I1, I2, mask = self.transform(I1, I2, mask)

        # 统一为 float32（规范第 4 节 dtype 约定）
        I1 = I1.astype(np.float32, copy=False)
        I2 = I2.astype(np.float32, copy=False)
        mask = mask.astype(np.float32, copy=False)

        self._validate(I1, I2, mask)

        return {
            "I1": I1,
            "I2": I2,
            "mask": mask,
            "tile_id": rec["tile_id"],
            "split": rec["split"],
            "scene": rec["scene"],
        }

    # ---------- 第 8 节 meta ----------
    @property
    def meta(self) -> dict:#包含一些相关信息，调用方法参见接口契约
        c = self._conf
        return {
            "dataset": self.dataset,
            "channels": c["channels"],
            "bands": list(c["bands"]),
            "resolution": c["resolution"],
            "norm": "divide_255_to_[0,1]",
            "raw_scale": c["raw_scale"],
            "n_train": self._count_split("train"),
            "n_val": self._count_split("val"),
            "n_test": self._count_split("test"),
            "scene": c["scene"],
            "mask_encoding": c["mask_encoding"],
            "license": c["license"],
        }

    # ================= 内部方法 =================
    def _load_index(self) -> list:
        """从 index.csv 加载样本列表（规范 7.1，列名严格对齐，按行序）。"""
        #从index.csv文件中获取每一行，每一行组成一个字典，这些字典再组成一个records列表

        index_csv = self.split_dir / "index.csv"
        if not index_csv.exists():
            raise FileNotFoundError(f"找不到索引文件: {index_csv}")
        records = []
        with open(index_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = set(INDEX_COLUMNS) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"index.csv 缺列: {missing}")
            tiles_dir = self.split_dir / "tiles"
            for row in reader:
                records.append({
                    "tile_id": row["tile_id"],
                    "I1_path": str(tiles_dir / row["I1_file"]),
                    "I2_path": str(tiles_dir / row["I2_file"]),
                    "mask_path": str(tiles_dir / row["mask_file"]),
                    "scene": row["scene"],
                    "split": row["split"],
                })
        return records

    #保证路径正确
    def _load_npy(self, path: str) -> np.ndarray:
        if not Path(path).exists():
            # 规范 7.2：index 引用与磁盘实体不一致 → FileNotFoundError
            raise FileNotFoundError(f"index.csv 引用的文件缺失: {path}")
        return np.load(path)

    #保证错误的数据形态不流向下游
    def _validate(self, I1: np.ndarray, I2: np.ndarray, mask: np.ndarray) -> None:
        """规范第 11 节加载期断言，任一失败即抛错、不静默通过。"""
        # 形状：H=W=256，三者一致；通道维在第 0 维(CHW)
        assert I1.ndim == 3 and I2.ndim == 3 and mask.ndim == 3, "影像/mask 须为 3 维 (C,H,W)"
        assert I1.shape[1] == I2.shape[1] == mask.shape[1] == 256, "H 须=256 且三者一致"
        assert I1.shape[2] == I2.shape[2] == mask.shape[2] == 256, "W 须=256 且三者一致"
        assert I1.shape == I2.shape, "I1 与 I2 形状须完全一致"
        assert mask.shape[0] == 1, "mask 通道维恒为 1"
        # dtype
        assert I1.dtype == np.float32 and I2.dtype == np.float32 and mask.dtype == np.float32, "dtype 须为 float32"
        # 值域
        assert mask.min() >= 0.0 and mask.max() <= 1.0, "mask 值越界"
        assert set(np.unique(mask)).issubset({0.0, 1.0}), "mask 须仅含 {0,1}"
        assert I1.min() >= 0.0 and I1.max() <= 1.0, "I1 值须在 [0,1]"
        assert I2.min() >= 0.0 and I2.max() <= 1.0, "I2 值须在 [0,1]"

    #历遍列表，并将每个元素中tile_id键所对应的值加入ids列表，通过检查列表ids与集合ids长度判断是否混入重复数据
    def _assert_unique_tile_id(self) -> None:
        ids = [r["tile_id"] for r in self._index]
        if len(ids) != len(set(ids)):
            raise ValueError("tile_id 在本 split 内不唯一")

    def _count_split(self, split: str) -> int:
        p = Path(self._conf["root"]) / self.dataset / split / "index.csv"
        if not p.exists():
            return 0
        with open(p, "r", encoding="utf-8") as f:
            return sum(1 for _ in csv.DictReader(f))
        
#自检模块，用于出现问题时检查是否获取数据的脚本出错
if __name__ == "__main__":
    """
    如果更改了
    """
    import tempfile
    import shutil
    import numpy as np

    #测试用函数
    def _make_smoke_data(root: Path, dataset: str, channels: int, scene: str) -> None:
        """在 root/dataset/split 下按规范第 7 节造 2 个 train 样本。"""
        rng = np.random.default_rng(42)
        for split in ("train", "val"):
            split_dir = root / dataset / split
            tiles = split_dir / "tiles"
            tiles.mkdir(parents=True, exist_ok=True)
            rows = []
            for k in range(2):
                sid, r, c = "A001", k, 0
                tile_id = f"{dataset}_{split}_{sid}_{r:03d}_{c:03d}"
                I1 = rng.random((channels, 256, 256), dtype=np.float32)
                I2 = rng.random((channels, 256, 256), dtype=np.float32)
                mask = rng.choice([0, 1], size=(1, 256, 256), p=[0.85, 0.15]).astype(np.float32)
                f1, f2, fm = f"{tile_id}_I1.npy", f"{tile_id}_I2.npy", f"{tile_id}_mask.npy"
                np.save(tiles / f1, I1); np.save(tiles / f2, I2); np.save(tiles / fm, mask)
                rows.append([tile_id, f1, f2, fm, scene, split])
            with open(split_dir / "index.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(INDEX_COLUMNS); w.writerows(rows)

    tmp = Path(tempfile.mkdtemp(prefix="change_smoke_"))
    try:
        # 临时改写 root 指向自检数据目录（不影响用户真实配置）
        DATASET_REGISTRY["LEVIR-CD"]["root"] = str(tmp)
        DATASET_REGISTRY["LEVIR-CD"]["channels"] = 3
        DATASET_REGISTRY["LEVIR-CD"]["bands"] = ["R", "G", "B"]
        DATASET_REGISTRY["LEVIR-CD"]["scene"] = "urban"
        _make_smoke_data(tmp, "LEVIR-CD", 3, "urban")

        ds = ChangeDataset("LEVIR-CD", "train")
        assert len(ds) == 2, f"len 期望 2，实际 {len(ds)}"
        s = ds[0]

        # 第 4 节：六键齐全
        assert set(s.keys()) == {"I1", "I2", "mask", "tile_id", "split", "scene"}, "输出键不符"
        # 第 11 节：形状 / dtype / 值域 / mask 合法 / CHW
        assert s["I1"].shape == (3, 256, 256) and s["I2"].shape == (3, 256, 256)
        assert s["mask"].shape == (1, 256, 256)
        assert s["I1"].dtype == np.float32 and s["mask"].dtype == np.float32
        assert 0.0 <= s["I1"].min() and s["I1"].max() <= 1.0
        assert set(np.unique(s["mask"]).tolist()) <= {0.0, 1.0}
        # tile_id 规范格式：5 段
        assert s["tile_id"].count("_") == 4, f"tile_id 段数不符: {s['tile_id']}"
        # 负索引可用
        assert ds[-1]["tile_id"] == ds[len(ds) - 1]["tile_id"]
        # 越界抛 IndexError
        try:
            ds[999]; raise AssertionError("越界未抛 IndexError")
        except IndexError:
            pass
        # transform 三元组契约：水平翻转后形状/值域不变
        ds_flips = ChangeDataset("LEVIR-CD", "train",
                                 transform=lambda a, b, m: (a[:, :, ::-1], b[:, :, ::-1], m[:, :, ::-1]))
        sf = ds_flips[0]
        assert sf["I1"].shape == (3, 256, 256) and 0.0 <= sf["I1"].max() <= 1.0

        # DataLoader collate 模拟（不依赖 torch，手动堆叠验证可堆性）
        stacked = np.stack([ds[i]["I1"] for i in range(len(ds))], axis=0)
        assert stacked.shape == (2, 3, 256, 256)

        print("自检全部通过 ✅")
        print("  len(train)          =", len(ds))
        print("  输出键              =", sorted(s.keys()))
        print("  I1 shape/dtype      =", tuple(s["I1"].shape), s["I1"].dtype)
        print("  mask shape          =", tuple(s["mask"].shape))
        print("  堆叠 batch shape    =", tuple(stacked.shape))
        print("  tile_id 样例        =", s["tile_id"])
        print("  meta.channels/bands =", ds.meta["channels"], ds.meta["bands"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
