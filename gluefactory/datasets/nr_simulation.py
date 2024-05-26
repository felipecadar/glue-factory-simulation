"""
Simply load images from a folder or nested folders (does not have any split).
"""

import argparse
import logging
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from ..settings import DATA_PATH
from ..utils.image import ImagePreprocessor, load_image
from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_image_grid
from .base_dataset import BaseDataset
import h5py
import json

logger = logging.getLogger(__name__)

def load_sample(rgb_path:Path):
    import cv2
    rgb_path = cv2.imread(rgb_path)
    image = cv2.imread(rgb_path)
    mask = cv2.imread(rgb_path.replace('rgba', 'bgmask'), cv2.IMREAD_UNCHANGED)
    uv_coords = cv2.imread(rgb_path.replace('rgba', 'uv'), cv2.IMREAD_UNCHANGED)
    segmentation = cv2.imread(rgb_path.replace('rgba', 'segmentation'), cv2.IMREAD_UNCHANGED)

    sample = {}
    sample['image'] = image
    sample['mask'] = mask
    sample['uv_coords'] = uv_coords
    sample['segmentation'] = segmentation

    return sample

class NRSimulation(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        "data_dir": "../simulation_h5/",
        "splits": ["deformation_1", "deformation_2", "deformation_3"],
        "mode": "train",
    }

    def _init(self, conf):
        self.root = DATA_PATH / conf.data_dir
        if not self.root.exists():
            logger.error("Dataset not found.")
        self.h5 = h5py.File(self.root / "images.h5", "r")
        pairs = json.load(open(self.root / "selected_pairs.json", "r"))
        self.mode = self.conf.mode
        # filter by splits
        self.items = []
        for split in pairs:
            if split not in self.conf.splits:
                continue
            self.items.extend(pairs[split])

    def get_dataset(self, split):
        assert split in ["train", "val"], f"Invalid split. {split}"

        if split == "train":
            return NRSimulation({
                "mode": "train",
               	"data_dir": self.conf.data_dir,
               	"splits": self.conf.splits,
            })
        else:
            return NRSimulation({
                "mode": "val",
                "data_dir": self.conf.data_dir,
                "splits": self.conf.splits,
            })

    def load_sample(self, path):
        data = {
            'image': self.h5['image'][path][()],
            'mask': self.h5['mask'][path][()],
            'uv_coords': self.h5['uv_coords'][path][()],
            'segmentation': self.h5['segmentation'][path][()]
        }
        data['image'] = torch.tensor(data['image']).permute(2, 0, 1).float() / 255
        return data

    def __getitem__(self, idx):
        path0, path1 = self.items[idx]
        sample0 = self.load_sample(path0)
        sample1 = self.load_sample(path1)
        data = {
            "view0": {
                "image": sample0["image"],
            },
            "view1": {
                "image": sample1["image"],
            }
        }
        data.update({
            key + "0": value for key, value in sample0.items() if key != "image"
        })
        data.update({
            key + "1": value for key, value in sample1.items() if key != "image"
        })
        return data

    def __len__(self):
        if self.mode  == 'train':
            return len(self.items)
        else:
            return 500

def visualize(args):
    conf = {
        "batch_size": 1,
        "num_workers": 8,
        "prefetch_factor": 1,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = NRSimulation(conf)
    loader = dataset.get_data_loader("train")
    logger.info("The dataset has %d elements.", len(loader))

    with fork_rng(seed=dataset.conf.seed):
        images = []
        for _, data in zip(range(args.num_items), loader):
            images.append(
                [data[f"view{i}"]["image"][0].permute(1, 2, 0) for i in range(2)]
            )
    plot_image_grid(images, dpi=args.dpi)
    plt.tight_layout()
    plt.savefig("nr_simulation.png")

def write_sample(path:str, dataset_path:Path, h5:h5py.File):
    sample = load_sample(dataset_path / path)
    for key, value in sample.items():
        h5[key].create_dataset(path, data=value)

    return True

def make_h5(root:Path, out:Path):
    import json
    from tqdm import tqdm

    # make dir for out
    out.mkdir(parents=True, exist_ok=True)
    selected_json = root / "selected_pairs.json"

    selected_pairs = json.load(open(selected_json))
    json.dump(selected_pairs, open(out / "selected_pairs.json", "w"))

    unique_images = set()
    for split in selected_pairs.keys():
        for pair in selected_pairs[split]:
            for img in pair:
                unique_images.add(img)
    unique_images = list(unique_images)
    print(f"Found {len(unique_images)} unique images.")

    h5_file = out / "images.h5"
    h5 = h5py.File(h5_file, "w")

    h5.create_group("image")
    h5.create_group("mask")
    h5.create_group("uv_coords")
    h5.create_group("segmentation")

    for path in tqdm(unique_images):
        sample = load_sample(root / path)
        for key, value in sample.items():
            h5[key].create_dataset(path, data=value)

    h5.close()

if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_items", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--make_h5", action="store_true", default=False)
    parser.add_argument("--root", type=Path, default="")
    parser.add_argument("--out", type=Path, default="")
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()

    if args.make_h5:
        assert args.root.exists(), "Root does not exist."
        make_h5(args.root, args.out)

    visualize(args)
