"""
Simply load images from a folder or nested folders (does not have any split).
"""

import argparse
import logging
from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_image_grid
from .base_dataset import BaseDataset
from .simulation import KubrickInstances

logger = logging.getLogger(__name__)

LOCAL_DATA = '/srv/storage/datasets/cadar/GSO/simulation/train_single_obj'

class NRSimulation(BaseDataset, torch.utils.data.Dataset):
    """Wrapper dataset exposing KubrickInstances in GlueFactory format.

    Adapts the flat output of KubrickInstances (image0, image1, segmentation0, ...)
    into the expected structure with nested view0/view1 dictionaries so that
    two_view_pipeline models (extractor/matcher) receive inputs as
    data["view0"]["image"], data["view1"]["image"].
    Also implements a sample_new_items callback for training configs that
    request dynamic reshuffling each epoch.
    """

    default_conf = {
        "name": "nr_simulation",  # overwrites base '???'
        "data_dir": LOCAL_DATA,
        "max_pairs": -1,
        "return_tensors": True,
        "remove_background": True,
        "splits": [
            'illumination-viewpoint',
            'deformation_3',
            'deformation_3-illumination-viewpoint',
        ],
    }

    def _init(self, conf):
        # Build a plain python dict for the underlying raw dataset, stripping
        # BaseDataset meta keys (like train_batch_size, num_workers, etc.) and
        # avoiding unresolved mandatory placeholders ("???").
        subconf_keys = ["data_dir", "max_pairs", "return_tensors", "splits"]
        subconf = {k: OmegaConf.to_container(conf[k]) if isinstance(conf[k], (list, dict)) else conf[k]
                   for k in subconf_keys if k in conf and conf[k] != "???"}
        # Ensure splits is a plain list
        if isinstance(subconf.get("splits"), tuple):
            subconf["splits"] = list(subconf["splits"])
        self.dataset = KubrickInstances(subconf)

    # The training loop will call dataset.get_dataset(split)
    def get_dataset(self, split):  # noqa: D401 (simple pass-through)
        return self

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Build nested view dicts required by TwoViewPipeline.required_data_keys
        view0 = {"image": sample["image0"]}
        view1 = {"image": sample["image1"]}
        
        if self.conf.remove_background:
            mask0 = sample.get("bgmask0")
            mask1 = sample.get("bgmask1")
            if mask0 is not None:
                view0["image"] = view0["image"] * (mask0 > 0).float() 
            if mask1 is not None:
                view1["image"] = view1["image"] * (mask1 > 0).float()

        data = {
            "view0": view0,
            "view1": view1,
            # Keep auxiliary supervision signals at top-level for uv_matcher
            "segmentation0": sample.get("segmentation0"),
            "segmentation1": sample.get("segmentation1"),
            "uv_coords0": sample.get("uv_coords0"),
            "uv_coords1": sample.get("uv_coords1"),
            "bgmask0": sample.get("bgmask0"),
            "bgmask1": sample.get("bgmask1"),
        }
        # Optional name for logging/debugging
        # (paths might be useful; we drop them here to reduce size)
        return data

    # Called if train.dataset_callback_fn == 'sample_new_items'
    def sample_new_items(self, seed):
        random.seed(seed)
        random.shuffle(self.dataset.all_pairs)

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

def benchmark(args):
    import time
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    conf = {
        "batch_size": 32,
        "num_workers": 8,
        "prefetch_factor": 1,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = NRSimulation(conf)
    loader = DataLoader(dataset, batch_size=conf.batch_size, num_workers=conf.num_workers)
    logger.info("The dataset has %d elements.", len(loader))

    start = time.time()
    for data in tqdm(loader):
        continue
    logger.info("Time: %.2f", time.time() - start)
    

if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_items", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--root", type=Path, default="")
    parser.add_argument("--out", type=Path, default="")
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()

    if args.benchmark:
        benchmark(args)
        exit()
        
    visualize(args)