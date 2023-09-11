from typing import Optional

import torch
import torchdata.datapipes.iter
import webdataset as wds
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

from animatediff.data.dataset import WebVid10M



class WebVidLoader(LightningDataModule):
    def __init__(
        self,
        train: DictConfig,
        batch_size=4,
        num_workers=12,
        sample_size=256,
        sample_stride=4,
        sample_n_frames=16,
    ):
        super().__init__()
        self.train_config = train
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_size = sample_size
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames

    def setup(self, stage: str) -> None:
        print("Preparing datasets")

        self.train_dataset =  WebVid10M(
            csv_path="/mnt/petrelfs/guoyuwei/projects/datasets/webvid/results_2M_val.csv",
            video_folder="/mnt/petrelfs/guoyuwei/projects/datasets/webvid/2M_val",
            sample_size=self.sample_size,
            sample_stride=self.sample_stride,
            sample_n_frames=self.sample_n_frames,
            is_image=False,
        )
        
    def train_dataloader(self):
        train_dataset = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers ,)
        return train_dataset

