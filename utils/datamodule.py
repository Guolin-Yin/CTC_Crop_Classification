import numpy as np
import time
import torch
from typing import Any, Union
from pathlib import Path
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .settings.config import RANDOM_SEED, IMG_SIZE
from .dataset import CTCdataset, CTCdataset_train

# Set seed for everything
pl.seed_everything(RANDOM_SEED)

class CTCDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, num_workers=4, collate_fn=None):
        super().__init__()
        self.collate_fn   = collate_fn
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        self.dataset_train = None
        self.dataset_eval  = None
        self.dataset_test  = None
    def setup(self, stage=None):
        assert stage in ['fit', 'test'], f'Stage : "{stage}" must be fit or test!'
        if stage == 'fit':
           self.dataset_train  = CTCdataset_train()
           self.dataset_eval   = CTCdataset( 'val',)
        else: 
            self.dataset_test = CTCdataset( 'test', )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = self.num_workers,
            pin_memory  = True,
            collate_fn  = self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_eval,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = True,
            # collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = True,
            # collate_fn=self.collate_fn
        )
