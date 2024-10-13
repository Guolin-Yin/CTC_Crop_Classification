import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import pytorch_lightning as pl
from collections import defaultdict
from .settings.config import RANDOM_SEED, LINEAR_ENCODER_CTC, LINEAR_ENCODER, SELECTED_CLASSES_CTC
import torch
import random
from collections import defaultdict

# Set seed for everything
# pl.seed_everything(RANDOM_SEED)
def normalize_img(img):
    min_val = img.min(axis=(1, 2), keepdims=True)
    max_val = img.max(axis=(1, 2), keepdims=True)
    norm_img = (img - min_val) / (max_val - min_val)
    return norm_img
class CTCdataset(Dataset):
    def __init__(self, mode):
        data_dir = Path('./data/dataset')
        self.mode = mode
        self.data_paths = list(data_dir.glob(f'{mode}/**/*_data.npy'))
        assert len(self.data_paths) > 0, 'No data'
        # self.filter_data()
    def filter_data(self):
        if self.mode == 'train':
            self.n_samples = 2000+1
        else:
            self.n_samples = 200+1
        # Dictionary to hold paths grouped by label
        label_dict = defaultdict(list)
        for path in self.data_paths:
            label = path.parent.name
            label_dict[label].append(path)
        
        # Limiting the number of samples per label to 1000
        for label, paths in label_dict.items():
            if len(paths) >= self.n_samples:
                label_dict[label] = random.sample(paths, self.n_samples)  # Sample without replacement

        # Flatten the dictionary back to a list
        self.data_paths = [path for paths in label_dict.values() for path in paths]
        
    def __len__(self):
        return int(len(self.data_paths))
    def __getitem__(self, idx):
        data_path = self.data_paths[idx]

        time_path = Path(data_path.parent / data_path.stem.replace('data','time')).with_suffix('.npy')

        label = data_path.parent.name

        data = np.load(data_path)
        time = np.load(time_path)


        return {'data':normalize_img(data) ,'time':time,'label':LINEAR_ENCODER[label], 'path':str(data_path)}
        # return {'data':data ,'time':time,'label':LINEAR_ENCODER[label], 'path':str(data_path)}


class CTCdataset_train(Dataset):
    def __init__(self):
        data_dir = Path('./data/dataset')
        self.data_paths = list(data_dir.glob(f'train_shuffled_pixels/pixels_*_data.npy'))
        self.get_paths = lambda path,item: Path(path.parent / path.stem.replace('data',item)).with_suffix('.npy')
        assert len(self.data_paths) > 0, 'No data'
    def __len__(self):
        return int(len(self.data_paths))
    def __getitem__(self, idx):
        data_path = self.data_paths[idx]

        time_path = self.get_paths(data_path,'time')

        label_path = self.get_paths(data_path,'labels')

        data  = np.load(data_path)
        label = np.load(label_path)
        time  = np.load(time_path)


        # return {'data':data ,'time':time,'label':label, 'path':str(data_path)}
        return {'data':normalize_img(data) ,'time':time,'label':label, 'path':str(data_path)}
