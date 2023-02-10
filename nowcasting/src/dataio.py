import glob
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xarray as xr
import torch
from pytorch_lightning import (
    LightningDataModule,
)
from torch.utils.data import DataLoader

def getRadar(dt_list, region='jp', save_dir="/media/yoshimi/9E9401BB94019745/pixel/radar", original_radar_size=(2048, 2048)):
    """
    
    """
    if region == 'jp':
        files = [f'{save_dir}/{datetime.strftime(dt, format="%Y_%m_%d_%H_%M")}.nc' for dt in dt_list]
        return xr.open_mfdataset(files)
    
    if region == 'kor':
        data = []
        for dt in dt_list:
            file_path = f'/home/yoshimi/Research/Hydro/DeepRaNE-master/example_data/radar_{datetime.strftime(dt, format="%Y%m%d%H%M")}.bin.gz'
            with gzip.open(file_path) as f:
                target = f.read()
                result = np.frombuffer(target, 'i2').reshape(original_radar_size)
                img = result
            data.append(img)

        ds = xr.DataArray(data=data, dims=['time','lat', 'lon'],
                       coords={'time':dt_list, 'lat':np.linspace(29, 42, 2048), 'lon':np.linspace(120.5, 137.5, 2048)},
                       name='radar')
        return ds.to_dataset()

def getDates(dt, target=6):
    inputs = pd.date_range(dt - timedelta(hours=1), dt, freq='10min')
    outputs = pd.date_range(dt + timedelta(hours=1), dt + timedelta(hours=target), freq='1H')
    return inputs, outputs

def getDataset(dt, target=6, rect=734, data_path="/media/yoshimi/HDPH-UT/pixel/radar"):
    # center of jp
    h, w = 1800, 1300
    inp, out = getDates(dt, target=target)
    inp = getRadar(inp, save_dir=data_path).isel(lat=slice(h-rect,h+rect), lon=slice(w-rect,w+rect))
    out = getRadar(out, save_dir=data_path).isel(lat=slice(h-rect,h+rect), lon=slice(w-rect,w+rect))
    return inp, out


############# For DGMR modules #############
def getDgmrDates(dt, target=180):
    inputs = pd.date_range(dt - timedelta(minutes=30), dt, freq='10min')
    outputs = pd.date_range(dt + timedelta(minutes=10), dt + timedelta(minutes=target), freq='10min')
    return inputs, outputs

def getDgmrDataset(dt, target=180, rect=128, data_path="/media/yoshimi/9E9401BB94019745/pixel/radar"):
    # center of jp
    h, w = 1800, 1300
    inp, out = getDgmrDates(dt, target=target)
    inp = getRadar(inp, save_dir=data_path).isel(lat=slice(h-rect,h+rect), lon=slice(w-rect,w+rect))
    out = getRadar(out, save_dir=data_path).isel(lat=slice(h-rect,h+rect), lon=slice(w-rect,w+rect))
    return inp, out

class EchoGpvDataset(torch.utils.data.dataset.Dataset):
    """
    A custom dataset class for loading and processing echoGPV data.
    
    Args:
        input_frame (int, optional): Number of input frames. Default is 4.
        output_frame (int, optional): Number of output frames. Default is 18.
        split (str, optional): Split type, either 'train', 'validation' or 'test'. Default is 'test'.
        data_dir (str, optional): Path to the data directory. Default is "/media/yoshimi/9E9401BB94019745/pixel/radar".
        
    Attributes:
        input_frame (int): Number of input frames.
        output_frame (int): Number of output frames.
        total_frame (int): Total number of frames (input_frame + output_frame).
        split (str): Split type, either 'train', 'validation' or 'test'.
        df (pandas.DataFrame): A dataframe containing file names and dates.
        ds (numpy.ndarray): Array containing selected files for the specified split.
    """
    def __init__(self, input_frame=4, output_frame=18, split='test',
                 data_dir="/media/yoshimi/9E9401BB94019745/pixel/radar"):
        super().__init__()
        self.input_frame = input_frame
        self.output_frame = output_frame
        self.total_frame = input_frame + output_frame
        self.split = split
        
        # make file
        files = sorted(glob.glob(data_dir + "/*.nc"))
        dates = [datetime.strptime(Path(file).stem, '%Y_%m_%d_%H_%M') for file in files]
        self.df = pd.DataFrame(files, index=dates, columns=['file'])
        self.df = self.df.reindex(pd.date_range(start=self.df.index[0], end=self.df.index[-1], freq='10min'))
        
        # make dataset
        self._extract_split()
        self._prepare_dataset()
        
    def __len__(self):
        """
        Returns:
            int: The length of the dataset.
        """
        return len(self.ds)
    
    def __getitem__(self, item):
        """
        Get a sample from the dataset.
        
        Args:
            item (int): The index of the sample to retrieve.
            
        Returns:
            tuple: A tuple of two tensors, the input frames and target frames.
        """
        file_list = self.ds[item]
        da = self._crop_center(xr.open_mfdataset(file_list)).to_array()
        tensor = torch.tensor(da.values).transpose(0,1)
        return self._extract_input_and_target_frames(tensor)
    
    def _extract_split(self, train_start='2016-01-01', test_start='2019-01-01', test_end = '2020-01-01'):
        if self.split == 'train':
            self.df = self.df[(self.df.index >= train_start) & (self.df.index < test_start) & (self.df.index.day != 1)]
        elif self.split == 'validation':
            self.df = self.df[(self.df.index >= train_start) & (self.df.index < test_start) & (self.df.index.day == 1)]
        elif self.split == 'test':
            self.df = self.df[(self.df.index >= test_start) & (self.df.index < test_end)]
            
    def _prepare_dataset(self):
        # indexing timeseries dataset
        data_length = len(self.df)
        x_index = np.arange(self.total_frame)[None,:]
        y_index = np.arange(data_length-self.total_frame)[:,None]
        data_index = x_index + y_index
        # drop nan
        ds = np.squeeze(self.df.values[data_index])
        self.ds = pd.DataFrame(ds).dropna(how='any').values
        
    def _extract_input_and_target_frames(self, radar_frames):
        """Extract input and target frames from a dataset row's radar_frames."""
        # We align our targets to the end of the window, and inputs precede targets.
        input_frames = radar_frames[:self.input_frame]
        target_frames = radar_frames[self.input_frame:]
        return input_frames, target_frames
    
    def _crop_center(self, ds, size=256, center_idx=(1800, 1300)):
        return ds.isel(lat=slice(center_idx[0]-size//2,center_idx[0]+size//2), lon=slice(center_idx[1]-size//2,center_idx[1]+size//2))
    

    
class DGMRDataModule(LightningDataModule):
    """
    Example of LightningDataModule for NETCDF dataset.
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        num_workers: int = 1,
        pin_memory: bool = True,
        batch_size: int = 1
    ):
        """
        fake_data: random data is created and used instead. This is useful for testing
        """
        super().__init__()

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size

        self.dataloader_config = dict(
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=8,
            persistent_workers=True,
            # Disable automatic batching because dataset
            # returns complete batches.
            batch_size=None,
        )

    def train_dataloader(self):
        dataloader = DataLoader(EchoGpvDataset(split="train"), batch_size=self.batch_size)
        return dataloader

    def val_dataloader(self):
        train_dataset = EchoGpvDataset(
            split="validation",
        )
        dataloader = DataLoader(train_dataset, batch_size=self.batch_size)
        return dataloader