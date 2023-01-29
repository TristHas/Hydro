from datetime import datetime, timedelta
import pandas as pd
import xarray as xr

def getRadar(dt_list, region='jp', save_dir='/media/yoshimi/Elements/pixel/radar', original_radar_size=(2048, 2048)):
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


############# For DGMR function #############
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