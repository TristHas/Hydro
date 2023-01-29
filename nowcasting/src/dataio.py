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
    
def plot(ds, class_num=100):
    var_name = list(ds.data_vars.keys())[0]
    gall = ds[var_name].hvplot(title='precipitation', groupby='time', clim=(0,class_num-1), width=500, height=500, alpha=0.6, cmap='jet', geo=True, rasterize=True)
    return gsmap * gall

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