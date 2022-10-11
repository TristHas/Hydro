import sys
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.append("../src")
from dataio import *
from config import (RAIN_OBS, TRAIN_END, VAL_END, PCA,
                     CATCHMENT_AGGREGATION, HOURLY)
from tqdm.notebook import tqdm
from utils import HydroDataTransform


def add_snmlt_imit(ds):
    snmlt_imit = ds['temp'].values * ds['gsnwl'].values
    snmlt_imit = pd.DataFrame(snmlt_imit).set_index(ds.index)
    snmlt_imit.columns = ['snmlt_imit' for i in snmlt_imit.columns]
    return pd.concat([snmlt_imit, ds], axis=1)

def get_daily_dataset(idx, rain_obs='msm'):
    '''
    get concatenated pd.Dataframe (dayly data)
    
    
    Parameters
    ----------
    idx: int
        Dam idx(get this id's data like rainfall, temperature, snow melt etc,,,)
    
    rain_obs: string
        select observation method
            - msm: assimulated data
            - gauge: gauge only
            - gsmap: prediction from satellite images
            - gsmap+gauge: gsmap ajdusted by gauge
    '''
    if rain_obs == 'msm':
        rain = load_basin_msm(idx, "rain")
    elif rain_obs == 'gauge':
        rain = load_rain_gauge(idx)
    elif rain_obs == 'gsmap':
        rain = load_gsmap(idx, gauge_ajdusted=False)
    elif rain_obs == 'gsmap+gauge':
        rain = load_gsmap(idx, gauge_ajdusted=True)
    temp = load_basin_msm(idx, "temp")
    snmlt = load_basin_te(idx,var_name="snow_melt",daily=True)
    gsnwl = load_basin_te(idx,var_name="snow_amount",daily=True)

    rain.index = rain.index.round('D')
    rain.index.name='date'
    rain=rain.groupby(level=['date']).sum()

    temp.index = temp.index.round('D')
    temp.index.name='date'
    temp=temp.groupby(level=['date']).mean()

    rain.columns = ['rain' for i in rain.columns]
    temp.columns = ['temp' for i in temp.columns]
    snmlt.columns = ['snmlt' for i in snmlt.columns]
    gsnwl.columns = ['gsnwl' for i in gsnwl.columns]

    y = pd.concat(load_dam_discharge(idx))
    y.index = y.index.round('D')
    y.index.name='date'
    y = y.groupby(level=['date']).mean()
    
    dataset = pd.concat([rain,temp,snmlt,gsnwl,y],axis=1)
    
    ### add snow imit
    dataset = add_snmlt_imit(dataset)
    ### -------------
    
    start, end = dataset.index.min(), dataset.index.max()
    return dataset.reindex(pd.date_range(start,end,freq='D'))

def get_hourly_dataset(idx, rain_obs='msm'):
    '''
    get concatenated pd.Dataframe (hourly data)
    
    
    Parameters
    ----------
    idx: int
        Dam idx(get this id's data like rainfall, temperature, snow melt etc,,,)
    
    rain_obs: string
        select observation method
            - msm: assimulated data
            - gauge: gauge only
            - gsmap: prediction from satellite images
            - gsmap+gauge: gsmap ajdusted by gauge
    '''
    if rain_obs == 'msm':
        rain = load_basin_msm(idx, "rain")
    elif rain_obs == 'gauge':
        rain = load_rain_gauge(idx)
    elif rain_obs == 'gsmap':
        rain = load_gsmap(idx, gauge_ajdusted=False)
    elif rain_obs == 'gsmap+gauge':
        rain = load_gsmap(idx, gauge_ajdusted=True)
    temp = load_basin_msm(idx, "temp")
    snmlt = load_basin_te(idx,var_name='snow_melt',daily=True)
    t_index = pd.DatetimeIndex(pd.date_range(start=snmlt.index[0],end=snmlt.index[-1],freq='h'))
    snmlt = snmlt.reindex(t_index,method='ffill')
    gsnwl = load_basin_te(idx,var_name='snow_amount',daily=True)
    t_index = pd.DatetimeIndex(pd.date_range(start=gsnwl.index[0],end=gsnwl.index[-1],freq='h'))
    gsnwl = gsnwl.reindex(t_index,method='ffill')
    
    rain.columns = ['rain' for i in rain.columns]
    temp.columns = ['temp' for i in temp.columns]
    snmlt.columns = ['snmlt' for i in snmlt.columns]
    gsnwl.columns = ['gsnwl' for i in gsnwl.columns]
    
    input_data = pd.concat([rain,temp],axis=1)
    input_data = input_data.dropna(how='any')

    y = pd.concat(load_dam_discharge(idx))
    
    input_data = pd.concat([rain,temp,snmlt,gsnwl,y],axis=1)
    start, end = input_data.index.min(), input_data.index.max()
    return input_data.reindex(pd.date_range(start,end,freq='H'))

def generate_dataset(df,*, past=4, horizon=3, 
                     input_snmlt=False, input_discharge=False,
                     hourly=False, input_month=False, **kwargs):
    '''
    generate dataset to input ML model
    
    
    Parameters
    ----------
    df: pd.Dataframe
        from get_daily_ds or get_hourly_dataset
        
    past: int
        num of using past data
        
    horizon: int 
        Time(days or hours) to predicted date and time
    
    input_snmlt: bool or imit
        whether to use the snmlt data
        
    input_discharge: bool
        whether to use pase discharge data
        
    input_month: bool
        whether to use target month data
    
    houly: bool
        whether time unit is hourly or daily  
    '''
    if input_snmlt == 'gsnwl':
        data = ['rain','temp','gsnwl']
    elif input_snmlt:
        data = ['rain','temp','snmlt']
    else:
        data = ['rain','temp']
    
    x_np = df.loc[:,data].values
    T = past + horizon
    
    x_step = np.arange(T+1)[None]
    y_step = np.arange(x_np.shape[0]-T-1)[:,None]
    x_idx = x_step + y_step
    y_idx = y_step + T
    y_idx = y_idx.reshape(-1,)
    
    X = x_np[x_idx]
    X = x_np[x_idx].reshape(X.shape[0], -1)
    Y = df.values[y_idx, -1]
    
    if input_discharge:
        discharge_idx = x_idx[:,:past]
        x_discharge = df.values[discharge_idx,-1]
        X = np.concatenate([X,x_discharge],axis=1)
    
    if input_month:
        month=df.iloc[y_idx].index.month.values-1
        month_one_hot = np.identity(12)[month]
        X = np.concatenate([X,month_one_hot],axis=1)
    
    no_missing = (~np.isnan(X).any(axis=1)) & (~np.isnan(Y))
        
    return X[no_missing], Y[no_missing] , df.iloc[y_idx[no_missing]].index


def aggregate_per_catchment(df):
    sum = lambda x: x.sum(axis=1) if isinstance(x, pd.DataFrame) else x
    mean = lambda x: x.mean(axis=1) if isinstance(x, pd.DataFrame) else x
    cols = list(set(df.columns.unique()) - {'temp','inflow_vol'})
    aggregated_df = pd.DataFrame({var:sum(df[var]) for var in cols})
    aggregated_df['temp'] = mean(df['temp'])
    aggregated_df['inflow_vol'] = df['inflow_vol']
    return aggregated_df

def ds_generator(idxs, 
                 rain_obs=RAIN_OBS, 
                 train_end=TRAIN_END, 
                 val_end=VAL_END,
                 catchment_aggregation=CATCHMENT_AGGREGATION, 
                 pca_exec=PCA,
                 hourly=HOURLY, **kwargs):
    """
    """
    transform = HydroDataTransform(pca_exec=pca_exec)

    get_dataset = get_hourly_dataset if hourly \
             else get_daily_dataset
        
    for idx in tqdm(idxs):
        ds = aggregate_per_catchment(get_dataset(idx, rain_obs=rain_obs)) if catchment_aggregation \
        else get_dataset(idx, rain_obs=rain_obs)
        #memory error if hourly
        if idx == 1368060475060:
             continue

        tr_ds = ds[ds.index < train_end]
        train_x, train_y, train_date = generate_dataset(df = tr_ds, **kwargs)
        transform.fit(train_x)

        val_ds = ds[(ds.index >= train_end) & (ds.index < val_end)]
        val_x, val_y, val_date = generate_dataset(df = val_ds,**kwargs)

        te_ds = ds[ds.index >= val_end]
        test_x, test_y, test_date = generate_dataset(df = te_ds, **kwargs)
        yield (idx, ds, transform(train_x), train_y, train_date,
               transform(val_x), val_y, val_date,
               transform(test_x), test_y, test_date)