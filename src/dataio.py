import pickle, datetime, os
import glob
from pathlib import Path
import datetime
from sklearn.metrics import mean_squared_error
from hydroanalysis.metrics import calculate_nse,calculate_kge
from neuralhydrology.evaluation import metrics
import networkx as nx
import pandas as pd
import numpy as np
import geopandas as gpd
from utils import clean_names, preprocess_func
from config import *

def load_dam(idx):
    return pd.read_pickle(pj(dam_data_folder, str(idx)))

def load_dam_details():
    return pd.read_pickle(dam_details_path)

def load_dam_discharge(idx, max_missing=10, min_data_seq_len=2233):
    dam_data = clean_names(load_dam(idx))
    discharges = preprocess_func(dam_data, max_missing, min_data_seq_len)
    return [x.set_index("datetime")["inflow_vol"] for x in discharges]

def load_dam_network(idx):
    """
        Return dam watershed as a networkx DiGraph
    """
    with open(dam_networks_path, "rb") as f:
        dams = pickle.load(f)
    return nx.DiGraph([x[::-1] for x in dams[idx]])

def load_basin_msm(idx, var_name="rain"):
    with open(dam_networks_path, "rb") as f:
        dams = pickle.load(f)
        nodes = list(set([x for y in dams[idx] for x in y]))
    return pd.DataFrame({node: pd.read_pickle(pj(basin_msm_path, 
                                                 msm_variables[var_name],
                                                 f"{node}.pkl"))\
                        for node in nodes})

def load_basin_te(idx, var_name="snow_amount", daily=False):
    assert var_name in te_variables
    folder = "Daily" if daily else "Hourly"
    with open(dam_networks_path, "rb") as f:
        dams = pickle.load(f)
        nodes = list(set([x for y in dams[idx] for x in y]))
    return pd.DataFrame({node: pd.read_pickle(pj(basin_te_path,
                                                 folder, "DF",
                                                 te_variables[var_name],
                                                 f"{node}.pkl"))\
                        for node in nodes})


def load_dam_rivout(dam_idx, daily=False):
    """
    """
    folder = "Daily" if daily else "Hourly"
    df = pd.read_pickle(pj(basin_rivout_path, folder, "rivout.pkl"))[dam_idx]
    return df

def load_basin_shapes(idx):
    with open(dam_networks_path, "rb") as f:
        dams = pickle.load(f)
        nodes = list(set([x for y in dams[idx] for x in y]))
    return gpd.read_file(dam_basins_path).set_index("COMID").loc[nodes]



#### Data preparation to be done only once

def convert_basin_msm_storage(var_name="rain"):
    df = pd.read_csv(pj(basin_msm_path, old_msm_variables[var_name])).set_index("Unnamed: 0")
    df = df.replace(-9999, np.NaN).interpolate(limit=3)
    df.index = [datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=9) for x in df.index]#  ] #+  
    out_dir = pj(basin_msm_path, msm_variables[var_name])
    os.makedirs(out_dir, exist_ok=True)
    for node in df.columns:   
        df[node].astype("float32").to_pickle(pj(out_dir, f"{node}.pkl"))
        
def convert_basin_te_storage(var_name="snow_amount"):
    assert var_name in te_variables
    for folder in ["Daily", "Hourly"]:
        df = pd.read_pickle(pj(basin_te_path, folder, "DF", old_te_variables[var_name]))
        df = df.replace(-9999, np.NaN).interpolate(limit=3)
        if folder == "Hourly":
            df.index += datetime.timedelta(hours=9)

        out_dir = pj(basin_te_path, folder, "DF", te_variables[var_name])
        os.makedirs(out_dir, exist_ok=True)
        for node in df.columns:   
            df[node].astype("float32").to_pickle(pj(out_dir, f"{node}.pkl"))
        
        
def convert_storage():
    for var_name in msm_variables:
        convert_basin_msm_storage(var_name)
    
    for var_name in te_variables:
        convert_basin_te_storage(var_name)

#### Deprecated        
        
def load_basin_msm_old(idx, var_name="rain"):
    """
    """
    assert var_name in msm_variables
    df = pd.read_csv(pj(basin_msm_path, old_msm_variables[var_name])).set_index("Unnamed: 0")
    df.index = [datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=9)\
                for x in df.index]
    df.columns = [int(x) for x in df.columns]
    with open(dam_networks_path, "rb") as f:
        dams = pickle.load(f)
        nodes = list(set([x for y in dams[idx] for x in y]))
    
    return df[nodes]



def load_basin_te_old(idx, var_name="snow_amount", daily=False):
    """
    """
    assert var_name in te_variables
    folder = "Daily" if daily else "Hourly"
    df = pd.read_pickle(pj(basin_te_path, folder, "DF", old_te_variables[var_name]))
    with open(dam_networks_path, "rb") as f:
        dams = pickle.load(f)
        nodes = list(set([x for y in dams[idx] for x in y]))
    return df[nodes]

### New Observation Methods

def load_rain_gauge(dam_idx):
    """
        Be careful about the missing data (set to -5)
    """
    df = pd.read_pickle(f"{rain_gauge_root}/{dam_idx}.pkl")
    df.index = df.index - datetime.timedelta(hours=9)
    df = df.replace(-5, np.nan).interpolate(limits=3)
    return df

def load_gsmap(dam_idx, gauge_ajdusted=True):
    path = rain_gsmap_ga_root if gauge_ajdusted else rain_gsmap_root
    df = pd.read_pickle(path)
    with open(dam_networks_path, "rb") as f:
        dams = pickle.load(f)
        nodes = list(set([x for y in dams[dam_idx] for x in y]))
    df.index = df.index - datetime.timedelta(hours=9)
    return df[nodes]

### calculate RMSR, NS ... from predicted discharge
def get_results(model='linear', hourly=False,
                past=20, horizon=3, input_snmlt=True,
                rain_obs='msm', input_month=True,
                input_discharge=True, dim_reduce='pca',
                val_end=datetime.datetime(2018,9,1)):
    '''
    read experiment results and caluculate RMSE, MSE, NS, KGE, bias
    
    Parameters
    ----------
    val_end: datetime
        end of validation data 
    
    model: string
        select ML model
    
    dim_reduce: string
        Selecting a Dimensionality Reduction Method
        - pca_agg: PCA + catchment aggregation
        - pca
        - aggregation: Use one data set for the entire catchment area
    
    rain_obs: string
        select observation method
            - msm: assimulated data
            - gauge: gauge only
            - gsmap: prediction from satellite images
            - gsmap+gauge: gsmap ajdusted by gauge
    '''
    if model== 'LSTM':
        return get_lstm_result()
    
    unit = "hourly" if hourly else "daily"
    predict_dir = f"{prediction_dir}/{unit}/{model}"
    dirname = f"{predict_dir}/{past}_{horizon}_{hourly}_{rain_obs}_{input_snmlt}_{input_month}_{input_discharge}_{dim_reduce}_{val_end}"
    if model == 'CaMa-Flood':
        dirname = f"{prediction_dir}/{unit}/CaMa-Flood"
    score = {}

    files = glob.glob(dirname+'/*.pkl')
    for file in files:
        idx = int(Path(file).stem)
        df = pd.read_pickle(file)
        y, pred = df['y'].values, df['pred'].values
        score[idx]=[np.sqrt(mean_squared_error(pred, y)),
                      mean_squared_error(pred, y),
                      calculate_nse(y, pred, np.zeros_like(pred)),
                      calculate_kge(y, pred, np.zeros_like(pred))['kge'],
                      y.mean()-pred.mean()]

    score = pd.DataFrame(score, index=['RMSE','MSE','NSE','KGE','bias']).T  
    return score

def get_prediction(idx, model='linear', hourly=False,
                past=20, horizon=3, input_snmlt=True,
                rain_obs='msm', input_month=True,
                input_discharge=True, dim_reduce='pca',
                val_end=datetime.datetime(2018,9,1)):
    '''
    read experiment results
    
    Parameters
    ----------
    val_end: datetime
        end of validation data 
    
    model: string
        select ML model
    
    dim_reduce: string
        Selecting a Dimensionality Reduction Method
        - pca_agg: PCA + catchment aggregation
        - pca
        - aggregation: Use one data set for the entire catchment area
    
    rain_obs: string
        select observation method
            - msm: assimulated data
            - gauge: gauge only
            - gsmap: prediction from satellite images
            - gsmap+gauge: gsmap ajdusted by gauge
    '''
    if model == 'LSTM':
        return get_lstm_prediction(idx)
    
    unit = "hourly" if hourly else "daily"
    predict_dir = f"{prediction_dir}/{unit}/{model}"
    dirname = f"{predict_dir}/{past}_{horizon}_{hourly}_{rain_obs}_{input_snmlt}_{input_month}_{input_discharge}_{dim_reduce}_{val_end}"
    if model == 'CaMa-Flood':
        dirname = f"{prediction_dir}/{unit}/CaMa-Flood"
    df = pd.read_pickle(f"{dirname}/{idx}.pkl")
    return df

def get_lstm_result():
    score = {}
    for path in glob.glob('./runs/*/'):
        path = Path(path)
        idx = int(path.name)
        
        if not (path / "test" / "model_epoch050" / "test_results.p").exists():
            continue
        
        with open(path / "test" / "model_epoch050" / "test_results.p", "rb") as fp:
            results = pickle.load(fp)[str(idx)]['1D']['xr']
        
        qobs = results['qobs_obs']
        qsim = results['qobs_sim']
        
        values = metrics.calculate_all_metrics(qobs.isel(time_step=0), qsim.isel(time_step=0))
        score[idx] = values
    
    return pd.DataFrame(score).T

def get_lstm_prediction(idx):
    path = Path(f'./runs/{idx}/')

    with open(path / "test" / "model_epoch050" / "test_results.p", "rb") as fp:
        results = pickle.load(fp)[str(idx)]['1D']['xr']

    qobs = results['qobs_obs'][:,-1]
    qsim = results['qobs_sim'][:,-1]
    df = pd.concat([qobs.to_dataframe(), qsim.to_dataframe()], axis=1).dropna(how='any').drop('time_step', axis=1)
    df.columns = ['y', 'pred']
    return df