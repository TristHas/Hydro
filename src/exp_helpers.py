import os
import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm.notebook import tqdm

from dataio import load_gsm_fcst
from dataset import generate_forecast_ds, aggregate_per_catchment, get_hourly_dataset
from utils import HydroDataTransform

mean = lambda x: x.mean(1) if isinstance(x, pd.DataFrame) else x

def result_path(conf):
    """
    """
    dim_reduce = {
        (True,True):"pca_agg",
        (True, False):"pca",
        (False, True):"aggregation",
        (False, False):"None"
    }[(conf["pca_exec"], conf["catchment_aggregation"])]
    
    return f"./predictions/{conf['unit']}/{conf['model']}/" + \
           f"{conf['past']}_{conf['horizon']}_{conf['hourly']}_{conf['rain_obs']}_{conf['input_snmlt']}" + \
           f"_{conf['input_month']}_{conf['input_discharge']}_{dim_reduce}_{conf['val_end']}"

def format_results(pred, label, ds, date):
    cols = list(set(ds.columns.unique()) - {'inflow_vol'})
    inputs = pd.DataFrame({var:mean(ds[var]) for var in cols}).reindex(date)
    inputs["pred"]=pred
    inputs["y"]=label
    return inputs


def execute_forecast_exp(idxs,
                         observed_hour,*,
                         past=20,
                         horizon=3,
                         hourly=False,
                         rain_obs='gauge',
                         input_snmlt = False,
                         input_discharge = False,
                         input_month=False,
                         input_fcst=False,
                         train_end=None,
                         pca_exec='pca',
                         val_end=None,
                         res_dir='test',
                         **kwargs):
    
    results = {}
    
    # change
    dirname = f'{res_dir}/{horizon}'
    os.makedirs(dirname, exist_ok=True)

    for idx in tqdm(idxs):
        #init models
        model = LinearRegression()
        transform = HydroDataTransform(pca_exec=pca_exec)
        
        # load data
        past_ds = aggregate_per_catchment(get_hourly_dataset(idx, rain_obs=rain_obs))
        fcst_ds = load_gsm_fcst(idx)
        
        #memory error if hourly
        if idx == 1368060475060:
             continue
    
        
        tr_past, tr_fcst = past_ds[past_ds.index < train_end], fcst_ds[fcst_ds.index < train_end]
        train_x, train_y = generate_forecast_ds(tr_past, tr_fcst,
                                                observed_hour=observed_hour,
                                                past=past,
                                                horizon=horizon,
                                                hourly=hourly,
                                                input_discharge = input_discharge,
                                                input_month=input_month,
                                                input_fcst=input_fcst)


        test_past, test_fcst = past_ds, fcst_ds
        test_x, test_y = generate_forecast_ds(test_past, test_fcst,
                                                observed_hour=observed_hour,
                                                past=past,
                                                horizon=horizon,
                                                hourly=hourly,
                                                input_discharge = input_discharge,
                                                input_month=input_month,
                                                input_fcst=input_fcst)
        
        

        # fit and predict
        transform.fit(train_x.values)
        model.fit(transform(train_x.values), train_y.values)
        predict = model.predict(transform(test_x.values))
        

        res = pd.DataFrame(data=np.concatenate([predict[:,None], test_y.values[:,None]], axis=1), index=test_y.index, columns=['pred', 'y'])
        res.to_pickle(f"{dirname}/{idx}.pkl")
        
    return 