import pandas as pd

mean = lambda x: x.mean(1) if isinstance(x, pd.DataFrame) else x

def result_path(conf):
    """
    """
    dim_reduce = {
        (True,True):"pca_agg",
        (True, False):"pca",
        (False, True):"aggregation",
        (False, False):"none"
    }[(conf["pca_exec"], conf["catchment_aggregation"])]
    
    return f"./predictions/{conf['unit']}/linear/" + \
           f"{conf['past']}_{conf['horizon']}_{conf['hourly']}_{conf['rain_obs']}_{conf['input_snmlt']}" + \
           f"_{conf['input_month']}_{conf['input_discharge']}_{dim_reduce}_{conf['val_end']}"

def format_results(pred, label, ds, date):
    cols = list(set(ds.columns.unique()) - {'inflow_vol'})
    inputs = pd.DataFrame({var:mean(ds[var]) for var in cols}).reindex(date)
    inputs["pred"]=pred
    inputs["y"]=label
    return inputs