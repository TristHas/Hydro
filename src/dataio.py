import pickle, datetime
import networkx as nx
import pandas as pd
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
    """
    """
    assert var_name in msm_variables
    df = pd.read_csv(pj(basin_msm_path, msm_variables[var_name])).set_index("Unnamed: 0")
    df.index = [datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=9)\
                for x in df.index]
    df.columns = [int(x) for x in df.columns]
    with open(dam_networks_path, "rb") as f:
        dams = pickle.load(f)
        nodes = list(set([x for y in dams[idx] for x in y]))
    
    return df[nodes]

def load_basin_te(idx, var_name="snow_amount", daily=False):
    """
    """
    assert var_name in te_variables
    folder = "Daily" if daily else "Hourly"
    df = pd.read_pickle(pj(basin_te_path, folder, "DF", te_variables[var_name]))
    with open(dam_networks_path, "rb") as f:
        dams = pickle.load(f)
        nodes = list(set([x for y in dams[idx] for x in y]))
    return df

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


