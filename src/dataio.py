import pickle
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

def load_basin_rain(idx):
    with open(dam_networks_path, "rb") as f:
        dams = pickle.load(f)
        nodes = list(set([x for y in dams[idx] for x in y]))
    return pd.read_pickle(basin_rain_path)[nodes]

def load_basin_temp(idx):
    with open(dam_networks_path, "rb") as f:
        dams = pickle.load(f)
        nodes = list(set([x for y in dams[idx] for x in y]))
    return pd.read_pickle(basin_temp_path)[nodes]

def load_basin_shapes(idx):
    with open(dam_networks_path, "rb") as f:
        dams = pickle.load(f)
        nodes = list(set([x for y in dams[idx] for x in y]))
    return gpd.read_file(dam_basins_path).set_index("COMID").loc[nodes]


