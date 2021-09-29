import numpy as np
from shapely.geometry import Point

UP_COLUMNS = ["up1", "up2", "up3", "up4"]

def test_df(df, detail):
    return [Point(detail.loc[idx, "coordinates"][::-1]).within(df["geometry"].loc[idx]) for idx in df.index]

def center(X):
    return X.centroid.coords.xy[0][0], X.centroid.coords.xy[1][0]

def filter_candidates(basins, coords, N=100):
    x = np.array(basins["geometry"].apply(center).values.tolist())
    out = np.sqrt(((x[:,None]-coords[None])**2).sum(-1))
    return np.argsort(out, axis=0)[:N].T

def select_candidates(arg, basins, coords, idx, N=100):
    candidates, matches = arg[:,:N], []
    for i in range(coords.shape[0]):
        p = Point(coords[i])
        match = [p.within(x) for x in basins.loc[candidates[i]]["geometry"].tolist()]
        matches.append(match)
    matches = np.array(matches)
    X,Y = np.where(matches)
    df = basins.loc[candidates[X,Y]].copy()
    df.index = idx[X]
    return df

def map_dam_to_basins(dam_details, basins, N=100):
    coords = np.array(dam_details["coordinates"].values.tolist())[:,::-1]
    idx = dam_details.index
    arg = filter_candidates(basins, coords, N=N)
    df = select_candidates(arg, basins, coords, idx)
    assert all(test_df(df, dam_details))
    return df

def infer_river_networks(df, rivers):
    conv = {k:v for k,v in zip(df["COMID"], df.index)}
    networks, nodes, all_nodes = {},{},[]
    for idx in df["COMID"].values:
        net = get_network_edges(rivers, idx)
        networks[conv[idx]]=net
    return networks

def get_network_edges(rivers, idx):
    next_nodes = get_parent(rivers, idx)
    edges = [(idx, idx)] + [(idx, x) for x in next_nodes]
    while(len(next_nodes)>0):
        parent = next_nodes.pop()
        child = get_parent(rivers, parent)
        edges.extend( [(parent, x) for x in child])
        next_nodes = next_nodes.union(child)
    return edges

def get_parent(rivers, idx):
    return set(rivers.loc[idx][UP_COLUMNS].tolist()) - {0}