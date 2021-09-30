from os.path import join as pj

ROOT = "/home/yoshimi/data/Hydro"

###
### RIVER DATA
###
dam_data_folder = pj(ROOT, "Dams/dam") 
dam_details_path = pj(ROOT, "Dams/meta/dam_details.pkl")

###
### MERIT
###
dam_networks_path = pj(ROOT, "MERIT_BASIN/networks.pkl")
dam_basins_path = pj(ROOT, "MERIT_BASIN/dam_basins.shp")

###
### WEATHER
###
basin_msm_path = pj(ROOT, "MSM/MSM_basin_df")
basin_te_path = pj(ROOT, "TE/MATSIRO")
basin_rivout_path = pj(ROOT, "TE/CAMAFLOOD")