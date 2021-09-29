from os.path import join as pj

ROOT = "/media/tristan/Elements/Hydro/"

###
### RIVER DATA
###

WATER = pj(ROOT, "water")
dam_data_folder = pj(WATER, "download", "dam")
rain_data_folder = pj(WATER, "download", "rain")
discharge_data_folder = pj(WATER, "download", "discharge")

dam_details_path = pj(WATER, "meta", "dam_details.pkl")
rain_details_path = pj(WATER, "meta", "rain_details.pkl")
discharge_details_path = pj(WATER, "meta", "discharge_details.pkl")

###
### MERIT
###
MERIT = pj(ROOT, "MERIT", "MERIT_BASIN")
merit_rivers_path = pj(MERIT, "Level 1", "riv_pfaf_4_MERIT_Hydro_v07_Basins_v01.shp")
merit_catchments_path =  pj(MERIT, "Level 1", "cat_pfaf_4_MERIT_Hydro_v07_Basins_v01.shp")
dam_networks_path = pj(MERIT, "networks.pkl")
dam_basins_path = pj(MERIT, "dam_basins.shp")

###
### WEATHER
###
msm_surf_path = pj(ROOT, "weather", "MSM_surf")
msm_tmp_basin_path = pj(ROOT, "weather", "MSM_basin_temp/")
msm_rain_basin_path =  pj(ROOT, "weather", "MSM_basin_rain/")

basin_rain_path = pj(ROOT, "weather", "MSM_basin_rain.pkl")
basin_temp_path = pj(ROOT, "weather", "MSM_basin_temp.pkl")

basin_rain_path_old = pj(ROOT, "weather", "MSM_basin_rain_old.pkl")
basin_temp_path_old = pj(ROOT, "weather", "MSM_basin_temp_old.pkl")


###
### TODAY EARTH
###

dam_rivout_hourly_path = "/media/tristan/Elements/Hydro/TE/TE-Japan/Dams/Hourly/rivout.pkl"
basin_var_path = "Coming soon..."
