import datetime
from config_perso import *

old_msm_variables = {
    "rain":"APCP_P8_L1_GLL0_acc1h.csv",
    "temp":"TMP_P0_L103_GLL0.csv",
    #"middle_cloud":"MCDC_P0_L1_GLL0.csv",
    #"low_cloud":"LCDC_P0_L1_GLL0.csv"  ,      
    "total_cloud":"TCDC_P0_L1_GLL0.csv",
    "wind_U":"UGRD_P0_L103_GLL0.csv",
    "wind_V":"VGRD_P0_L103_GLL0.csv",
    "ground_level_pressure":"PRES_P0_L1_GLL0.csv",
    "seal_level_pressure":"PRMSL_P0_L101_GLL0.csv",
    "relative_humidity":"RH_P0_L103_GLL0.csv",  
}

old_te_variables = {
    "snow_amount":    "GLSNW.pkl", 
    "snow_melt":    "SNMLT.pkl"  ,
    "ice_melt":    "ICEMLT.pkl",  
    "ice_sub":    "ICESUB.pkl" , 
    "snow_freeze":    "SNFRZ.pkl"  ,
    "snow_sub":    "SNSUB.pkl",
}

msm_variables={
'rain': 'APCP_P8_L1_GLL0_acc1h',
 'temp': 'TMP_P0_L103_GLL0',
 'total_cloud': 'TCDC_P0_L1_GLL0',
 'wind_U': 'UGRD_P0_L103_GLL0',
 'wind_V': 'VGRD_P0_L103_GLL0',
 'ground_level_pressure': 'PRES_P0_L1_GLL0',
 'seal_level_pressure': 'PRMSL_P0_L101_GLL0',
 'relative_humidity': 'RH_P0_L103_GLL0'
}

te_variables = {
    "snow_amount":    "GLSNW", 
    "snow_melt":    "SNMLT"  ,
    "ice_melt":    "ICEMLT",  
    "ice_sub":    "ICESUB" , 
    "snow_freeze":    "SNFRZ"  ,
    "snow_sub":    "SNSUB",
}


# horizon setting
PAST = 20
HORIZON = 3
HOURLY = False
# input data setting
INPUT_SNMLT = False
INPUT_MONTH = True
INPUT_DISCHARGE = True
# rain observation type
RAIN_OBS = 'gauge'
# Split parameters:
TRAIN_END = datetime.datetime(2017,9,1)
VAL_END = datetime.datetime(2018,9,1)
# dimensional reduction
CATCHMENT_AGGREGATION = False
PCA = True

def data_configuration(**kwargs):
    settings = dict(
        past=PAST,
        horizon=HORIZON,
        hourly=HOURLY,
        input_snmlt = INPUT_SNMLT,
        input_discharge = INPUT_DISCHARGE,
        input_month=INPUT_MONTH,
        train_end=TRAIN_END,
        val_end=VAL_END,
        catchment_aggregation=CATCHMENT_AGGREGATION,
        rain_obs = RAIN_OBS,
        pca_exec=PCA,
    )
    settings.update(kwargs)
    settings["unit"] = "hourly" if settings["hourly"] else "daily"

    if settings["hourly"]:
        settings["past"] = settings["past"] * 24
        settings["horizon"] = settings["horizon"] * 24
    return settings



