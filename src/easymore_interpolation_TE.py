import os, pickle, multiprocessing, subprocess
from os.path import join as pj
from os import listdir as ld
from tqdm import tqdm
import geopandas as gpd
from easymore.easymore import *

NCL_CONVERT_PATH = "/home/tristan/anaconda3/envs/ncl/bin:"

def __weighted_average(self,
                       nc_name,
                       target_time,
                       varibale_name,
                       mapping_df):
    """
    """
    ds = xr.open_dataset(nc_name)
    if self.var_time != 'time':
        ds = ds.rename({self.var_time:'time'})
    weighted_value = np.zeros([self.length_of_time,self.number_of_target_elements])
    m = 0 # counter
    for date in target_time: # loop over time
        #ds_temp = ds.sel(time=np.timedelta64(date*3600000000000), method="nearest")
        ds_temp = ds.sel(time=date.strftime("%Y-%m-%d %H:%M:%S"),method="nearest")
        data = np.array(ds_temp[varibale_name])
        data = np.squeeze(data)
        if self.case ==1 or self.case ==2:
            values = data [self.rows,self.cols]
        if self.case ==3:
            values = data [self.rows]
        values = np.array(values)
        mapping_df['values'] = values
        mapping_df['values_w'] = mapping_df['weight']*mapping_df['values']
        df_temp = mapping_df.groupby(['order_t'], as_index=False).agg({'values_w': 'sum'})
        df_temp = df_temp.sort_values(by=['order_t'])
        weighted_value [m,:] = np.array(df_temp['values_w'])
        m += 1
    return weighted_value


def __target_nc_creation(self):
    """
        @ author:                  Shervan Gharari
        @ Github:                  https://github.com/ShervanGharari/EASYMORE
        @ author's email id:       sh.gharari@gmail.com
        @ license:                 GNU-GPLv3
        This funciton read different grids and sum them up based on the
        weight provided to aggregate them over a larger area
    """
    remap = pd.read_csv(self.remap_csv)
    target_ID_lat_lon = pd.DataFrame()
    target_ID_lat_lon ['ID_t']  = remap ['ID_t']
    target_ID_lat_lon ['lat_t'] = remap ['lat_t']
    target_ID_lat_lon ['lon_t'] = remap ['lon_t']
    target_ID_lat_lon ['order_t'] = remap ['order_t']
    target_ID_lat_lon = target_ID_lat_lon.drop_duplicates()
    target_ID_lat_lon = target_ID_lat_lon.sort_values(by=['order_t'])
    target_ID_lat_lon = target_ID_lat_lon.reset_index(drop=True)
    hruID_var = np.array(target_ID_lat_lon['ID_t'])
    hruID_lat = np.array(target_ID_lat_lon['lat_t'])
    hruID_lon = np.array(target_ID_lat_lon['lon_t'])
    self.rows = np.array(remap['rows']).astype(int)
    self.cols = np.array(remap['cols']).astype(int)
    self.number_of_target_elements = len(hruID_var)
    nc_names = glob.glob(self.source_nc)
    nc_names = sorted(nc_names)
    #print(nc_names)
    for nc_name in nc_names:
        # get the time unit and time var from source
        ncids = nc4.Dataset(nc_name)
        # Check data license, calendar and time units
        nc_att_list = ncids.ncattrs()
        nc_att_list = [each_att for each_att in nc_att_list]
        nc_att_list_lower = [each_att.lower() for each_att in nc_att_list]
        if 'units' in ncids.variables[self.var_time].ncattrs():
            time_unit = ncids.variables[self.var_time].units
            #print(f"Time unit: {time_unit}")
        else:
            sys.exit('units is not provided for the time varibale for source NetCDF of'+ nc_name)
        time_var = ncids[self.var_time][:]
        self.length_of_time = len(time_var)
        target_date_times = time_var.data
        #print(type(target_date_times))
        #print(target_date_times)
        target_date_times = nc4.num2date(time_var,  units=time_unit) # , calendar = time_cal)
        #print(target_date_times)
        #print("ok")
        target_name = self.output_dir + self.case_name 
        #+ '_remapped_' + "givendate.nc"#target_date_times[0].strftime("%Y-%m-%d-%H-%M-%S")+'.nc'
        if os.path.exists(target_name):
            if self.overwrite_existing_remap: 
                os.remove(target_name)
            else:
                continue # skip to next file
        for var in ncids.variables.values():
            if var.name == self.var_time:
                time_dtype =  str(var.dtype)
        time_dtype_code = 'f8' # initialize the time as float
        if 'float' in time_dtype.lower():
            time_dtype_code = 'f8'
        elif 'int' in time_dtype.lower():
            time_dtype_code = 'i4'

        with nc4.Dataset(target_name, "w", format="NETCDF4") as ncid: # creating the NetCDF file
            # define the dimensions
            dimid_N = ncid.createDimension(self.remapped_dim_id, len(hruID_var))  # limited dimensiton equal the number of hruID
            dimid_T = ncid.createDimension('time', None)   # unlimited dimensiton
            # Variable time
            time_varid = ncid.createVariable('time', time_dtype_code, ('time', ))
            # Attributes
            time_varid.long_name = self.var_time
            time_varid.units = time_unit  # e.g. 'days since 2000-01-01 00:00' should change accordingly
            #time_varid.calendar = time_cal
            time_varid.standard_name = self.var_time
            time_varid.axis = 'T'
            time_varid[:] = time_var
            # Variables lat, lon, subbasin_ID
            lat_varid = ncid.createVariable(self.remapped_var_lat, 'f8', (self.remapped_dim_id, ))
            lon_varid = ncid.createVariable(self.remapped_var_lon, 'f8', (self.remapped_dim_id, ))
            hruId_varid = ncid.createVariable(self.remapped_var_id, 'f8', (self.remapped_dim_id, ))
            # Attributes
            lat_varid.long_name = self.remapped_var_lat
            lon_varid.long_name = self.remapped_var_lon
            hruId_varid.long_name = 'shape ID'
            lat_varid.units = 'degrees_north'
            lon_varid.units = 'degrees_east'
            hruId_varid.units = '1'
            lat_varid.standard_name = self.remapped_var_lat
            lon_varid.standard_name = self.remapped_var_lon
            lat_varid[:] = hruID_lat
            lon_varid[:] = hruID_lon
            hruId_varid[:] = hruID_var
            # general attributes for NetCDF file
            ncid.Conventions = 'CF-1.6'
            ncid.Author = 'The data were written by ' + self.author_name
            ncid.License = self.license
            ncid.History = 'Created ' + time.ctime(time.time())
            ncid.Source = 'Case: ' + self.case_name + '; remapped by script from library of Shervan Gharari (https://github.com/ShervanGharari/EASYMORE).'
            # write varibales
            for i in np.arange(len(self.var_names)):
                var_value  = __weighted_average(self,  nc_name,
                                                      target_date_times,
                                                      self.var_names[i],
                                                      remap)
                # Variables writing
                varid = ncid.createVariable(self.var_names_remapped[i], 
                                            self.format_list[i], 
                                            ('time',self.remapped_dim_id ), 
                                            fill_value = self.fill_value_list[i])
                varid [:] = var_value
                # Pass attributes
                if 'long_name' in ncids.variables[self.var_names[i]].ncattrs():
                    varid.long_name = ncids.variables[self.var_names[i]].long_name
                if 'units' in ncids.variables[self.var_names[i]].ncattrs():
                    varid.units = ncids.variables[self.var_names[i]].units
        if self.save_csv:
            ds = xr.open_dataset(target_name)
            for i in np.arange(len(self.var_names_remapped)):
                new_list = list(self.var_names_remapped) # new lists
                del new_list[i] # remove one value
                #ds_temp = ds.drop(new_list) # drop all the other varibales excpet target varibale, lat, lon and time
                ds_temp = ds.drop_vars(new_list) # drop all the other varibales excpet target varibale, lat, lon and time
                if 'units' in ds[self.var_names_remapped[i]].attrs.keys():
                    dictionary = {self.var_names_remapped[i]:self.var_names_remapped[i]+' ['+ds[self.var_names_remapped[i]].attrs['units']+']'}
                    ds_temp = ds_temp.rename_vars(dictionary)
                target_name_csv = self.output_dir + self.case_name.replace(".nc", '.csv')
                if os.path.exists(target_name_csv): # remove file if exists
                    os.remove(target_name_csv)
                ds_temp = ds_temp.set_coords([self.remapped_var_lat,self.remapped_var_lon])
                df = ds_temp.to_dataframe()
                df['ID'] = df.index.get_level_values(level=0)
                df['time'] = df.index.get_level_values(level=1)
                df = df.set_index(['ID','time',self.remapped_var_lat,self.remapped_var_lon])
                df = df.unstack(level=-3)
                df = df.transpose()
                if 'units' in ds[self.var_names_remapped[i]].attrs.keys():
                    df = df.replace(self.var_names_remapped[i], self.var_names_remapped[i]+' '+ds[self.var_names_remapped[i]].attrs['units'])
                df.to_csv(target_name_csv)
            ds.close()
            os.remove(target_name)

        
def check_easymore_remap(  self,
                         remap_df):
    """
    @ author:                  Shervan Gharari
    @ Github:                  https://github.com/ShervanGharari/EASYMORE
    @ author's email id:       sh.gharari@gmail.com
    @ license:                 GNU-GPLv3
    this function check the remapping dataframe
    Parameters:
    ----------
    remap_df: dataframe, including remapping information including the following colomns:
                ID_target
                lon_target
                lat_target
                ID_source
                lat_source
                lon_source
                rows
                cols
                order
    """
    if not (len(np.unique(np.array(remap_df['easymore_case'])))==1):
        sys.exit('the EASYMORE_case is not unique in the remapping file')
    if not (np.unique(np.array(remap_df['easymore_case'])) == 1 or\
    np.unique(np.array(remap_df['easymore_case'])) == 2 or\
    np.unique(np.array(remap_df['easymore_case'])) == 3):
        sys.exit('EASYMORE case should be one of 1, 2 or 3; please refer to the documentation')
    self.case = np.unique(np.array(remap_df['easymore_case']))
    if not set(['ID_t','lat_t','lon_t','order_t','ID_s','lat_s','lon_s','weight']) <= set(remap_df.columns):
        sys.exit('provided remapping file does not have one of the needed fields: \n'+\
            'ID_t, lat_t, lon_t, order_t, ID_2, lat_s, lon_s, weight')

def check_source_nc(self):
    """
    @ author:                  Shervan Gharari
    @ Github:                  https://github.com/ShervanGharari/EASYMORE
    @ author's email id:       sh.gharari@gmail.com
    @ license:                 GNU-GPLv3
    This function checks the consistency of the dimentions and varibales for source netcdf file(s)
    """
    flag_do_not_match = False
    nc_names = glob.glob (self.source_nc)
    if not nc_names:
        sys.exit('EASYMORE detects no netCDF file; check the path to the soure netCDF files')
    else:
        ncid      = nc4.Dataset(nc_names[0])
        var_dim   = list(ncid.variables[self.var_names[0]].dimensions)
        lat_dim   = list(ncid.variables[self.var_lat].dimensions)
        lon_dim   = list(ncid.variables[self.var_lon].dimensions)
        lat_value = np.array(ncid.variables[self.var_lat])
        lon_value = np.array(ncid.variables[self.var_lon])
        # dimension check based on the first netcdf file
        if not (set(lat_dim) <= set(var_dim)):
            flag_do_not_match = True
        if not (set(lon_dim) <= set(var_dim)):
            flag_do_not_match = True
        if (len(lat_dim) == 2) and (len(lon_dim) == 2) and (len(var_dim) == 3): # case 2
            if not (set(lat_dim) == set(lon_dim)):
                flag_do_not_match = True
        if (len(lat_dim) == 1) and (len(lon_dim) == 1) and (len(var_dim) == 2): # case 3
            if not (set(lat_dim) == set(lon_dim)):
                flag_do_not_match = True
        # dimension check and consistancy for variable latitude
        for nc_name in nc_names:
            ncid = nc4.Dataset(nc_name)
            temp = list(ncid.variables[self.var_lat].dimensions)
            # fist check the length of the temp and lat_dim
            if len(temp) != len(lat_dim):
                flag_do_not_match = True
            else:
                for i in np.arange(len(temp)):
                    if temp[i] != lat_dim[i]:
                        flag_do_not_match = True
            temp = np.array(ncid.variables[self.var_lat])
            if np.sum(abs(lat_value-temp))>self.tolerance:
                flag_do_not_match = True
        # dimension check and consistancy for variable longitude
        for nc_name in nc_names:
            ncid = nc4.Dataset(nc_name)
            temp = list(ncid.variables[self.var_lon].dimensions)
            # fist check the length of the temp and lon_dim
            if len(temp) != len(lon_dim):
                flag_do_not_match = True
            else:
                for i in np.arange(len(temp)):
                    if temp[i] != lon_dim[i]:
                        flag_do_not_match = True
            temp = np.array(ncid.variables[self.var_lon])
            if np.sum(abs(lon_value-temp))>self.tolerance:
                flag_do_not_match = True
        # dimension check consistancy for variables to be remapped
        for var_name in self.var_names:
            # get the varibale information of lat, lon and dimensions of the varibale.
            for nc_name in nc_names:
                ncid = nc4.Dataset(nc_name)
                temp = list(ncid.variables[var_name].dimensions)
                # fist check the length of the temp and var_dim
                if len(temp) != len(var_dim):
                    flag_do_not_match = True
                else:
                    for i in np.arange(len(temp)):
                        if temp[i] != var_dim[i]:
                            flag_do_not_match = True
        # check varibale time and dimension time are the same name so time is coordinate
        for nc_name in nc_names:
            ncid = nc4.Dataset(nc_name)
            temp = ncid.variables[self.var_time].dimensions
            if len(temp) != 1:
                sys.exit('EASYMORE expects 1D time varibale, it seems time varibales has more than 1 dimension')
            if str(temp[0]) != self.var_time:
                sys.exit('EASYMORE expects time varibale and dimension to be different, they should be the same\
                for xarray to consider time dimension as coordinates')
    if flag_do_not_match:
        sys.exit('EASYMORE detects that all the provided netCDF files and varibale \
has different dimensions for the varibales or latitude and longitude')

def remap_csv(self):
    self.check_easymore_input()
    if not os.path.isfile(self.remap_csv):
        target_shp_gpd = gpd.read_file(self.target_shp)
        target_shp_gpd = self.check_target_shp(target_shp_gpd)
        target_shp_gpd.to_file(self.temp_dir+self.case_name+'_target_shapefile.shp') # save
        #check_source_nc(self)
        NetCDF_SHP_lat_lon(self)
        if (self.case == 1 or self.case == 2)  and (self.source_shp == ''):
            if self.case == 1:
                if hasattr(self, 'lat_expanded') and hasattr(self, 'lon_expanded'):
                    self.lat_lon_SHP(self.lat_expanded, self.lon_expanded,\
                        self.temp_dir+self.case_name+'_source_shapefile.shp')
                else:
                    self.lat_lon_SHP(self.lat, self.lon,\
                        self.temp_dir+self.case_name+'_source_shapefile.shp')
            else:
                self.lat_lon_SHP(self.lat, self.lon,\
                    self.temp_dir+self.case_name+'_source_shapefile.shp')
        if (self.case == 1 or self.case == 2)  and (self.source_shp != ''):
            source_shp_gpd = gpd.read_file(self.source_shp)
            source_shp_gpd = self.add_lat_lon_source_SHP(source_shp_gpd, self.source_shp_lat,\
                self.source_shp_lon, self.source_shp_ID)
            source_shp_gpd.to_file(self.temp_dir+self.case_name+'_source_shapefile.shp')
        if (self.case == 3) and (self.source_shp != ''):
            self.check_source_nc_shp() # check the lat lon in soure shapefile and nc file
            source_shp_gpd = gpd.read_file(self.source_shp)
            source_shp_gpd = self.add_lat_lon_source_SHP(source_shp_gpd, self.source_shp_lat,\
                self.source_shp_lon, self.source_shp_ID)
            source_shp_gpd.to_file(self.temp_dir+self.case_name+'_source_shapefile.shp')
            
        source_shp_gpd = gpd.read_file(self.temp_dir+self.case_name+'_source_shapefile.shp')
        source_shp_gpd = source_shp_gpd.set_crs("EPSG:4326")
        expanded_source = self.expand_source_SHP(source_shp_gpd, self.temp_dir, self.case_name)
        expanded_source.to_file(self.temp_dir+self.case_name+'_source_shapefile_expanded.shp')
        shp_1 = gpd.read_file(self.temp_dir+self.case_name+'_target_shapefile.shp')
        shp_2 = gpd.read_file(self.temp_dir+self.case_name+'_source_shapefile_expanded.shp')
        min_lon, min_lat, max_lon, max_lat = shp_1.total_bounds

        warnings.simplefilter('ignore')
        shp_2 ['lat_temp'] = shp_2.centroid.y
        shp_2 ['lon_temp'] = shp_2.centroid.x
        warnings.simplefilter('default') # back to normal

        if (-180<min_lon) and max_lon<180:
            shp_2 = shp_2 [shp_2['lon_temp'] <=  180]
            shp_2 = shp_2 [-180 <= shp_2['lon_temp']]
        if (0<min_lon) and max_lon<360:
            shp_2 = shp_2 [shp_2['lon_temp'] <=  360]
            shp_2 = shp_2 [0    <= shp_2['lon_temp']]
        shp_2.drop(columns=['lat_temp', 'lon_temp'])
        if (str(shp_1.crs).lower() == str(shp_2.crs).lower()) and ('epsg:4326' in str(shp_1.crs).lower()):
            shp_1 = shp_1.to_crs ("EPSG:6933") # project to equal area
            shp_1.to_file(self.temp_dir+self.case_name+'test.shp')
            shp_1 = gpd.read_file(self.temp_dir+self.case_name+'test.shp')
            shp_2 = shp_2.to_crs ("EPSG:6933") # project to equal area
            shp_2.to_file(self.temp_dir+self.case_name+'test.shp')
            shp_2 = gpd.read_file(self.temp_dir+self.case_name+'test.shp')
            removeThese = glob.glob(self.temp_dir+self.case_name+'test.*')
            for file in removeThese:
                os.remove(file)
        shp_int = self.intersection_shp(shp_1, shp_2)
        shp_int = shp_int.sort_values(by=['S_1_ID_t']) # sort based on ID_t
        shp_int = shp_int.to_crs ("EPSG:4326") # project back to WGS84
        shp_int.to_file(self.temp_dir+self.case_name+'_intersected_shapefile.shp') # save the intersected files
        shp_int = shp_int.drop(columns=['geometry']) # remove the geometry
        dict_rename = {'S_1_ID_t' : 'ID_t',
                       'S_1_lat_t': 'lat_t',
                       'S_1_lon_t': 'lon_t',
                       'S_1_order': 'order_t',
                       'S_2_ID_s' : 'ID_s',
                       'S_2_lat_s': 'lat_s',
                       'S_2_lon_s': 'lon_s',
                       'AP1N'     : 'weight'}
        shp_int = shp_int.rename(columns=dict_rename) # rename fields for remapping file
        shp_int = pd.DataFrame(shp_int) # move to data set and save as a csv
        shp_int.to_csv(self.temp_dir+self.case_name+'_intersected_shapefile.csv') # save the intersected files
        int_df = pd.read_csv (self.temp_dir+self.case_name+'_intersected_shapefile.csv')
        lat_source = self.lat
        lon_source = self.lon
        int_df = self.create_remap(int_df, lat_source, lon_source)
        int_df.to_csv(self.remap_csv)

def NetCDF_SHP_lat_lon(self):
    """
    @ author:                  Shervan Gharari
    @ Github:                  https://github.com/ShervanGharari/EASYMORE
    @ author's email id:       sh.gharari@gmail.com
    @ license:                 GNU-GPLv3
    This function checks dimension of the source shapefile and checks the case of regular, rotated, and irregular
    also created the 2D array of lat and lon for creating the shapefile
    """
    import geopandas as gpd
    from   shapely.geometry import Polygon
    import shapefile # pyshed library
    import shapely
    #
    nc_names = glob.glob (self.source_nc)
    var_name = self.var_names[0]
    # open the nc file to read
    ncid = nc4.Dataset(nc_names[0])
    # deciding which case
    # case #1 regular latitude/longitude
    if (len(ncid.variables[self.var_lon].dimensions)==1) and\
    (len(ncid.variables[self.var_lon].dimensions)==1) and\
    (len(ncid.variables[self.var_names[0]].dimensions)==2):
        #print('EASYMORE detects case 1 - regular lat/lon')
        self.case = 1
        # get the list of dimensions for the ncid sample varibale
        list_dim_name = list(ncid.variables[self.var_names[0]].dimensions)
        # get the location of lat dimensions
        location_of_lat = list_dim_name.index(list(ncid.variables[self.var_lat].dimensions)[0])
        locaiton_of_lon = list_dim_name.index(list(ncid.variables[self.var_lon].dimensions)[0])
        # det the dimensions of lat and lon
        len_of_lat = len(ncid.variables[self.var_lat][:])
        len_of_lon = len(ncid.variables[self.var_lon][:])
        if locaiton_of_lon > location_of_lat:
            lat = np.zeros([len_of_lat, len_of_lon])
            lon = np.zeros([len_of_lat, len_of_lon])
            for i in np.arange(len(ncid.variables[self.var_lon][:])):
                lat [:,i] = ncid.variables[self.var_lat][:]
            for i in np.arange(len(ncid.variables[self.var_lat][:])):
                lon [i,:] = ncid.variables[self.var_lon][:]
        else:
            lat = np.zeros([len_of_lon, len_of_lat])
            lon = np.zeros([len_of_lon, len_of_lat])
            for i in np.arange(len(ncid.variables[self.var_lon][:])):
                lat [i,:] = ncid.variables[self.var_lat][:]
            for i in np.arange(len(ncid.variables[self.var_lat][:])):
                lon [:,i] = ncid.variables[self.var_lon][:]
        # check if lat and lon are spaced equally
        lat_temp = np.array(ncid.variables[self.var_lat][:])
        lat_temp_diff = np.diff(lat_temp)
        lat_temp_diff_2 = np.diff(lat_temp_diff)
        max_lat_temp_diff_2 = max(abs(lat_temp_diff_2))
        #print('max difference of lat values in source nc files are : ', max_lat_temp_diff_2)
        lon_temp = np.array(ncid.variables[self.var_lon][:])
        lon_temp_diff = np.diff(lon_temp)
        lon_temp_diff_2 = np.diff(lon_temp_diff)
        max_lon_temp_diff_2 = max(abs(lon_temp_diff_2))
        #print('max difference of lon values in source nc files are : ', max_lon_temp_diff_2)
        # save lat, lon into the object
        lat      = np.array(lat).astype(float)
        lon      = np.array(lon).astype(float)
        self.lat = lat
        self.lon = lon
        # expanding just for the the creation of shapefile with first last rows and columns
        if (max_lat_temp_diff_2<self.tolerance) and (max_lon_temp_diff_2<self.tolerance): # then lat lon are spaced equal
            # create expanded lat
            lat_expanded = np.zeros(np.array(lat.shape)+2)
            lat_expanded [1:-1,1:-1] = lat
            lat_expanded [:, 0]  = lat_expanded [:, 1] + (lat_expanded [:, 1] - lat_expanded [:, 2]) # populate left column
            lat_expanded [:,-1]  = lat_expanded [:,-2] + (lat_expanded [:,-2] - lat_expanded [:,-3]) # populate right column
            lat_expanded [0, :]  = lat_expanded [1, :] + (lat_expanded [1, :] - lat_expanded [2, :]) # populate top row
            lat_expanded [-1,:]  = lat_expanded [-2,:] + (lat_expanded [-2,:] - lat_expanded [-3,:]) # populate bottom row
            # create expanded lat
            lon_expanded = np.zeros(np.array(lon.shape)+2)
            lon_expanded [1:-1,1:-1] = lon
            lon_expanded [:, 0]  = lon_expanded [:, 1] + (lon_expanded [:, 1] - lon_expanded [:, 2]) # populate left column
            lon_expanded [:,-1]  = lon_expanded [:,-2] + (lon_expanded [:,-2] - lon_expanded [:,-3]) # populate right column
            lon_expanded [0, :]  = lon_expanded [1, :] + (lon_expanded [1, :] - lon_expanded [2, :]) # populate top row
            lon_expanded [-1,:]  = lon_expanded [-2,:] + (lon_expanded [-2,:] - lon_expanded [-3,:]) # populate bottom row
            # pass to the lat, lon extended
            self.lat_expanded = lat_expanded
            self.lon_expanded = lon_expanded
    # case #2 rotated lat/lon
    elif (len(ncid.variables[self.var_lat].dimensions)==2) and (len(ncid.variables[self.var_lon].dimensions)==2):
        #print('EASYMORE detects case 2 - rotated lat/lon')
        self.case = 2
        lat = ncid.variables[self.var_lat][:,:]
        lon = ncid.variables[self.var_lon][:,:]
        # creating/saving the shapefile
        lat = np.array(lat).astype(float)
        lon = np.array(lon).astype(float)
        self.lat = lat
        self.lon = lon
    # case #3 1-D lat/lon and 2 data for irregulat shapes
    elif (len(ncid.variables[self.var_lat].dimensions)==1) and (len(ncid.variables[self.var_lon].dimensions)==1) and\
       (len(ncid.variables[self.var_names[0]].dimensions)==2):
        #print('EASYMORE detects case 3 - irregular lat/lon; shapefile should be provided')
        self.case = 3
        lat = ncid.variables[self.var_lat][:]
        lon = ncid.variables[self.var_lon][:]
        #print(lat, lon)
        if self.var_ID  == '':
            #print('EASYMORE detects that no varibale for ID of the source netCDF file; an arbitatiry ID will be provided')
            ID =  np.arange(len(lat))+1 # pass arbitarary values
        else:
            ID = ncid.variables[self.var_ID][:]
        # creating/saving the shapefile
        lat = np.array(lat).astype(float)
        lon = np.array(lon).astype(float)
        self.lat = lat
        self.lon = lon
        self.ID  = ID
    print("Done it")
        
def map_nc(input_path, 
           varin=["TMP_P0_L103_GLL0"],
           varout=["temperature"],
           remap_csv='../data/nc_remapping.csv',
           target_shp="/media/tristan/Elements/Hydro/MERIT/MERIT_BASIN/dam_basins.shp",
           outdir = '/media/tristan/Elements/Hydro/weather/MSM_basin_temp/',
           var_time = 'forecast_time0',
           var_lon='lon_0',
           var_lat='lat_0'
          ):
    
    esmr = easymore()
    esmr.case_name                = os.path.basename(input_path)              
    esmr.target_shp               = target_shp
    esmr.source_nc                = input_path
    esmr.var_names                = varin# 'TMP_P0_L103_GLL0',  ["Temperature", "Total precipitation"]
    esmr.var_names_remapped       = varout #'temperature',
    esmr.var_lon                  = var_lon#'lon_0'
    esmr.var_lat                  = var_lat#'lat_0'
    esmr.var_time                 = var_time#'forecast_time0'
    esmr.output_dir               = outdir
    esmr.format_list              = ['f4']
    esmr.fill_value_list          = ['-9999.00']
    esmr.save_csv                 = True
    esmr.temp_dir = "/tmp/esmr/"
    esmr.remap_csv = remap_csv
    
    os.makedirs(esmr.temp_dir, exist_ok=True)
    
    try:
        nc_remapper(esmr)
        out_path = pj(esmr.output_dir, esmr.case_name)
        if os.path.isfile(out_path):
            return out_path
        else:
            return 0
    except Exception:
        return 0
    
def nc_remapper(self):
    """
    """
    remap_csv(self)
    int_df  = pd.read_csv(self.remap_csv)
    check_easymore_remap(self, int_df)
    #check_source_nc(self)
    __target_nc_creation(self)

def grib_to_nc(infile, tmpdir="/tmp/ncl_convert_TE"):
    env = NCL_CONVERT_PATH + os.environ["PATH"]
    os.makedirs(tmpdir, exist_ok=True)
    return_code = subprocess.call(["ncl_convert2nc", infile, "-o", tmpdir], 
                                  env=dict(os.environ, PATH=env))
    if return_code == 0:
        path = pj(tmpdir, os.path.basename(infile).replace(".grib2", ".nc"))
        if os.path.isfile(path):
            return path
        else:
            return 0
    else:
        return -1
    
variables = {
    'precipitation':"APCP_P8_L1_GLL0_acc1h",
    "temperature":"TMP_P0_L103_GLL0"
}
    
def process_grib(in_path, 
                 varin=["TMP_P0_L103_GLL0"], 
                 varout=["temperature"],
                 outdir = '/media/tristan/Elements/Hydro/weather/MSM_basin_temp/',
                 remap_csv= '../data/nc_remapping.csv',
                 target_shp="/media/tristan/Elements/Hydro/MERIT/MERIT_BASIN/dam_basins.shp"):  
    f = os.path.basename(in_path)
    tmp_path = grib_to_nc(in_path)
    
    if isinstance(tmp_path, str):
        out_path = map_nc(tmp_path, varin, varout,
                          remap_csv=remap_csv,
                          target_shp=target_shp,
                          outdir=outdir)
        os.remove(tmp_path)
        return_code = out_path
    else:
        return_code=tmp_path
    return return_code

#def _process_grib(x): 
#    return process_grib(x, varin, varout, outdir=outdir)

def parallel_process_grib(indir, outdir, varin, varout, nproc=20):
    infiles = [pj(indir, x) for x in os.listdir(indir)]
    todo = list(set([x.split(".")[0] for x in os.listdir(indir)]) - set([x.split(".")[0] for x in os.listdir(outdir)]))
    remaining = list(set([os.path.basename(x) for x in infiles]) \
                   - set([os.path.basename(x).replace(".nc", ".grib2") for x in os.listdir(outdir)]))
    print(f"Extracting {len(remaining)} remaining files")
    pool = multiprocessing.Pool(nproc)
    start = time.time()
    
    return_codes = pool.starmap(process_grib, [(pj(indir, x), varin, varout, outdir) for x in remaining])
    print(f"Extraction done in  {int(time.time() - start)} seconds")
    return return_codes