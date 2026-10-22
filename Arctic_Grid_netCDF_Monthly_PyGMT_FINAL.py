import numpy as np
import numpy.ma as ma
import pygmt
import matplotlib.pyplot as plt
import pandas as pd
import glob, os, warnings, time, csv, math
from datetime import datetime, date, timezone
from global_land_mask import globe
import xarray as xr
from scipy.interpolate import griddata, RegularGridInterpolator, RBFInterpolator
from scipy.stats import binned_statistic_2d
from pyproj import Proj, transform
from netCDF4 import Dataset
from scipy.spatial import distance

warnings.filterwarnings("ignore")
np.printoptions(suppress=True)

def reproject_to_PSG(x, y):
    inProj,outProj = Proj(init='epsg:4326'), Proj(init='epsg:6931')
    x2,y2 = transform(inProj,outProj,x,y)
    return x2,y2

def reproject_to_WGS84(x, y):
    inProj,outProj = Proj(init='epsg:6931'), Proj(init='epsg:4326')
    x2,y2 = transform(inProj,outProj,x,y)
    return x2,y2

np.set_printoptions(suppress=True)

start_time = time.time()

pathname = '//' ##Path to netcdf files
pathname2 = '//' ##Path to AMSR2 monthly mean sea ice concentration files
pathname3 = '//' ##Path to save files to

years = ['2025']#,'2020','2021','2022','2023','2024','2025']
months = ['09','10']#,'11']#,'03','04','05','06','07','08','09','10','11','12']

for year in years:
    print (year)
    for month in months:
        print (month)
        nc_files_list = []
        if ((int(year) == 2018) & (int(month) >= 10)):
            nc_files_list.append(glob.glob(pathname + str(year) + '/' + str(month) + '/*.nc'))

        elif ((int(year) >= 2019) & (int(year) <= 2025)):
            nc_files_list.append(glob.glob(pathname + str(year) + '/' + str(month) + '/*.nc'))

        elif ((int(year) == 2026) & (int(month) <= 10)):
            nc_files_list.append(glob.glob(pathname + str(year) + '/' + str(month) + '/*.nc'))

        else:
            continue

        nc_files = sorted(np.hstack(nc_files_list))

        surfrough_lon = []
        surfrough_lat = []
        surfrough = []
        surfrough_lt60_lon = []
        surfrough_lt60_lat = []
        surfrough_lt60 = []
        surfrough_lt20_lon = []
        surfrough_lt20_lat = []
        surfrough_lt20 = []
        max_sh_lon = []
        max_sh_lat = []
        max_sh = []
        sh_lon = []
        sh_lat = []
        sh = []
        spacing_lon = []
        spacing_lat = []
        spacing = []
        RI_lon = []
        RI_lat = []
        RI = []
        width_lon = []
        width_lat = []
        width = []
        length_lon = []
        length_lat = []
        length = []
        max_sh_20_lon = []
        max_sh_20_lat = []
        max_sh_20 = []
        sh_20_lon = []
        sh_20_lat = []
        sh_20 = []
        spacing_20_lon = []
        spacing_20_lat = []
        spacing_20 = []
        RI_20_lon = []
        RI_20_lat = []
        RI_20 = []
        width_20_lon = []
        width_20_lat = []
        width_20 = []
        length_20_lon = []
        length_20_lat = []
        length_20 = []

        for i in nc_files:
            #print (i)
            nc_fid = Dataset(i, 'r')

            nc_fid.set_auto_mask(False)

            if (nc_fid.groups['surf_rough']['1km'].variables != {}):
                surfrough_lon.append(nc_fid.groups['surf_rough']['1km']['longitude'][:])
                surfrough_lat.append(nc_fid.groups['surf_rough']['1km']['latitude'][:])
                surfrough.append(nc_fid.groups['surf_rough']['1km']['surface_roughness_1km'][:])

            if (nc_fid.groups['surf_rough']['lt60'].variables != {}):
                surfrough_lt60_lon.append(nc_fid.groups['surf_rough']['lt60']['longitude'][:])
                surfrough_lt60_lat.append(nc_fid.groups['surf_rough']['lt60']['latitude'][:])
                surfrough_lt60.append(nc_fid.groups['surf_rough']['lt60']['surface_roughness_1km_lt60'][:])

            if (nc_fid.groups['surf_rough']['lt20'].variables != {}):
                surfrough_lt20_lon.append(nc_fid.groups['surf_rough']['lt20']['longitude'][:])
                surfrough_lt20_lat.append(nc_fid.groups['surf_rough']['lt20']['latitude'][:])
                surfrough_lt20.append(nc_fid.groups['surf_rough']['lt20']['surface_roughness_1km_lt20'][:])

            if (nc_fid.groups['max_sail_height'].variables != {}):
                max_sh_lon.append(nc_fid.groups['max_sail_height']['longitude'][:])
                max_sh_lat.append(nc_fid.groups['max_sail_height']['latitude'][:])
                max_sh.append(nc_fid.groups['max_sail_height']['max_sail_height'][:])

            if (nc_fid.groups['sail_height'].variables != {}):
                sh_lon.append(nc_fid.groups['sail_height']['longitude'][:])
                sh_lat.append(nc_fid.groups['sail_height']['latitude'][:])
                sh.append(nc_fid.groups['sail_height']['sail_height'][:])

            if (nc_fid.groups['ridge_sail_spacing'].variables != {}):
                spacing_lon.append(nc_fid.groups['ridge_sail_spacing']['longitude'][:])
                spacing_lat.append(nc_fid.groups['ridge_sail_spacing']['latitude'][:])
                spacing.append(nc_fid.groups['ridge_sail_spacing']['ridge_sail_spacing'][:])

            if (nc_fid.groups['ridging_intensity'].variables != {}):
                RI_lon.append(nc_fid.groups['ridging_intensity']['longitude'][:])
                RI_lat.append(nc_fid.groups['ridging_intensity']['latitude'][:])
                RI.append(nc_fid.groups['ridging_intensity']['ridging_intensity'][:])

            if (nc_fid.groups['ridge_width'].variables != {}):
                width_lon.append(nc_fid.groups['ridge_width']['longitude'][:])
                width_lat.append(nc_fid.groups['ridge_width']['latitude'][:])
                width.append(nc_fid.groups['ridge_width']['ridge_width'][:])

            if (nc_fid.groups['ridge_length'].variables != {}):
                length_lon.append(nc_fid.groups['ridge_length']['longitude'][:])
                length_lat.append(nc_fid.groups['ridge_length']['latitude'][:])
                length.append(nc_fid.groups['ridge_length']['ridge_length'][:])

            if (nc_fid.groups['20cm_cutoff']['max_sail_height'].variables != {}):
                max_sh_20_lon.append(nc_fid.groups['20cm_cutoff']['max_sail_height']['longitude'][:])
                max_sh_20_lat.append(nc_fid.groups['20cm_cutoff']['max_sail_height']['latitude'][:])
                max_sh_20.append(nc_fid.groups['20cm_cutoff']['max_sail_height']['max_sail_height'][:])

            if (nc_fid.groups['20cm_cutoff']['sail_height'].variables != {}):
                sh_20_lon.append(nc_fid.groups['20cm_cutoff']['sail_height']['longitude'][:])
                sh_20_lat.append(nc_fid.groups['20cm_cutoff']['sail_height']['latitude'][:])
                sh_20.append(nc_fid.groups['20cm_cutoff']['sail_height']['sail_height'][:])

            if (nc_fid.groups['20cm_cutoff']['ridge_sail_spacing'].variables != {}):
                spacing_20_lon.append(nc_fid.groups['20cm_cutoff']['ridge_sail_spacing']['longitude'][:])
                spacing_20_lat.append(nc_fid.groups['20cm_cutoff']['ridge_sail_spacing']['latitude'][:])
                spacing_20.append(nc_fid.groups['20cm_cutoff']['ridge_sail_spacing']['ridge_sail_spacing'][:])

            if (nc_fid.groups['20cm_cutoff']['ridging_intensity'].variables != {}):
                RI_20_lon.append(nc_fid.groups['20cm_cutoff']['ridging_intensity']['longitude'][:])
                RI_20_lat.append(nc_fid.groups['20cm_cutoff']['ridging_intensity']['latitude'][:])
                RI_20.append(nc_fid.groups['20cm_cutoff']['ridging_intensity']['ridging_intensity'][:])

            if (nc_fid.groups['20cm_cutoff']['ridge_width'].variables != {}):
                width_20_lon.append(nc_fid.groups['20cm_cutoff']['ridge_width']['longitude'][:])
                width_20_lat.append(nc_fid.groups['20cm_cutoff']['ridge_width']['latitude'][:])
                width_20.append(nc_fid.groups['20cm_cutoff']['ridge_width']['ridge_width'][:])

            if (nc_fid.groups['20cm_cutoff']['ridge_length'].variables != {}):
                length_20_lon.append(nc_fid.groups['20cm_cutoff']['ridge_length']['longitude'][:])
                length_20_lat.append(nc_fid.groups['20cm_cutoff']['ridge_length']['latitude'][:])
                length_20.append(nc_fid.groups['20cm_cutoff']['ridge_length']['ridge_length'][:])

            nc_fid.close()

        surfrough_lon_arr = np.hstack(surfrough_lon)
        surfrough_lat_arr = np.hstack(surfrough_lat)
        surfrough_arr = np.hstack(surfrough)

        surfrough_lt60_lon_arr = np.hstack(surfrough_lt60_lon)
        surfrough_lt60_lat_arr = np.hstack(surfrough_lt60_lat)
        surfrough_lt60_arr = np.hstack(surfrough_lt60)

        surfrough_lt20_lon_arr = np.hstack(surfrough_lt20_lon)
        surfrough_lt20_lat_arr = np.hstack(surfrough_lt20_lat)
        surfrough_lt20_arr = np.hstack(surfrough_lt20)

        max_sh_lon_arr = np.hstack(max_sh_lon)
        max_sh_lat_arr = np.hstack(max_sh_lat)
        max_sh_arr = np.hstack(max_sh)

        sh_lon_arr = np.hstack(sh_lon)
        sh_lat_arr = np.hstack(sh_lat)
        sh_arr = np.hstack(sh)

        spacing_lon_arr = np.hstack(spacing_lon)
        spacing_lat_arr = np.hstack(spacing_lat)
        spacing_arr = np.hstack(spacing)

        RI_lon_arr = np.hstack(RI_lon)
        RI_lat_arr = np.hstack(RI_lat)
        RI_arr = np.hstack(RI)

        width_lon_arr = np.hstack(width_lon)
        width_lat_arr = np.hstack(width_lat)
        width_arr = np.hstack(width)

        length_lon_arr = np.hstack(length_lon)
        length_lat_arr = np.hstack(length_lat)
        length_arr = np.hstack(length)

        max_sh_20_lon_arr = np.hstack(max_sh_20_lon)
        max_sh_20_lat_arr = np.hstack(max_sh_20_lat)
        max_sh_20_arr = np.hstack(max_sh_20)

        sh_20_lon_arr = np.hstack(sh_20_lon)
        sh_20_lat_arr = np.hstack(sh_20_lat)
        sh_20_arr = np.hstack(sh_20)

        spacing_20_lon_arr = np.hstack(spacing_20_lon)
        spacing_20_lat_arr = np.hstack(spacing_20_lat)
        spacing_20_arr = np.hstack(spacing_20)

        RI_20_lon_arr = np.hstack(RI_20_lon)
        RI_20_lat_arr = np.hstack(RI_20_lat)
        RI_20_arr = np.hstack(RI_20)

        width_20_lon_arr = np.hstack(width_20_lon)
        width_20_lat_arr = np.hstack(width_20_lat)
        width_20_arr = np.hstack(width_20)

        length_20_lon_arr = np.hstack(length_20_lon)
        length_20_lat_arr = np.hstack(length_20_lat)
        length_20_arr = np.hstack(length_20)

        #grid_res = 25000
        #grid_res = 12500
        #grid_res = 10000
        #grid_res = 9000
        #grid_res = 6250
        grid_res = 3125

        ncfile = Dataset(pathname3 + '.nc',mode='w',format='NETCDF4')

        ncfile.title = 'ICESat-2 UMD-RDA Monthly Gridded Arctic Sea Ice Deformation Data'
        ncfile.institution = 'Earth System Science Interdisciplinary Center, University of Maryland, College Park, MD, 20740-3823 USA'
        ncfile.source = 'ICESat-2 Global Geolocated Photon product (ATL03), Version 007, and the ICESat-2 University of Maryland-Ridge Detection Algorithm (UMD-RDA)'
        ncfile.references = 'Duncan, K., & Farrell, S. L. (2022). Determining variability in Arctic sea ice pressure ridge topography with ICESat-2.' \
        'Geophysical Research Letters, 49, e2022GL100272. https://doi.org/10.1029/2022GL100272'
        ncfile.Conventions = 'CF-1.8'
        ncfile.license = 'CC0-1.0'
        ncfile.product_version = 'Version 1'
        ncfile.file_name = 'IS2-UMDRDA-NH-EASE2-' + str(grid_res) + 'm-v01-' + year + month + '.nc'
        ncfile.summary = 'The ICESat-2 UMD-RDA Monthly Gridded Arctic Sea Ice Deformation dataset is a monthly product derived from the University of ' \
                'Maryland-Ridge Detection Algorithm (UMD-RDA), gridded at 25.0, 10.0, and 3.125 km. The UMD-RDA is a surface tracking ' \
                'algorithm specifically designed to measure sea ice surface topography at a nominal along-track sampling of 0.7 m, as ' \
                'described in Duncan and Farrell (2022). The ICESat-2 ATL03 Global Geolocated Photon product (Neumann et al., 2023) provides ' \
                'input data for UMD-RDA, and the algorithm is applied to each ATL03 granule over the Arctic Ocean domain (i.e., ATL03 granules ' \
                '"03", "04", and "05"). UMD-RDA computes surface elevation relative to the DTU21 mean sea surface (Andersen, 2022). Parameters ' \
                'describing sea ice deformation are derived using the UMD-RDA and include surface roughness, sail height, maximum sail height, ' \
                'sail spacing, sail width, ridge length, and ridging intensity.'
        ncfile.contact_email = 'kduncan@umd.edu'
        ncfile.spatial_resolution = str(float(grid_res/1000)) + ' km'
        ncfile.geospatial_lat_min = '60.0'
        ncfile.geospatial_lat_max = '90.0'
        ncfile.geospatial_lat_units = 'degrees_north'
        ncfile.geospatial_lon_min = '-180.0'
        ncfile.geospatial_lon_max = '180.0'
        ncfile.geospatial_lon_units = 'degrees_east'
        ncfile.reference_ellipsoid = 'WGS 84'
        ncfile.key_variables = 'surface_roughness, sail height, maximum sail height, sail spacing, sail width, ridging intensity, ridge length'
        ncfile.production_date = str(datetime.now())[:4] + str(datetime.now())[5:7] + str(datetime.now())[8:10] + 'T000000Z'
        ncfile.acknowledgement_statement = 'Users are free to use the information provided in this file in their research, provided they cite the ' \
                                           'following reference: Duncan, K., & Farrell, S. L. (2022). Determining variability in Arctic sea ice' \
                                           'pressure ridge topography with ICESat-2. Geophysical Research Letters, 49, e2022GL100272. https://doi.org/10.1029/2022GL100272'

        #grid_xy = 720 ## 25 km
        #grid_xy = 432 ## AWI modified 25 km
        #grid_xy = 1440 ## 12.5 km
        #grid_xy = 1080 ## AWI modified 10 km
        #grid_xy = 2000 ## 9 km
        #grid_xy = 1200 ## AWI modified 9 km
        #grid_xy = 2880 ## 6.25 km
        #grid_xy = 1728 ## AWI modified 6.25 km
        #grid_xy = 5760 ## 3.125 km
        grid_xy = 3456 ## AWI modified 3.125 km

        crs_nc = ncfile.createVariable('crs', np.int32, ())
        crs_nc.grid_mapping_name = 'lambert_azimuthal_equal_area'
        crs_nc.epsg = '6931'
        crs_nc.false_easting = '0.0'
        crs_nc.false_northing = '0.0'
        crs_nc.latitude_of_projection_origin = '90.0'
        crs_nc.longitude_of_projection_origin = '0.0'
        crs_nc.proj4_string = '+proj=laea +lon_0=0 +datum=WGS84 +ellps=WGS84 +lat_0=90.0'
        crs_nc.inverse_flattening = '298.257223563'
        crs_nc.semi_major_axis = '6378137.0'
        crs_nc.semi_minor_axis = "6356752.3"
        crs_nc.crs_wkt = 'PROJCS["WGS 84 / NSIDC EASE-Grid 2.0 North",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["latitude_of_center",90],PARAMETER["longitude_of_center",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AUTHORITY["EPSG","6931"]]'
        crs_nc.long_name = 'NSIDC NH EASE2 ' + str(float(grid_res/1000)) + 'km'


        lon_vals = np.arange(-5400000.0+(grid_res/2.0),5400000.0+(grid_res/2.0),float(grid_res))
        lat_vals = np.arange(-5400000.0+(grid_res/2.0),5400000.0+(grid_res/2.0),float(grid_res))

        lon_dim = ncfile.createDimension('x',grid_xy)
        lat_dim = ncfile.createDimension('y',grid_xy)

        x_lon,y_lat = np.meshgrid(lon_vals,lat_vals)

        WGS84_coords = reproject_to_WGS84(y_lat.ravel(),x_lon.ravel())

        longitude = ncfile.createVariable('longitude', np.float64, ('y','x'),compression='zlib',least_significant_digit=8)
        longitude.long_name = 'Longitude of grid cell center'
        longitude.units = 'degrees_east'

        latitude = ncfile.createVariable('latitude', np.float64, ('y','x'),compression='zlib',least_significant_digit=8)
        latitude.long_name = 'Latitude of grid cell center'
        latitude.units = 'degrees_north'

        longitude[:,:] = WGS84_coords[0].reshape(grid_xy,grid_xy)
        latitude[:,:] = WGS84_coords[1].reshape(grid_xy,grid_xy)

        x_proj_nc = ncfile.createVariable('x_proj', np.float32, ('x',),compression='zlib',least_significant_digit=5)
        x_proj_nc.long_name = 'X coordinate of projection'
        x_proj_nc.units = 'km'

        y_proj_nc = ncfile.createVariable('y_proj', np.float32, ('y',),compression='zlib',least_significant_digit=5)
        y_proj_nc.long_name = 'Y coordinate of projection'
        y_proj_nc.units = 'km'

        x_proj_nc[:] = lon_vals/1000.0
        y_proj_nc[:] = lat_vals/1000.0

        time_dim = ncfile.createDimension('Time',1)

        timestamp = ncfile.createVariable('time', np.float64, ('Time',),compression='zlib',least_significant_digit=8)
        timestamp.standard_name = 'time'
        timestamp.units = "days since 2018-01-01 00:00:00"
        timestamp.long_name = "Elapsed number of days for the period 2018-01-01 through " + str(i[-9:-5]) + "-" + str(i[-5:-3]) + "-15"
        timestamp.calendar = 'standard'

        timestamp[:] = int((datetime(int(year),int(month),15) - datetime(2018,1,1)).days) + 1

        surfrough_nc = ncfile.createVariable('surface_roughness_1km', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        surfrough_nc.long_name = 'Surface roughness at 1 km along-track'
        surfrough_nc.units = 'meters'
        surfrough_nc.description = 'surface roughness is calculated as the along-track standard deviation of surface elevations within 1 km non-overlapping segments'
        surfrough_nc.fill_value = 'nan'

        surfrough_count_nc = ncfile.createVariable('surface_roughness_1km_n', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        surfrough_count_nc.long_name = 'Number of measurements per grid cell'
        surfrough_count_nc.units = 'count'
        surfrough_count_nc.fill_value = 'nan'

        surfrough_interp_nc = ncfile.createVariable('surface_roughness_1km_interp', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        surfrough_interp_nc.long_name = 'Surface roughness at 1 km along-track interpolated'
        surfrough_interp_nc.units = 'meters'
        surfrough_interp_nc.description = 'surface roughness data from "surface_roughness_1km" interpolated using the GMT nearneighbor algorithm and masked with AMSR2 sea ice concentration data where SIC >= 15%'
        surfrough_interp_nc.fill_value = 'nan'

        surfrough_lt60_nc = ncfile.createVariable('surface_roughness_1km_ltT60', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        surfrough_lt60_nc.long_name = 'Surface roughness < 60 cm at 1 km along-track'
        surfrough_lt60_nc.units = 'meters'
        surfrough_lt60_nc.description = 'surface roughness is calculated as the along-track standard deviation of surface elevations < 60 cm above the local level ice surface, within 1 km non-overlapping segments'
        surfrough_lt60_nc.fill_value = 'nan'

        surfrough_lt60_count_nc = ncfile.createVariable('surface_roughness_1km_ltT60_n', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        surfrough_lt60_count_nc.long_name = 'Number of measurements per grid cell'
        surfrough_lt60_count_nc.units = 'count'
        surfrough_lt60_count_nc.fill_value = 'nan'

        surfrough_lt60_interp_nc = ncfile.createVariable('surface_roughness_1km_ltT60_interp', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        surfrough_lt60_interp_nc.long_name = 'Surface roughness < 60 cm at 1 km along-track interpolated'
        surfrough_lt60_interp_nc.units = 'meters'
        surfrough_lt60_interp_nc.description = 'surface roughness data from "surface_roughness_1km_ltT60" interpolated using the GMT nearneighbor algorithm and masked with AMSR2 sea ice concentration data where SIC >= 15%'
        surfrough_lt60_interp_nc.fill_value = 'nan'

        surfrough_lt20_nc = ncfile.createVariable('surface_roughness_1km_ltT20', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        surfrough_lt20_nc.long_name = 'Surface roughness < 20 cm at 1 km along-track'
        surfrough_lt20_nc.units = 'meters'
        surfrough_lt20_nc.description = 'surface roughness is calculated as the along-track standard deviation of surface elevations < 20 cm above the local level ice surface, within 1 km non-overlapping segments'
        surfrough_lt20_nc.fill_value = 'nan'

        surfrough_lt20_count_nc = ncfile.createVariable('surface_roughness_1km_ltT20_n', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        surfrough_lt20_count_nc.long_name = 'Number of measurements per grid cell'
        surfrough_lt20_count_nc.units = 'count'
        surfrough_lt20_count_nc.fill_value = 'nan'

        surfrough_lt20_interp_nc = ncfile.createVariable('surface_roughness_1km_ltT20_interp', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        surfrough_lt20_interp_nc.long_name = 'Surface roughness < 20 cm at 1 km along-track interpolated'
        surfrough_lt20_interp_nc.units = 'meters'
        surfrough_lt20_interp_nc.description = 'surface roughness data from "surface_roughness_1km_ltT20" interpolated using the GMT nearneighbor algorithm and masked with AMSR2 sea ice concentration data where SIC >= 15%'
        surfrough_lt20_interp_nc.fill_value = 'nan'

        max_sh_nc = ncfile.createVariable('max_sail_H_1km_T60', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        max_sh_nc.long_name = 'Maximum sail height at 1 km along-track using 60 cm threshold'
        max_sh_nc.units = 'meters'
        max_sh_nc.description = 'single maximum sail height per 1 km of along-track non-overlapping segments'
        max_sh_nc.fill_value = 'nan'

        max_sh_count_nc = ncfile.createVariable('max_sail_H_1km_T60_n', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        max_sh_count_nc.long_name = 'Number of measurements per grid cell'
        max_sh_count_nc.units = 'count'
        max_sh_count_nc.fill_value = 'nan'

        max_sh_interp_nc = ncfile.createVariable('max_sail_H_1km_T60_interp', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        max_sh_interp_nc.long_name = 'Maximum sail height at 1 km along-track using 60 cm threshold interpolated'
        max_sh_interp_nc.units = 'meters'
        max_sh_interp_nc.description = 'maximum sail height data from "max_sail_H_1km_T60" interpolated using the GMT nearneighbor algorithm and masked with AMSR2 sea ice concentration data where SIC >= 15%'
        max_sh_interp_nc.fill_value = 'nan'

        sh_nc = ncfile.createVariable('sail_H_T60', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        sh_nc.long_name = 'Sail height using 60 cm threshold'
        sh_nc.units = 'meters'
        sh_nc.description = 'sail height for ridge sails equal to or above the 60 cm threshold, relative to the local level ice surface, and meeting the Rayleigh criterion'
        sh_nc.fill_value = 'nan'

        sh_count_nc = ncfile.createVariable('sail_H_T60_n', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        sh_count_nc.long_name = 'Number of measurements per grid cell'
        sh_count_nc.units = 'count'
        sh_count_nc.fill_value = 'nan'

        sh_interp_nc = ncfile.createVariable('sail_H_T60_interp', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        sh_interp_nc.long_name = 'Sail height using 60 cm threshold interpolated'
        sh_interp_nc.units = 'meters'
        sh_interp_nc.description = 'sail height data from "sail_H_T60" interpolated using the GMT nearneighbor algorithm and masked with AMSR2 sea ice concentration data where SIC >= 15%'
        sh_interp_nc.fill_value = 'nan'

        spacing_nc = ncfile.createVariable('sail_D_T60', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=4)
        spacing_nc.long_name = 'Distance between consecutive ridge sails using 60 cm threshold'
        spacing_nc.units = 'meters'
        spacing_nc.description = 'distance between consecutive sail heights in "sail_H_T60"'
        spacing_nc.fill_value = 'nan'

        spacing_count_nc = ncfile.createVariable('sail_D_T60_n', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        spacing_count_nc.long_name = 'Number of measurements per grid cell'
        spacing_count_nc.units = 'count'
        spacing_count_nc.fill_value = 'nan'

        spacing_interp_nc = ncfile.createVariable('sail_D_T60_interp', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=4)
        spacing_interp_nc.long_name = 'Distance between consecutive ridge sails using 60 cm threshold interpolated'
        spacing_interp_nc.units = 'meters'
        spacing_interp_nc.description = 'data from "sail_D_T60" interpolated using the GMT nearneighbor algorithm and masked with AMSR2 sea ice concentration data where SIC >= 15%'
        spacing_interp_nc.fill_value = 'nan'

        RI_nc = ncfile.createVariable('ridging_I_1km_T60', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        RI_nc.long_name = 'Ridging intensity at 1 km along-track using 60 cm threshold'
        RI_nc.units = 'None'
        RI_nc.description = 'ridging intensity is the mean sail height multiplied by the sail frequency per 1 km along-track non-overlapping segments. It is a unitless measurement.'
        RI_nc.fill_value = 'nan'

        RI_count_nc = ncfile.createVariable('ridging_I_1km_T60_n', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        RI_count_nc.long_name = 'Number of measurements per grid cell'
        RI_count_nc.units = 'count'
        RI_count_nc.fill_value = 'nan'

        RI_interp_nc = ncfile.createVariable('ridging_I_1km_T60_interp', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        RI_interp_nc.long_name = 'Ridging intensity at 1 km along-track using 60 cm threshold interpolated'
        RI_interp_nc.units = 'None'
        RI_interp_nc.description = 'ridging intensity data from "ridging_I_1km_T60" interpolated using the GMT nearneighbor algorithm and masked with AMSR2 sea ice concentration data where SIC >= 15%'
        RI_interp_nc.fill_value = 'nan'

        width_nc = ncfile.createVariable('sail_W_T60', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=4)
        width_nc.long_name = 'Sail width using 60 cm threshold'
        width_nc.units = 'meters'
        width_nc.description = 'sail width at the 60 cm threshold from "sail_H_T60"'
        width_nc.fill_value = 'nan'

        width_count_nc = ncfile.createVariable('sail_W_T60_n', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        width_count_nc.long_name = 'Number of measurements per grid cell'
        width_count_nc.units = 'count'
        width_count_nc.fill_value = 'nan'

        width_interp_nc = ncfile.createVariable('sail_W_T60_interp', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=4)
        width_interp_nc.long_name = 'Sail width using 60 cm threshold interpolated'
        width_interp_nc.units = 'meters'
        width_interp_nc.description = 'sail width data from "sail_W_T60" interpolated using the GMT nearneighbor algorithm and masked with AMSR2 sea ice concentration data where SIC >= 15%'
        width_interp_nc.fill_value = 'nan'

        length_nc = ncfile.createVariable('ridge_L_T60', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=4)
        length_nc.long_name = 'Ridge length using 60 cm threshold'
        length_nc.units = 'meters'
        length_nc.description = 'ridge length determined from the points to the left and right of ridge sails from "sail_H_T60" where elevations first cross the local level sea ice surface'
        length_nc.fill_value = 'nan'

        length_count_nc = ncfile.createVariable('ridge_L_T60_n', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        length_count_nc.long_name = 'Number of measurements per grid cell'
        length_count_nc.units = 'count'
        length_count_nc.fill_value = 'nan'

        length_interp_nc = ncfile.createVariable('ridge_L_T60_interp', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=4)
        length_interp_nc.long_name = 'Ridge length using 60 cm threshold interpolated'
        length_interp_nc.units = 'meters'
        length_interp_nc.description = 'ridge length data from "ridge_L_T60" interpolated using the GMT nearneighbor algorithm and masked with AMSR2 sea ice concentration data where SIC >= 15%'
        length_interp_nc.fill_value = 'nan'

        RI_20_nc = ncfile.createVariable('ridging_I_1km_T20', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        RI_20_nc.long_name = 'Ridging intensity at 1 km along-track using 20 cm threshold'
        RI_20_nc.units = 'None'
        RI_20_nc.description = 'ridging intensity is the mean sail height multiplied by the sail frequency per 1 km along-track non-overlapping segments. It is a unitless measurement.'
        RI_20_nc.fill_value = 'nan'

        RI_20_count_nc = ncfile.createVariable('ridging_I_1km_T20_n', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        RI_20_count_nc.long_name = 'Number of measurements per grid cell'
        RI_20_count_nc.units = 'count'
        RI_20_count_nc.fill_value = 'nan'

        RI_20_interp_nc = ncfile.createVariable('ridging_I_1km_T20_interp', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        RI_20_interp_nc.long_name = 'Ridging intensity at 1 km along-track using 20 cm threshold interpolated'
        RI_20_interp_nc.units = 'None'
        RI_20_interp_nc.description = 'ridging intensity data from "ridging_I_1km_T20" interpolated using the GMT nearneighbor algorithm and masked with AMSR2 sea ice concentration data where SIC >= 15%'
        RI_20_interp_nc.fill_value = 'nan'

        max_sh_20_nc = ncfile.createVariable('max_sail_H_1km_T20', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        max_sh_20_nc.long_name = 'Maximum sail height at 1 km along-track using 20 cm threshold'
        max_sh_20_nc.units = 'meters'
        max_sh_20_nc.description = 'single maximum ridge sail height per 1 km of along-track non-overlapping segments'
        max_sh_20_nc.fill_value = 'nan'

        max_sh_20_count_nc = ncfile.createVariable('max_sail_H_1km_T20_n', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        max_sh_20_count_nc.long_name = 'Number of measurements per grid cell'
        max_sh_20_count_nc.units = 'count'
        max_sh_20_count_nc.fill_value = 'nan'

        max_sh_20_interp_nc = ncfile.createVariable('max_sail_H_1km_T20_interp', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        max_sh_20_interp_nc.long_name = 'Maximum sail height at 1 km along-track using 20 cm threshold interpolated'
        max_sh_20_interp_nc.units = 'meters'
        max_sh_20_interp_nc.description = 'maximum sail height data from "max_sail_H_1km_T20" interpolated using the GMT nearneighbor algorithm and masked with AMSR2 sea ice concentration data where SIC >= 15%'
        max_sh_20_interp_nc.fill_value = 'nan'

        sh_20_nc = ncfile.createVariable('sail_H_T20', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        sh_20_nc.long_name = 'Sail height using 20 cm threshold'
        sh_20_nc.units = 'meters'
        sh_20_nc.description = 'sail height for ridge sails equal to or above the 20 cm threshold, relative to the local level ice surface, and meeting the Rayleigh criterion'
        sh_20_nc.fill_value = 'nan'

        sh_20_count_nc = ncfile.createVariable('sail_H_T20_n', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        sh_20_count_nc.long_name = 'Number of measurements per grid cell'
        sh_20_count_nc.units = 'count'
        sh_20_count_nc.fill_value = 'nan'

        sh_20_interp_nc = ncfile.createVariable('sail_H_T20_interp', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        sh_20_interp_nc.long_name = 'Sail height using 20 cm threshold interpolated'
        sh_20_interp_nc.units = 'meters'
        sh_20_interp_nc.description = 'sail height data from "sail_H_T20" interpolated using the GMT nearneighbor algorithm and masked with AMSR2 sea ice concentration data where SIC >= 15%'
        sh_20_interp_nc.fill_value = 'nan'

        spacing_20_nc = ncfile.createVariable('sail_D_T20', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=4)
        spacing_20_nc.long_name = 'Distance between consecutive ridge sails using 20 cm threshold'
        spacing_20_nc.units = 'meters'
        spacing_20_nc.description = 'Distance between consecutive sail heights in "sail_H_T20"'
        spacing_20_nc.fill_value = 'nan'

        spacing_20_count_nc = ncfile.createVariable('sail_D_T20_n', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        spacing_20_count_nc.long_name = 'Number of measurements per grid cell'
        spacing_20_count_nc.units = 'count'
        spacing_20_count_nc.fill_value = 'nan'

        spacing_20_interp_nc = ncfile.createVariable('sail_D_T20_interp', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=4)
        spacing_20_interp_nc.long_name = 'Distance between consecutive ridge sails using 20 cm threshold interpolated'
        spacing_20_interp_nc.units = 'meters'
        spacing_20_interp_nc.description = 'data from "sail_D_T20" interpolated using the GMT nearneighbor algorithm and masked with AMSR2 sea ice concentration data where SIC >= 15%'
        spacing_20_interp_nc.fill_value = 'nan'

        width_20_nc = ncfile.createVariable('sail_W_T20', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=4)
        width_20_nc.long_name = 'Sail width using 20 cm threshold'
        width_20_nc.units = 'meters'
        width_20_nc.description = 'sail width at the 20 cm threshold from "sail_H_T60"'
        width_20_nc.fill_value = 'nan'

        width_20_count_nc = ncfile.createVariable('sail_W_T20_n', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        width_20_count_nc.long_name = 'Number of measurements per grid cell'
        width_20_count_nc.units = 'count'
        width_20_count_nc.fill_value = 'nan'

        width_20_interp_nc = ncfile.createVariable('sail_W_T20_interp', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=4)
        width_20_interp_nc.long_name = 'Sail width using 20 cm threshold interpolated'
        width_20_interp_nc.units = 'meters'
        width_20_interp_nc.description = 'sail width data from "sail_W_T20" interpolated using the GMT nearneighbor algorithm and masked with AMSR2 sea ice concentration data where SIC >= 15%'
        width_20_interp_nc.fill_value = 'nan'

        length_20_nc = ncfile.createVariable('ridge_L_T20', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=4)
        length_20_nc.long_name = 'Ridge length using 20 cm threshold'
        length_20_nc.units = 'meters'
        length_20_nc.description = 'ridge length determined from the points to the left and right of ridge sails from "sail_H_T20" where elevations cross the local level sea ice surface'
        length_20_nc.fill_value = 'nan'

        length_20_count_nc = ncfile.createVariable('ridge_L_T20_n', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        length_20_count_nc.long_name = 'Number of measurements per grid cell'
        length_20_count_nc.units = 'count'
        length_20_count_nc.fill_value = 'nan'

        length_20_interp_nc = ncfile.createVariable('ridge_L_T20_interp', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=4)
        length_20_interp_nc.long_name = 'Ridge length using 20 cm threshold intepolated'
        length_20_interp_nc.units = 'meters'
        length_20_interp_nc.description = 'ridge length data from "ridge_L_T20" interpolated using the GMT nearneighbor algorithm and masked with AMSR2 sea ice concentration data where SIC >= 15%'
        length_20_interp_nc.fill_value = 'nan'


        if (grid_res == 3125):
            SIC_data = pd.read_csv('Path to 3.125 km AMSR2 or other SIC file',sep='\t',header=None)
        else:
            SIC_data = pd.read_csv('Path to 10 km or 25 km AMSR2 or other SIC file',sep='\t',header=None)
        

        SIC_lon = SIC_data[0].values.reshape(grid_xy,grid_xy)
        SIC_lat = SIC_data[1].values.reshape(grid_xy,grid_xy)
        SIC = SIC_data[2].values.reshape(grid_xy,grid_xy)

        SIC[SIC < 15.0] = np.nan

        SIC_WGS84_coords = reproject_to_WGS84(SIC_data[0].values,SIC_data[1].values)
        SIC_WGS84_lon = SIC_WGS84_coords[0].reshape(grid_xy,grid_xy)
        SIC_WGS84_lat = SIC_WGS84_coords[1].reshape(grid_xy,grid_xy)

        sic_nc = ncfile.createVariable('sea_ice_conc', np.float32, ('Time','y','x'),compression='zlib',least_significant_digit=6)
        sic_nc.long_name = 'AMSR2 sea ice concentration (>=15%)'
        sic_nc.units = '%'
        sic_nc.description = 'monthly mean sea ice concentration derived by calculating mean sea ice concentration per grid cell for an entire month and retaining values >= 15%.'
        sic_nc.fill_value = 'nan'
        sic_nc.data_source = 'University of Bremen, "https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/n3125/", Arctic, 3.125 km'

        sic_nc[:,:] = SIC

        spacing = [float(grid_res)]
        bins = [grid_xy]

        params = [surfrough_arr,surfrough_lt60_arr,surfrough_lt20_arr,max_sh_arr,sh_arr,spacing_arr,RI_arr,width_arr,length_arr,RI_20_arr,max_sh_20_arr,sh_20_arr,spacing_20_arr,width_20_arr,length_20_arr]
        params_lon = [surfrough_lon_arr,surfrough_lt60_lon_arr,surfrough_lt20_lon_arr,max_sh_lon_arr,sh_lon_arr,spacing_lon_arr,RI_lon_arr,width_lon_arr,length_lon_arr,RI_20_lon_arr,max_sh_20_lon_arr,sh_20_lon_arr,
                      spacing_20_lon_arr,width_20_lon_arr,length_20_lon_arr]
        params_lat = [surfrough_lat_arr,surfrough_lt60_lat_arr,surfrough_lt20_lat_arr,max_sh_lat_arr,sh_lat_arr,spacing_lat_arr,RI_lat_arr,width_lat_arr,length_lat_arr,RI_20_lat_arr,max_sh_20_lat_arr,sh_20_lat_arr,
                      spacing_20_lat_arr,width_20_lat_arr,length_20_lat_arr]

        names = ['surfrough','surfrough_lt60','surfrough_lt20','max_sh','sh','spacing','RI','width','length','RI_20','max_sh_20','sh_20','spacing_20','width_20','length_20']

        nc_groups = [surfrough_nc,surfrough_lt60_nc,surfrough_lt20_nc,max_sh_nc,sh_nc,spacing_nc,RI_nc,width_nc,length_nc,RI_20_nc,max_sh_20_nc,sh_20_nc,spacing_20_nc,width_20_nc,length_20_nc]
        nc_count = [surfrough_count_nc,surfrough_lt60_count_nc,surfrough_lt20_count_nc,max_sh_count_nc,sh_count_nc,spacing_count_nc,RI_count_nc,width_count_nc,length_count_nc,RI_20_count_nc,max_sh_20_count_nc,sh_20_count_nc,
                    spacing_20_count_nc,width_20_count_nc,length_20_count_nc]
        nc_interp = [surfrough_interp_nc,surfrough_lt60_interp_nc,surfrough_lt20_interp_nc,max_sh_interp_nc,sh_interp_nc,spacing_interp_nc,RI_interp_nc,width_interp_nc,length_interp_nc,RI_20_interp_nc,max_sh_20_interp_nc,
                     sh_20_interp_nc,spacing_20_interp_nc,width_20_interp_nc,length_20_interp_nc]

        for z in range(len(params)):
            print (names[z])
            for p in range(len(spacing)):

                non_outlier_idx = np.where(params[z] < np.nanpercentile(params[z],99.75))[0] 

                new_coords = reproject_to_PSG(params_lon[z][non_outlier_idx],params_lat[z][non_outlier_idx])

                finite_idx = np.where(np.isfinite(new_coords[0]) == True)[0]

                bin_count_final = binned_statistic_2d(new_coords[0][finite_idx],new_coords[1][finite_idx],values=params[z][non_outlier_idx][finite_idx],
                                                statistic='count',bins=bins[p],range=[[-5400000,5400000],[-5400000,5400000]])

                bin_count_final[0][np.where(bin_count_final[0] < 3)] = np.nan

                nc_count[z][:,:] = bin_count_final[0]

                gridded = pygmt.xyz2grd(x=new_coords[0][finite_idx],y=new_coords[1][finite_idx],z=params[z][non_outlier_idx][finite_idx],
                                            spacing=float(grid_res),
                                            region=[-5400000.0+(grid_res/2.0),5400000.0-(grid_res/2.0),-5400000.0+(grid_res/2.0),5400000.0-(grid_res/2.0)])

                gridded.values.T[np.where(np.isnan(bin_count_final[0]) == True)] = np.nan

                nc_groups[z][:,:] = gridded.values.T

                gridded_xyz = pygmt.grd2xyz(grid=gridded, output_type='numpy')

                gridded_xyz_lon_reshape = np.rot90(gridded_xyz[:,0].reshape(grid_xy,grid_xy),k=-1)
                gridded_xyz_lat_reshape = np.rot90(gridded_xyz[:,1].reshape(grid_xy,grid_xy),k=-1)
                gridded_xyz_reshape = np.rot90(gridded_xyz[:,2].reshape(grid_xy,grid_xy),k=-1)

                grid_good = np.where(np.isnan(gridded_xyz_reshape) == False)


                data_interp_nn = pygmt.nearneighbor(x=gridded_xyz_lon_reshape[grid_good].ravel(),y=gridded_xyz_lat_reshape[grid_good].ravel(),z=gridded_xyz_reshape[grid_good].ravel(),
                                                    spacing="3125+e",
                                                    region=[-5400000.0+(grid_res/2.0),5400000.0-(grid_res/2.0),-5400000.0+(grid_res/2.0),5400000.0-(grid_res/2.0)],
                                                    search_radius="100000",
                                                    sectors="18+m3")


                data_interp_nn.values.T[grid_good] = gridded_xyz_reshape[grid_good]
                data_interp_nn.values.T[np.where(np.isnan(SIC) == True)] = np.nan
                data_interp_nn.values.T[np.where(SIC_WGS84_lat > 88.0)] = np.nan
                data_interp_nn.values.T[np.where(SIC_WGS84_lat < 60.0)] = np.nan

                nc_interp[z][:,:] = data_interp_nn.values.T


        ncfile.close()


print("%s seconds" % (time.time() - start_time))
