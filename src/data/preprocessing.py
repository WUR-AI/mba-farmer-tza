from xyzservices.providers import data_path
from IPython.core import display_trap
from IPython.core import display_trap
from IPython.core import display_trap
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import json
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from CAgsalML.transformers import SpatialTransform
from scipy.spatial import cKDTree

TIMINGS = [
    'basal', 'kneehigh', '56leaves', '810leaves',
    'tasselingsilking'
]

FIELD_COLS = [
    'fmba_campaign_year', 'FRMRgender', 'FRMRregion', 'FRMRdistrict',
    'FRMRward', 'previousadvice', 'mainharvestcrop', 'minorharvestcrop',
    'crophistory', 'crophistory_cropnumber', 'crophistory_intercrop',
    'fieldslope', 'landscape', 'soil_texture', 'est_harvestarea_ha',
    'gpsharvestarea_corrected_ha', 'area_diff_m2', 'harvestarea_ha',
    # Residues refer to last season, that's why it goes on field data
    'maizeresidues', 'beansresidues', 'gnutsresidues', 'soyaresidues', 

    'instanceID', 'surveyDate', 'start', 'end', 'PHONEmodel', 'PHONEcode', 
    'is_revisit', 'is_splitted', 'is_duplicated', 'n_overlap',
    'id_unique', 'id_cross', 'is_duplicated_multiple', 'is_duplicated_saved',
    'is_tiny_area', 'geometry'
]
MAIZE_COLS = [
    'maizeprodkg_ha', 'maizemarketing',
    'maizeharvest_seperate', 'maizeharvestsheller', 'maizevariety',
    'maizerecycledseed', 'maizefieldprep', 'maizeplantingmethod',
    'maizeplantmonth', 'maizeplantmonthpart', 'maizedryplant',
    'maizePdeficiency', 'maizeKdeficiency', 'maizeherbicides',
    'maizeweedings', 'maizepestdisease', 'maizekgseed_ha'
]
FERT_COLS = [
    'manurefreq', 'manure', 'fertiliser_use', 'basal_application',
    'basaltypes', 'KGbasal_ha', 'KGNbasal_ha', 'KGPbasal_ha', 'KGKbasal_ha',
    'TShbasal_ha', 'topnumber', 'top1timing', 'top1application',
    'top1types', 'KGtop1_ha', 'KGNtop1_ha', 'KGPtop1_ha', 'KGKtop1_ha',
    'TShtop1_ha', 'top2timing', 'top2application', 'top2types', 'KGtop2_ha',
    'KGNtop2_ha', 'KGPtop2_ha', 'KGKtop2_ha', 'TShtop2_ha', 'top3timing',
    'top3application', 'top3types', 'KGtop3_ha', 'KGNtop3_ha', 'KGPtop3_ha',
    'KGKtop3_ha', 'TShtop3_ha', 'KGtop_ha', 'KGNtop_ha', 'KGPtop_ha',
    'KGKtop_ha', 'TShtop_ha', 'KGtotal_ha', 'KGNtotal_ha', 'KGPtotal_ha',
    'KGKtotal_ha', 'TShtotal_ha', 'nomaizefertilisers',
    'KGnomaizefertiltype1_ha', 'KGnomaizefertiltype2_ha',
    'KGnomaizefertil_ha', 'KGNnomaizefertil_ha', 'KGPnomaizefertil_ha',
    'KGKnomaizefertil_ha', 'TShnomaizefertil_ha'
]
SOIL_VARS = [
    'nitrogen_total_mean_0_20', 'clay_content_mean_0_20', 'sand_content_mean_0_20', 
    'bulk_density_mean_0_20', 'cation_exchange_capacity_mean_0_20', 
    'ph_mean_0_20', 'carbon_organic_mean_0_20'
]

COLS_TO_DROP =[
    'fmba_campaign_year', 'FRMRregion', 'FRMRdistrict', 'FRMRward',
    'minorharvestcrop', 'crophistory_intercrop',
    'surveyDate', 'start', 'end', 'PHONEmodel', 'PHONEcode',
    'is_splitted', 'is_duplicated', 'n_overlap', 'id_unique', 'id_cross',
    'is_duplicated_multiple', 'is_duplicated_saved', 'is_tiny_area'
]

def load_raw_data_and_add_external(data_path: str, data_version: str = '260116'):
    """
    It loads raw survey data and adds external data. It performs a high-level pre-processing,
    such as removing non-important columns, and filtering the subset of our interest.
    """ 
    # Raw data
    data = pd.read_csv(f'{data_path}/survey/05_FMBA24sf2_no-advice_{data_version}_n3754.csv', sep='\t')
    data = data.set_index('fieldID')
    data = data[data.mainharvestcrop == 'maize'].copy()
    data = data.drop(columns="geometry") # I'll drop geometry, I'll add lat, lon from geospatial features

    # Import geometries
    crs = 'EPSG:32736'
    points = gpd.read_file(f'{data_path}/geo/points.geojson')
    points = points.set_index('fieldID')[['geometry']]
    points = points.loc[data.index]
    points = points.to_crs(crs)
    points['lat'] = points.geometry.y
    points['lon'] = points.geometry.x

    # Import admin units
    admin = gpd.read_file(f'{data_path}/geo/tza_shp/tza_admbnda_adm3_20181019.shp')
    admin = admin.to_crs(crs)
    points = gpd.sjoin(
        points, admin[['ADM3_PCODE', 'ADM2_PCODE', 'geometry']], 
        how="left", predicate="within"
    )
    points = points.drop(columns=['index_right'])
    # There is only a few points there, then I'll move then to the neighboring unit
    points['ADM2_PCODE'] = points['ADM2_PCODE'].replace({'TZ2610': 'TZ2609'}) 

    # Import soil data
    soil_data = pd.read_pickle(f'{data_path}/external-covs/soil-data.pkl')
    soil_data = soil_data[SOIL_VARS]
    # Phenology data
    phenology = pd.read_pickle(f'{data_path}/external-covs/maize-soya-phenology.pkl')
    # Weather data
    weather_data_daily = pd.read_pickle(f'{data_path}/external-covs/weather-data-daily-2024-linearinterp.pkl')
    weather_data_daily.columns = ['tmean', 'tmax', 'tmin', 'vp', 'prec', 'srad', 'ws']
    weather_data_daily['tmean'] -= 273.15
    weather_data_daily['tmax'] -= 273.15
    weather_data_daily['tmin'] -= 273.15
    weather_data_daily['srad'] /= 1e6

    rain_threshold = 1
    def get_weather_vars(idx):
        """
        Generates the weather variables partitioned in phenological ranges
        """
        tmp_df = weather_data_daily.loc[idx]
        if idx not in phenology.index:
            season_dates = pd.date_range('2023-12-01', '2024-05-31')
        else:
            season_dates = pd.date_range(phenology.loc[idx, 'estimated_sos'], phenology.loc[idx, 'estimated_eos'])
        periods = {
            f'{(i*10)}to{10*(i+1)}perc_season': dates
            for i, dates in zip(range(0, 10), np.array_split(season_dates, 10))
        }
        vars_dict = {}
        for per_name, per in periods.items():
            # for var in ['tmin', 'tmean', 'tmax', 'srad']:
            for var in ['tmean', 'srad']:
                vars_dict[f'{per_name}_{var}'] = tmp_df[var].loc[per].mean()
            # As the season lenght and period is variable, then it's better to have mean precipitation
            # instead of sum. Mean would be a way to normalize for season lenght. 
            vars_dict[f'{per_name}_prec'] = tmp_df.loc[per, 'prec'].mean() 
            vars_dict[f'{per_name}_rainydays'] = (tmp_df.loc[per, 'prec'] > rain_threshold).mean()
            
        per_name = '0to100perc_season'
        for var in ['tmean', 'srad']:
            vars_dict[f'{per_name}_{var}'] = tmp_df[var].loc[season_dates].mean()
        vars_dict[f'{per_name}_prec'] = tmp_df.loc[season_dates, 'prec'].mean()
        vars_dict[f'{per_name}_rainydays'] = (tmp_df.loc[season_dates, 'prec'] > rain_threshold).mean()
        return vars_dict

    weather_data = pd.DataFrame.from_dict({
        i: get_weather_vars(i)
        for i in data.index
    }, orient='index')
    # Accessibility index
    inindex = pd.read_csv(f'{data_path}/external-covs/inaccessibility-index.csv')
    inindex = inindex.set_index('fieldID')
    inindex.columns = ['accessibility_rai']
    # Attach weather, soil, RAI index  data to data frame.
    data = data.join(soil_data).join(weather_data).join(inindex).join(points[['lat', 'lon', "ADM2_PCODE"]])
    data = data.drop(columns=COLS_TO_DROP)
    return data

def clean_raw_data(data, data_ranges):
    """
    It cleans the raw data by replacing non-important columns, 
    such as 'none', 'None', 'NONE', 'NAN', 'NaN', with None.
    It also removes the records that are outside the range defined in data_ranges
    """

    data = data.replace({'none': None, 'None': None, 'NONE': None, 'NAN': None, 'NaN': None})
    data = data[data.maizeprodkg_ha.notna()]

    # Assign amount of zero, when no fertilizer was applied
    farmer_applied_fertilizer = data[f'basal_application'].notna()
    for element in ('N', 'P', 'K', ''):
        data.loc[~farmer_applied_fertilizer, f'KG{element}basal_ha'] = 0

    for i in range(1, 4):
        farmer_applied_fertilizer = data[f'top{i}application'].notna()
        for element in ('N', 'P', 'K', ''):
            data.loc[~farmer_applied_fertilizer, f'KG{element}top{i}_ha'] = 0

    # Re-calculate total amounts
    for element in ('N', 'P', 'K', ''):
        data[f'KG{element}total_ha'] = \
            data[[f'KG{element}basal_ha']+[f'KG{element}top{i}_ha' for i in range(1,4)]].sum(axis=1, skipna=False)

    # Set to NaN every continuous value outside the range defined in data_ranges
    for col, bounds in data_ranges.items():
        min_val = bounds.get('min')
        max_val = bounds.get('max')
        numeric_col = data[col]
        if min_val is not None:
            data.loc[numeric_col < min_val, col] = np.nan
        if max_val is not None:
            data.loc[numeric_col > max_val, col] = np.nan
    return data

def transform_soil(data):
    """
    Transforms the soil data to create useful representations
    """
    data = data.rename(columns={
        'carbon_organic_mean_0_20': 'soilfert_oc', 'cation_exchange_capacity_mean_0_20': 'soilfert_cec',
        'clay_content_mean_0_20': 'soilpp_clay', 'sand_content_mean_0_20': 'soilpp_sand', 
        'bulk_density_mean_0_20': 'soilpp_density', 
    })
    data['soil_sandy'] = data.soil_texture == 'sandy'
    return data
    
def transform_fert(data):
    """
    Transform fertilizer application data
    """
    fert_df = data[['KGNbasal_ha', 'KGPbasal_ha']].copy()
    fert_df.columns = ['fertN_basal', 'fertP_basal']

    # Group fertilizer applications by timing and nutrient
    for element in ('N', 'P'):
        for i in range(1, 4):
            tmp_df = data[[f'top{i}timing', f'KG{element}top{i}_ha']].copy()
            tmp_df[f'top{i}timing'] = tmp_df[f'top{i}timing'].replace({'tasseling': 'tasseling-silking', 'silking': 'tasseling-silking'})
            tmp_df = pd.pivot_table(
                tmp_df,
                values=f'KG{element}top{i}_ha', columns=f'top{i}timing',
                index='fieldID', fill_value=0, dropna=True
            )
            tmp_df = tmp_df.reindex(data.index)
            tmp_df.columns = tmp_df.columns.str.replace('-', '')
            tmp_df.columns = [f'fert{element}_{i}' for i in tmp_df.columns]
            for col in tmp_df.columns:
                if col in fert_df.columns:
                    nan_rows = fert_df[col].isna() & tmp_df[col]
                    fert_df[col] = fert_df[col].add(tmp_df[col], fill_value=0)
                    fert_df.loc[nan_rows, col] = None
                else:
                    fert_df[col] = tmp_df[col]

    # We will combine the planting and emergence applications into a single one as 
    # they are the same in practice
    for e in ('N', 'P'):
        fert_df[f'fert{e}_basal'] = fert_df[[f'fert{e}_basal', f'fert{e}_emergence']].sum(axis=1)
        fert_df = fert_df.drop(columns=[f'fert{e}_emergence'])
    TIMINGS = [
        'basal', 'kneehigh', '56leaves', '810leaves',
        'tasselingsilking'
    ]
    fert_df_alt = pd.DataFrame(index=fert_df.index)
    # Create total N, P and timing dummy variables
    for e in ('N', 'P'):
        # min_count=1 ensures that if at least 1 value is non-na, sum is computed and different from NaN
        fert_df_alt[f'fert{e}_total'] = fert_df[[f'fert{e}_{t}' for t in TIMINGS]].sum(axis=1, min_count=1)
        fert_df_alt[f'fert{e}_total'] = fert_df_alt[f'fert{e}_total'].where(lambda x: x<250, np.nan)
        fert_df_alt[[f'fert{e}_{t}' for t in TIMINGS]] = \
            fert_df[[f'fert{e}_{t}' for t in TIMINGS]].div(fert_df_alt[f'fert{e}_total'], axis=0)
        # fert_df_alt = fert_df_alt.dropna()

    fert_df = fert_df_alt
    fert_df['fertP_any'] = fert_df.fertP_total > 0

    return data.join(fert_df)

def transform_variety(data):
    variety_df = pd.DataFrame(index=data.index)
    variety_df['variety_hybrid'] = ~data.maizevariety.isin(['LOCAL_VARIETY', 'CANNOT-REMEMBER-VARIETY', 'UNKNOWN_VARIETY'])
    variety_df['variety_hybrid'] = ~variety_df.fillna(data.soyavariety.isin(['local', 'dontknow']))
    variety_df['variety_recycled'] = data.maizerecycledseed == 'recycled'
    variety_df['variety_recycled'] = variety_df.variety_recycled.fillna(
        data.maizerecycledseed.isin(['own', 'otherfarmer', 'relative_friend'])
    )
    return data.join(variety_df)

def transform_field_history(data):
    crop_groups = {
        'legume': ['beans', 'groundnuts', 'pigeonpea', 'soya'],
        'rootandtuber': ['cassava', 'roundpotatoes', 'sweetpotatoes'],
        'oilseed': ['sesame', 'sunflower']
    }

    def get_group_history(history:str):
        hist = {
            'history_maize': 'maize' in history,
            'history_nocrop': 'no_crop' in history
        }
        for group_name, crops in crop_groups.items():
            hist[f'history_{group_name}'] = any(map(lambda x: x in history, crops))
        return hist
        
    history_df = pd.DataFrame.from_dict(data.crophistory.map(get_group_history).to_dict(), orient='index')
    history_df['history_maizelegume'] = history_df.history_maize & history_df.history_legume
    return data.join(history_df)

def transform_other_management(data):
    # Residue will not be incorporated because it was measured for the end of the season, 
    # therefore it could have not affected production. However it could serve as proxy,
    # farmers doing it this time are more likely to do it every season
    residue_df = pd.get_dummies(
        data.maizeresidues.where(
            data.maizeresidues.fillna(data.soyaresidues).isin(
                ['leave on field', 'burn', 'feed to animals', 'incorporate in the soil']
            ), 
            '0_other'
        ),
        prefix='residue_', drop_first=True, prefix_sep='' 
    )
    residue_df.columns = [col.replace(' ', '') for col in residue_df.columns]
    residue_df['residue_onfield'] =  residue_df[['residue_incorporateinthesoil', 'residue_leaveonfield']].any(axis=1)

    # Herbicide
    herbicide_df = pd.DataFrame(index=data.index)
    herbicide_df['herbicide_post'] = data.maizeherbicides.str.contains('post-emergence').astype(bool)
    herbicide_df['herbicide_pre'] = data.maizeherbicides.str.contains('pre-emergence').astype(bool)
    herbicide_df['herbicide_any'] = herbicide_df.any(axis=1)

    # Manure
    manure_df = pd.get_dummies(data.manurefreq, drop_first=False).iloc[:, [0, 2, 3]]
    manure_df.columns = ['manure_everyseason', 'manure_oneinmany', 'manure_oneintwo']
    manure_df['manure_any'] = manure_df.any(axis=1)
    # Leave this to be the only manure variable
    manure_df = manure_df[['manure_any']]

    # Planting methods
    # This could be used as a proxy of farmer knowledge. 
    planting_df = pd.DataFrame(index=data.index)
    planting_df = planting_df.join(pd.get_dummies(
        data.maizeplantingmethod.fillna(data.soyaplantingmethod), 
        drop_first=True, prefix='planting'
    ))
    planting_df.columns = planting_df.columns.str.replace('-', '')

    # Other Management
    othermgmt_df = pd.concat([herbicide_df, manure_df, planting_df, residue_df], axis=1)
    return data.join(othermgmt_df)
    
def add_investing_capacity_index(data):
    """
    Calculates an 'Investing Capacity Index' (0-1) for farmers based on survey data.
    
    Dimensions:
    1. Land Score: Percentile rank of harvest area.
    2. Mechanization Score: Max level of tech used in prep/planting (Tractor > Oxen > Hand).
    3. Input Access Score: Use of Fertilizer, Manure, Herbicides, Purchased Seeds, Inoculants.
    
    Args:
    df (pd.DataFrame): The dataframe containing the survey data.
    
    Returns:
    pd.DataFrame: The original dataframe with an added column 'investing_capacity_index'
                  and sub-score columns for analysis.
    """
    ici_df = data.copy()
    
    # --- 1. Land Score (Percentile Rank) ---
    # We use est_harvestarea_ha or harvestarea_ha. 
    # Using rank(pct=True) is robust to outliers and gives a 0-1 uniform distribution.
    q05 = ici_df.harvestarea_ha.quantile(.05)
    q95 = ici_df.harvestarea_ha.quantile(.95)
    ici_df['score_land'] = (ici_df.harvestarea_ha.clip(lower=q05, upper=q95) - q05)/(q95 - q05)
        
    # --- 2. Mechanization Score ---
    # Define mapping: Tractor=1, Animal=0.5, Hand=0
    def get_mech_score(val):
        if pd.isna(val): return 0
        val_str = str(val).lower()
        if 'tractor' in val_str or 'machine' in val_str:
            return 1.0
        elif 'ox' in val_str or 'cattle' in val_str or 'plough' in val_str or 'animal' in val_str:
            return 0.5
        else:
            return 0.0

    # Columns related to land prep and planting for all crops
    mech_cols = [c for c in ici_df.columns if 'fieldprep' in c or 'plantingmethod' in c]
    
    # Calculate max score across all relevant columns for each farmer
    # If they use a tractor anywhere, they get the max score.
    mech_scores = ici_df[mech_cols].map(get_mech_score)
    ici_df['score_mech'] = mech_scores.max(axis=1)

    # --- 3. Input Access Score ---
    input_scores = pd.DataFrame(index=ici_df.index)

    # A. Fertilizer (Binary)
    if 'fertiliser_use' in ici_df.columns:
        # map 'yes'->1, 'little'->0.5, 'no'->0
        input_scores['fert'] = ici_df['fertiliser_use'].map(
            {'yes': 1, 'little': 0.5, 'no': 0, 'dontknow': 0}).fillna(0)
            
    # B. Manure (Binary)
    if 'manure' in ici_df.columns:
        input_scores['manure'] = ici_df['manure'].map(
            {'yes': 1, 'little': 0.5, 'no': 0, 'dontknow': 0}).fillna(0)
    
    # C. Herbicides (Binary: Any use vs no-herbicide)
    # Check all herbicide columns (maize, beans, etc)
    herb_cols = [c for c in ici_df.columns if 'herbicides' in c]
    if herb_cols:
        # If any column has a value that is NOT 'no-herbicide' and not nan, it's a 1.
        def check_herbicide(row):
            for col in herb_cols:
                val = str(row[col]).lower()
                if 'pre-emergence' in val or 'post-emergence' in val:
                    return 1.0
            return 0.0
        input_scores['herb'] = ici_df.apply(check_herbicide, axis=1)

    # D. Seeds (Commercial vs Recycled)
    # Maize: recycled=yes means LOW capital.
    if 'maizerecycledseed' in ici_df.columns:
        # Recycled? Yes=0, No=1
        input_scores['seed_maize'] = ici_df['maizerecycledseed'].map(
            {'no': 1, 'yes': 0, 'little': 0.5}).fillna(0)
            
    # # Beans/Gnuts/Soya: source 'own' vs 'market'
    # # We look for "seedsource" columns
    # seed_source_cols = [c for c in data.columns if 'seedsource' in c]
    # def check_seed_source(val):
    #     if pd.isna(val): return np.nan # Skip if crop not grown
    #     val = str(val).lower()
    #     if 'own' in val or 'neighbor' in val or 'friend' in val:
    #         return 0.0
    #     return 1.0 # Agrodealer, market, government, etc.

    # for col in seed_source_cols:
    #     input_scores[col] = data[col].apply(check_seed_source)

    # Average the input scores (ignoring NaNs for crops not grown)
    ici_df['score_inputs'] = input_scores.mean(axis=1).fillna(0)

    # --- 4. Final Composite Index ---
    ici_df['investing_capacity_index'] = ici_df[['score_land', 'score_mech', 'score_inputs']].mean(axis=1)
    
    # Standardize to 0-1 strictly (optional, usually mean is already 0-1)
    # We round for cleaner output
    ici_df['investing_capacity_index'] = ici_df['investing_capacity_index'].round(3)

    return data.join(ici_df[['investing_capacity_index', 'score_land', 'score_mech', 'score_inputs']])


def load_and_preprocess_data(data_path: str, data_version: str = '260116'):
    """
    Loads and preprocesses the MBA Tanzania data.
    
    Args:
        data_path (str): Path to the data directory (e.g., '/data/mba-tza')
        data_version (str): The version string for the dataset
        
    Returns:
        tuple: (data, transformed_data, points, clean_data, demeaned_outcome, demeaned_treatment)
    """
    data_cache_path = f'outputs/cache/data_preprocessed_{data_version}-transformed.pkl'
    if os.path.exists(data_cache_path):
        print(f"Opening cached {data_cache_path}...")
        data = pd.read_pickle(data_cache_path)
        return data

    data_cache_path = f'outputs/cache/data_preprocessed_{data_version}.pkl' 
    if os.path.exists(data_cache_path):
        data = pd.read_pickle(data_cache_path)
        print(f"Opening cached {data_cache_path}...")
        ranges_path = f'outputs/cache/data_ranges_{data_version}.json'
        if os.path.exists(ranges_path):
            with open(ranges_path, 'r') as f:
                data_ranges = json.load(f)
    else: 
        print("Loading raw data...")
        data = load_raw_data_and_add_external(data_path, data_version)
        # Ensure every column that can be considered a number is typed as numeric
        for col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col])
            except (ValueError, TypeError):
                converted = pd.to_numeric(data[col].replace({'none': None, 'None': None, 'NONE': None, 'NAN': None, 'NaN': None}), errors='coerce')
                orig_non_na = data[col].replace({'none': None, 'None': None, 'NONE': None, 'NAN': None, 'NaN': None}).notna()
                if orig_non_na.any() and (converted.notna() == orig_non_na).all():
                    data[col] = converted
        data.to_pickle(data_cache_path)
        # Save continuous variable ranges for subsequent data cleaning
        ranges_path = f'outputs/cache/data_ranges_{data_version}.json'
        numeric_df = data.select_dtypes(include=[np.number])
        data_ranges = {
            col: {
                'min': float(numeric_df[col].min()) if not pd.isna(numeric_df[col].min()) else None,
                'max': float(numeric_df[col].max()) if not pd.isna(numeric_df[col].max()) else None
            }
            for col in numeric_df.columns
        }
        if not os.path.exists(ranges_path):
            with open(ranges_path, 'w') as f:
                json.dump(data_ranges, f, indent=4)
    print(f"{len(data)} raw maize samples")
    
    # Do high-level cleaning of data
    data_cache_path = f'outputs/cache/data_preprocessed_{data_version}-clean.pkl'
    if os.path.exists(data_cache_path):
        print(f"Opening cached {data_cache_path}...")
        data = pd.read_pickle(data_cache_path)
    else:
        data = clean_raw_data(data, data_ranges)
        data.to_pickle(data_cache_path)

    # Transform the data to create useful representations
    data_cache_path = f'outputs/cache/data_preprocessed_{data_version}-transformed.pkl'
    if os.path.exists(data_cache_path):
        print(f"Opening cached {data_cache_path}...")
        data = pd.read_pickle(data_cache_path)
    else:
        data['fieldsize_value'] = data.harvestarea_ha
        data = transform_soil(data)
        data = transform_fert(data)
        data = transform_variety(data)
        data = transform_field_history(data)
        data = transform_other_management(data)
        data = add_investing_capacity_index(data)         
        # Assign to 0 all the stage-specific fertilizer amounts if the total is zero
        data.loc[data.fertP_total == 0, data.filter(like='fertP').columns] = 0
        data.loc[data.fertN_total == 0, data.filter(like='fertN').columns] = 0
        data['soil_group'] = np.where(data['soil_sandy'] == 1, 'Sandy', 'Non-Sandy')
        # data['outcome'] = data.maizeprodkg_ha
        data.to_pickle(data_cache_path)   
    print(f"Data preprocessing complete. {len(data)} samples remaining.")
    return data

def spatial_demean_data_krigging(data: pd.DataFrame, coords: pd.DataFrame, variables: list, maxlag: float = 50e3, n_lags: int = 50):
    """
    Applies spatial demeaning to specified variables in the dataframe using SpatialTransform.
    It returns the updated data and coords (dropping any rows that resulted in NaN).
    """
    print(f"Applying spatial demeaning to variables: {variables}...")
    
    # Construct geometry
    points_geometry = gpd.points_from_xy(coords['lon'], coords['lat'], crs='EPSG:32736')
    print(f"points_geometry length: {len(points_geometry)}")
    print(f"data length: {len(data)}")
    print(f"coords x sample: {coords['lon'].iloc[:5].tolist()}")
    print(f"coords y sample: {coords['lat'].iloc[:5].tolist()}")
    
    demeaned_series_list = []
    
    for var in variables:
        print(f"Fitting SpatialTransform for {var}...")
        print(f"var head: {data[var].head().tolist()}")
        print(f"var uniques: {data[var].nunique()}")
        spatial_transformer = SpatialTransform(
            points_geometry, 
            maxlag=maxlag, 
            esitmator='cressie', 
            model='matern', 
            n_lags=n_lags, 
            use_nugget=True
        )
        spatial_transformer.fit(data[var])
        demeaned_series = spatial_transformer.transform_demean()
        demeaned_series_list.append(demeaned_series)
        
    for var, demeaned_series in zip(variables, demeaned_series_list):
        data[var] = demeaned_series
        
    # Drop rows where demeaned values are NaN
    initial_len = len(data)
    data = data.dropna(subset=variables)
    print(f"Spatial demeaning completed. Dropped {initial_len - len(data)} rows with NaN demeaned values. {len(data)} rows remaining.")
    valid_coords = coords.loc[data.index]
    
    return data, valid_coords

def spatial_demean_data_idw(data: pd.DataFrame, coords: pd.DataFrame, variables: list, radius: float = 10000):
    """
    Applies spatial demeaning to specified variables in the dataframe using an Inverse Distance Weighted (IDW) average.
    It calculates the local mean of points within the specified radius (excluding the point itself) and subtracts it.
    """
    print(f"Applying IDW spatial demeaning to variables: {variables} with radius: {radius}m...")
    points = coords[['lon', 'lat']].values
    tree = cKDTree(points)
    
    demeaned_data = data.copy()
    
    for var in variables:
        values = data[var].values
        demeaned_values = np.zeros_like(values, dtype=float)
        
        for i, point in enumerate(points):
            indices = tree.query_ball_point(point, r=radius)
            # Remove self from neighbors
            indices = [idx for idx in indices if idx != i]
            
            if len(indices) > 0:
                neighbors_points = points[indices]
                dists = np.linalg.norm(neighbors_points - point, axis=1)
                dists[dists == 0] = 1e-8
                # dists = np.ones_like(dists)
                weights = 1.0 / dists
                weights /= weights.sum()
                
                local_mean = np.sum(weights * values[indices])
                demeaned_values[i] = values[i] - local_mean
            else:
                demeaned_values[i] = np.nan
                
        demeaned_data[var] = demeaned_values
        
    initial_len = len(demeaned_data)
    demeaned_data = demeaned_data.dropna(subset=variables)
    print(f"IDW spatial demeaning completed. Dropped {initial_len - len(demeaned_data)} rows with NaN demeaned values. {len(demeaned_data)} rows remaining.")
    
    valid_coords = coords.loc[demeaned_data.index]
    return demeaned_data, valid_coords

# Set the active spatial demeaning function here:
spatial_demean_data = spatial_demean_data_krigging
# spatial_demean_data = spatial_demean_data_idw
