import os
import re
import time
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.preprocessing import OneHotEncoder
from shapely.geometry import Point, Polygon, MultiPolygon

class KnowledgeBase:
    def __init__(self, load_knowledge_base=False):
        if load_knowledge_base:
            self.load_knowledge_base()
        else:
            pass

    def load_knowledge_base(self):
        load_stime = time.time()
        
        print("Loading public school coordinates as of SY 2023-2024.")
        self.public_coordinates = self.use_talino_public_school_coordinates(
            fpath='../datasets/public/SY 2023-2024 LIST OF SCHOOLS WITH LONGITUDE AND LATITUDE.xlsx',
        )
        print(f"Time elapsed for public school coordinates: {(time.time()-load_stime):.2f} seconds\n")
        load_stime = time.time()

        print("Loading private school coordinates as of 2024.")
        self.private_coordinates = self.set_private_coordinates(
            dirpath='../datasets/processed/raw_validation_sheets'
        )
        print(f"Time elapsed for private school coordinates: {(time.time()-load_stime):.2f} seconds\n")
        load_stime = time.time()

        print("Loading public and private school enrollment & SHS offerings for SY 2023-2024.")
        self.enrollment_info, self.enrollment = self.load_preprocess_enrollment(
            fpath='../datasets/public/Copy of SY 2023-2024 SCHOOL LEVEL DATA ON ENROLLMENT.csv',
            excel_sheet_name='DATABASE',
            split_header='modified'
        )
        
        self.shs_offerings = self.extract_shs_offerings(self.enrollment)
        print(f"Time elapsed for enrollment & SHS offerings: {(time.time()-load_stime):.2f} seconds\n")
        load_stime = time.time()

        print("Loading public & private school furnitures, namely seats, for SY 2023-2024.")
        self.public_seats_info, self.public_seats = self.load_preprocess_public_seats(
            '../datasets/public/SY 2023-2024 SEAT-LEARNER RATIO.xlsx',
            sheet_name='DATABASE',
        )
        
        self.private_seats_info, self.private_seats = self.load_preprocess_private_seats(
            fpath='../datasets/private/priv_classroom_furniture.xlsx'
        )
        print(f"Time elapsed for public & private seats: {(time.time()-load_stime):.2f} seconds\n")
        load_stime = time.time()

        print("Loading public school shifting schedule for SY 2023-2024.")
        self.public_shifting_info, _ = self.load_preprocess_enrollment(
            fpath='../datasets/public/SY 2022-2023 LIST OF PUBLIC SCHOOLS WITH SHIFTING SCHEDULE.xlsx',
            excel_sheet_name='DB',
            split_header='shifting'
        )
        print(f"Time elapsed for public shifting: {(time.time()-load_stime):.2f} seconds\n")
        load_stime = time.time()

        print("Loading private school ESC and SHS VP delivering schools as of 2024.")
        self.esc, self.shsvp = self.load_esc_amounts(
            fpath='../datasets/private/2024-2025 GASTPE Participating Schools Top-up Data.xlsx'
        )
        self.gastpe = self.load_gastpe_dataset(
            fpath='..datasets/private/ESC and SHSVP Tuition.xlsx'
        )
        print(f"Time elapsed for public shifting: {(time.time()-load_stime):.2f} seconds\n")
        load_stime = time.time()


    def use_talino_public_school_coordinates(self, fpath):
        df_excel = pd.read_excel(fpath, sheet_name='DB', header=5)
        
        df = df_excel.copy()
        df.columns = ['_'.join(str(col).strip().lower().split()) for col in df.columns]
        
        # drop nsbi_school_id
        df = df.drop(columns=['nsbi_school_id'])
        df['lis_school_id'] = df['lis_school_id'].astype('string')
        
        df = df.rename(
            columns={'lis_school_id':'school_id'}
        )

        return df
    
    def set_public_school_coordinates(self, fpath_nsbi, fpath_osm):
        df_nsbi = self.load_preprocess_nsbi(fpath_nsbi)
        gdf_osm = self.load_preprocess_osm(fpath_osm)

        df_merge = df_nsbi.merge(
            gdf_osm,
            left_on='lis school id',
            right_on='ref',
            how='left'
        )

        df_merge.index = [str(ix) for ix in df_merge.index.tolist()]
        
        return df_merge

    def load_preprocess_nsbi(self, fpath):
        df_nsbi = pd.read_csv(fpath)
        df_nsbi.columns = [col.lower() for col in df_nsbi.columns]
        df_nsbi.rename(
            columns={
                'longitude':'longitude_nsbi',
                'latitude':'latitude_nsbi',
            }, inplace=True
        )
        
        # Force data type of lis school id in preparation for merging
        df_nsbi['lis school id'] = df_nsbi['lis school id'].astype('string')
    
        return df_nsbi

    def load_preprocess_osm(self, fpath):
        tmp_osm = gpd.read_file(fpath)
        
        # Drop columns with mostly NaNs
        columns = tmp_osm.columns
        
        # Drop columns whose values are ALL NaN
        rel_cols = []
        for col in columns[3:]:
          is_na = tmp_osm[col].isna()
        
          if all(is_na):
            continue
          else:
            rel_cols.append(col)
        
        # Drop columns whoe values are > 50% NaN
        rel_cols_ = []
        for col in rel_cols:
          is_na = tmp_osm[col].isna()
          sum_na = is_na.sum()
          ratio_na = (sum_na / len(is_na)) * 100
        
          if ratio_na <= 50:
            rel_cols_.append(col)
        
        dtmp_osm = tmp_osm[rel_cols_].copy()
        dtmp_osm['ref'] = dtmp_osm['ref'].astype('string')

        # Drop rows whose column under 'ref' is "<NA>"
        mask_na = dtmp_osm['ref'].isna()
        dtmp_osm = dtmp_osm.loc[~mask_na]

        # Some public schools have coordinates for individual buildings in one campus.
        # We compile/dissolve them into one row to avoid duplicate rows later
        dtmp_osm = self._handle_duplicate_osm_coordinates(dtmp_osm)
        
        osMap_refs = dtmp_osm['ref'].unique().tolist()
        
        # Filter out rows whose operator:type is NOT public
        not_public = [
            'religious','private','private_sectarian',
            'religious;private','private_non_profit',
        ]
        mask = dtmp_osm['operator:type'].isin(not_public)
        dtmp_osm = dtmp_osm.loc[~mask]
        
        # Get the x and y from the geometry column, esp. when geometry type is Polygon, MultiPolygon
        dtmp_osm['longitude_osm'] = dtmp_osm['geometry'].apply(lambda x: self._get_coord(x))
        dtmp_osm['latitude_osm'] = dtmp_osm['geometry'].apply(lambda x: self._get_coord(x, longitude=False))
        
        rel_cols = ['ref','name','isced:level','longitude_osm','latitude_osm']
        gdf_osm_ = dtmp_osm[rel_cols].reset_index(drop=True).copy()
        
        # Force dtype of ref column in preparation for merging
        gdf_osm_['ref'] = gdf_osm_['ref'].astype('string')

        return gdf_osm_

    def _get_coord(self, x, longitude=True):
        coord = 0
        if type(x) in [Polygon, MultiPolygon]:
            coord = x.centroid.x if longitude else x.centroid.y
            
        elif type(x) == Point:
            coord = x.x if longitude else x.y
            
        return coord

    def _handle_duplicate_osm_coordinates(self, gdf_osm):
        gdf = gdf_osm.copy()

        df_dupes = gdf[gdf['ref'].duplicated()]
        dupe_idxs = df_dupes['ref'].unique()

        dissolves = []
        drop_idxs = []
        for d_id in dupe_idxs:
            tmp = gdf[gdf['ref'] == d_id]
            tmp_dissolve = tmp.dissolve()
            tmp_refs = tmp.index.tolist()
            
            dissolves.append(tmp_dissolve)
            drop_idxs.extend(tmp_refs)

        df_dis = pd.concat(dissolves)
        gdf_ = gdf.drop(index=drop_idxs)

        gdf_out = pd.concat([gdf_, df_dis], ignore_index=True)

        return gdf_out

    def set_private_coordinates(self, dirpath):
        files = os.listdir(dirpath)
        
        all_regions = []
        for path in files[1:]:
            fp = os.path.join(dirpath, path)
            sheets = pd.read_excel(fp, sheet_name=None)
        
            region_dfs = []
            for sheet in sheets.keys():
                tdf = sheets[sheet].copy()
                tdf = tdf.dropna(axis=0, how='all')
        
                # get index of row with "Region"
                idx_r = self._find_region_index(tdf)
                headers = tdf.iloc[idx_r, :].values.tolist()
        
                # slice dataframe
                tdf.columns = headers
                tdf = tdf.iloc[idx_r+1:,:]
        
                # drop columns whose headers will be NaNs and duplicated
                headers_ = [header.strip() for header in headers if type(header) == str]
                tdf = tdf[headers_]
                tdf = tdf.loc[:, ~tdf.columns.duplicated()]
        
                # drop rows whose values are all NaN
                tdf = tdf.dropna(axis=1, how='all')
        
                # label with sheet_name for tracking
                tdf['sheet_name'] = [sheet for _ in range(tdf.shape[0])]
                
                region_dfs.append(tdf)
        
            region_dfs = pd.concat(region_dfs)
        
            all_regions.append(region_dfs)

        regions_df = pd.concat(all_regions)
        
        replace_coc = {
            'ELEMENTARY':'Purely ES',
            'SHS':'Purely SHS',
            'K TO G6':'Purely ES',
            'KINDERGARTEN':'Kindergarten',
            'KINDEGARTEN':'Kindergarten',
            'K TO JHS':'ES and JHS (K to 10)',
            'K TO SHS':'All Offering (K to 12)',
            'JHS and SHS':'JHS with SHS',
            'Preschool and ES':'Kindergarten',
            'Elementary and JHS':'ES and JHS (K to 10)',
            'ES':'Purely ES',
            'K, ES, JHS & SHS':'All Offering (K to 12)',
            'K, Grade 1-3':'Purely ES',
            'K,Grades 1-10, Grades 11-12':'All Offering (K to 12)',
            'K, Grade 1 - 2':'Purely ES',
            'K, ES, JHS':'ES and JHS (K to 10)',
            'K, ES, JHS, SHS':'All Offering (K to 12)',
            'K,ES, JHS':'ES and JHS (K to 10)',
            'JHS, SHS':'JHS with SHS',
            'K, Gade 1-6, JHS':'ES and JHS (K to 10)',
            'K, Grade1-6,JHS':'ES and JHS (K to 10)',
            'Kinder, Grade 1 to 6':'Purely ES',
            'K, Grs. I - VI':'Purely ES',
            'Kinder, Grade 1-6':'Purely ES',
            'ES & JHS':'ES and JHS (K to 10)',
            'ES,JHS and SHS':'All Offering (K to 12)',
            'ES,JHS,and SHS':'All Offering (K to 12)',
            'K, Grade 1 - 6':'Purely ES',
            'All Offering':'All Offering (K to 12)',
        }
        regions_df['Modified COC'] = regions_df['Modified COC'].replace(replace_coc)
        
        mask_no_coords = regions_df['Latitude'].notna()
        regions_df = regions_df.loc[mask_no_coords]
        
        mask_no_name = regions_df['BEIS School ID'].notna()
        regions_df = regions_df.loc[mask_no_name].copy()

        n_cols = [str(col).strip().lower() for col in regions_df.columns]
        regions_df.columns = n_cols

        regions_df = regions_df.rename(
            columns={
                'beis school id':'school_id'
            }
        )
        regions_df['school_id'] = regions_df['school_id'].astype('string')
        
        return regions_df

    def _find_region_index(self, df, col_idx=0):
        """
        Recursively search for 'region' in each column of a dataframe.
        
        Parameters:
        df (pandas.DataFrame): The dataframe to search in
        col_idx (int): The current column index to check (default: 0)
        
        Returns:
        int or None: The row index where 'region' is found, or None if not found in any column
        """
        # Base case: If we've checked all columns, return None
        if col_idx >= len(df.columns):
            return None
        
        # Get the values of the current column
        col_values = df.iloc[:, col_idx].astype(str).str.lower().values
        
        # Check if 'region' is in the values
        for idx, value in enumerate(col_values):
            if 'region' in value:
                return idx
        
        # If 'region' is not found in this column, check the next column
        return self._find_region_index(df, col_idx + 1)

    def load_preprocess_enrollment(
        self,
        fpath,
        excel_sheet_name=None,
        split_header='modified',
        non_graded={'Elementary':True, 'Junior High School':True}
    ):
        if '.xlsx' in fpath:
            pdf = pd.read_excel(fpath, sheet_name=excel_sheet_name)

        elif '.csv' in fpath:
            pdf = pd.read_csv(fpath, low_memory=False)
        
        # Programmatically determine row in the dataframe where we will extract
        # our header values. This row usually starts with the string "Region"
        column_values = pdf.iloc[:,0].values.tolist()
        index = [i for i, val in enumerate(column_values) if re.search(r'^Region$', str(val).strip())][0]
        
        headers = pdf.iloc[index,:].values
        pdf.columns = [col.strip().lower() for col in headers]
        pdf = pdf.iloc[index+1:-1,:].reset_index(drop=True).copy()
        
        # Programmatically determine the column header that we will use to divide
        # the dataset into two, namely columns containing school information only and
        # columns that containt enrollment data only. This column header will usually
        # contain the word "offering" or "modified" esp for SY 2
        split_column = [col for col in pdf.columns if split_header in col.lower()][0]
        
        df_info = pdf.loc[:,:split_column].copy()
        
        enrollment_column = pdf.columns.tolist().index(split_column) + 1
        val_cols = pdf.iloc[:, enrollment_column:].columns
        
        # Also programmatically determine the column where school ids are found. Note
        # that there is a column "mother school id", which is NOT our target
        pattern = r'^(?!.*mother).*school[_ ]id$'
        id_column = [col for col in pdf.columns if re.search(pattern, col, re.IGNORECASE)][0]
        
        # For this specific dataset for the E4E release, the values in some columns have trailing and
        # leading whitespaces. We iterate over every column and remove them.
        for column in df_info.columns:
            column_values = df_info[column].values.tolist()
            new_values = [str(val).strip() for val in column_values]
            df_info[column] = new_values

        for column in pdf.columns:
            column_values = pdf[column].values.tolist()
            new_values = [str(val).strip() for val in column_values]
            pdf[column] = new_values
        
        # We then fix the data types, especially our columns containing numerical values for enrollment
        to_numeric = lambda val: np.nan if val == '-' else int(val.replace(',',''))
        for column in val_cols:
            # print(column)
            pdf[column] = pdf[column].apply(to_numeric)
        
        # We drop all columns that have the word "total"
        pattern = r'total|g1to6|kto6|jhs male|jhs female|g11acad male|g11acad female|g12acad male|g12acad female|kto12 male|kto12 female'
        total_cols = [col for col in val_cols if re.search(pattern, col)]
        ntotal_cols = list(set(val_cols).difference(set(total_cols)))
        # arrange trimmed columns following sequential order of original dataset
        ntotal_cols = [col for col in val_cols if col in ntotal_cols]
        pdf = pdf.drop(columns=total_cols)

        # We transform the dataframe from wide to long
        pdf_melt = pdf.melt(
            id_vars=id_column,
            value_vars=ntotal_cols,
            var_name='category',
            value_name='count_enrollment'
        )
        category_vals = pdf_melt['category'].unique()
        
        mask_0 = pdf_melt['count_enrollment'] >= 1
        pdf_melt = pdf_melt.loc[mask_0].copy()

        # We replace the original labels/headers of the columns containing enrollment data
        # with manually encoded alternative labels
        category_enc, lvl_educ = self._build_alternate_categories(
            category_vals,
            non_graded=non_graded
        )
        pdf_melt['category'] = pdf_melt['category'].replace(category_enc)

        # We create new columns that is split using arbitrarilly made delimeters
        pdf_melt['grade_level'] = pdf_melt['category'].str.extract(r'(.+)_')
        pdf_melt['sex'] = pdf_melt['category'].str.extract(r'_(\w+)')
        pdf_melt['shs_strand'] = pdf_melt['category'].str.extract(r'ale-(.+)$')

        # We fill the NaNs in this columns with "Not Applicable" so K to G10 can still
        # be included in a pivot table
        mask = pdf_melt['shs_strand'].isna()
        pdf_melt.loc[mask, 'shs_strand'] = 'Not Applicable'

        # We relabel the column containing the school ids
        pdf_melt.rename(columns={id_column:'school_id'}, inplace=True)
        df_info.rename(columns={id_column:'school_id'}, inplace=True)

        for df in [pdf_melt, df_info]:
            df['school_id'] = df['school_id'].astype('string')

        # Add Level of Education columns (i.e., Elem, JHS, SHS)
        for lvl, grades in lvl_educ.items():
            mask = pdf_melt['grade_level'].isin(grades)
            pdf_melt.loc[mask, 'level_of_education'] = lvl

        return df_info, pdf_melt

    def _build_alternate_categories(self, category_vals, non_graded):
        sexes = ['Male','Female']
        elem = ['Kindergarten'] + ['Grade ' + str(i) for i in range(1,7)]
        if non_graded.get('Elementary') == True:
            elem = elem + ['Non-Grade Elementary']
        elem_sex = [level+'_'+sex for level in elem for sex in sexes]
        
        jhs = ['Grade ' + str(i) for i in range(7,11)]
        if non_graded.get('Junior High School') == True:
            jhs = jhs + ['Non-Grade Junior High School']
        jhs_sex = [level+'_'+sex for level in jhs for sex in sexes]
        
        shs_strands = ['ABM','HUMSS','STEM','GAS','PBM','TVL','SPORTS','ARTS & DESIGN']
        shs = ['Grade ' + str(i) for i in range(11,13)]
        shs_sex = []
        for level in shs:
            for strand in shs_strands:
                for sex in sexes:
                    label = level+'_'+sex+'-'+strand
                    shs_sex.append(label)
                    
        alt_category = elem_sex + jhs_sex + shs_sex
        level_of_educ = {
            'Elementary': elem,
            'Junior High School': jhs,
            'Senior High School': shs,
        }
        
        category_enc = {cat: alt for cat, alt in zip(category_vals, alt_category)}
        
        return category_enc, level_of_educ

    def extract_shs_offerings(self, enrollment):
        mask_shs = enrollment['level_of_education'] == 'Senior High School'
        shs_enr = enrollment.loc[mask_shs].copy()
        
        shs_offerings = shs_enr.pivot_table(
            index='school_id',
            columns='shs_strand',
            values='count_enrollment',
            aggfunc='sum'
        )
        
        for col in shs_offerings.columns:
            shs_offerings[col] = shs_offerings[col].apply(lambda x: 1 if x >= 1 else 0)

        return shs_offerings

    def load_preprocess_public_seats(self, fpath, sheet_name, split_header='modified'):
        pdf = pd.read_excel(fpath, sheet_name=sheet_name)

        column_values = pdf.iloc[:,0].values.tolist()
        nrow = column_values.index('Region')
        
        headers = pdf.iloc[nrow,:].values
        pdf.columns = [str(col).lower() for col in headers]
        pdf = pdf.iloc[nrow+1:-1,:].reset_index(drop=True).copy()
        
        columns = pdf.columns.tolist()
        coc_column = [col for col in columns if split_header in col.strip().lower()][0]
        coc_idx = [i for i, col in enumerate(columns) if split_header in col.strip().lower()][0]
        df_info = pdf.loc[:,:coc_column].copy()
        
        pattern = r'^(?!.*mother).*school[_ ]id$'
        id_column = [col for col in columns if re.search(pattern, col, re.IGNORECASE)][0]
        
        pdf_trim = pd.concat(
            [
                pdf.iloc[:, :coc_idx+1],
                pdf.iloc[:, coc_idx+1+5:-4]
            ], axis=1
        )
        value_columns = ['es','jhs','shs']
        id_cols = id_column
        
        pdf_melt = pdf_trim.melt(
            id_vars=id_column,
            value_vars=value_columns,
            var_name='category',
            value_name='count_seats',
        )
        mask_0 = pdf_melt['count_seats'] >= 1
        pdf_melt = pdf_melt.loc[mask_0].copy()

        # Unlike the Excel file for enrollment, the columns containing our target values
        # are only "es", "jhs", and "shs". We can be straightforward in relabling them.
        replace_dict = {
            'es':'Elementary',
            'jhs':'Junior High School',
            'shs':'Senior High School',
        }
        
        pdf_melt['category'] = pdf_melt['category'].replace(replace_dict)

        # There are seats with decimal values that we will round down for this project
        pdf_melt['count_seats'] = np.floor(pdf_melt['count_seats'])

        # We relabel the column containing the school ids
        pdf_melt.rename(
            columns={id_column:'school id', 'category':'level_of_education'},
            inplace=True
        )
        df_info.rename(
            columns={id_column:'school id', 'category':'level_of_education'},
            inplace=True
        )

        for df in [pdf_melt, df_info]:
            df['school id'] = df['school id'].astype('string')

        return df_info, pdf_melt

    def load_preprocess_private_seats(self, fpath, sheet_name='priv_classroom_furniture'):
        tdf_excel = pd.read_excel(fpath, sheet_name=sheet_name)
        df_fr = tdf_excel.copy()

        # The preprocessing steps below are similar to public enrollment with some changes
        col_vals = df_fr.iloc[:, 1].values
        index = [i for i, val in enumerate(col_vals) if val == 'Region'][0]
        headers = df_fr.iloc[index,:].values
        
        df_fr.columns = [h.lower() for h in headers]
        df_fr = df_fr.iloc[index+1:,1:].reset_index(drop=True).copy()
        
        pattern = r'^(?!.*mother).*school[_ ]id$'
        id_column = [col for col in df_fr.columns if re.search(pattern, col, re.IGNORECASE)][0]
        
        split_col = 'school name'
        split_ix = [ix for ix, col in enumerate(df_fr.columns) if col == split_col][0]
        val_vars = df_fr.iloc[:,split_ix+1:].columns

        fr_info = df_fr.iloc[:, :split_ix+1].copy()
        
        fr_melt = df_fr.melt(
            id_vars=id_column,
            value_vars=val_vars,
            var_name='furniture_category',
            value_name='furniture_count'
        )
        
        # The raw file has desks and other furnitures so we do NOT include them.
        # We only include headers that has the word "chair", namely "sets of chairs and tables"
        # and "arm chairs"
        pattern = r'chairs'
        mask_seats = (
            (fr_melt['furniture_category'].str.contains(pattern, regex=True, flags=re.IGNORECASE))
            & (fr_melt['furniture_count'] >= 1)
        )
        
        fr_seats = fr_melt.loc[mask_seats].reset_index(drop=True).copy()
        
        # Extract grade level from furniture category
        patterns = [r'kinder', r'gr1to6', r'jhs', r'shs']
        levels = ['Kindergarten','Elementary','Junior High School','Senior High School']
        for pattern, level in zip(patterns, levels):
            mask = fr_seats['furniture_category'].str.contains(pattern, regex=True, flags=re.IGNORECASE)
            fr_seats.loc[mask, 'level_of_education'] = [level for _ in range(fr_seats.loc[mask].shape[0])]
        
        fr_seats['school id'] = fr_seats['school id'].astype('string')
       
        return fr_info, fr_seats

    def load_preprocess_public_shifting(self, fpath, excel_sheet_name='DB', split_header='shifting'):
        shifting_info, shifting_enr = self.load_preprocess_enrollment(
            fpath=fpath,
            excel_sheet_name=excel_sheet_name,
            split_header=split_header
        )
        return shifting_info

    def load_esc_amounts(self, fpath):
        df_excel = pd.read_excel(fpath, sheet_name=None)
        esc = df_excel['ESC'].copy()
        shsvp = df_excel['SHS VP'].copy()

        id_cols = ['ESC School Id', 'DepEd School ID']
        dfs = [esc, shsvp]
        for i, df in enumerate(dfs):
            df[id_cols[i]] = df[id_cols[i]].astype('string')
            df.columns = ['_'.join(col.strip().lower().split()) for col in df.columns]

        esc = esc.rename(
            columns={
                'esc_school_id':'esc_school_id_ref',
                'average_jhs_school_fees':'avg_jhs_school_fees',
                'top-up':'esc_top_up',
            }
        )
        
        shsvp = shsvp.rename(
            columns={
                'deped_school_id':'school_id',
                'average_shs_school_fees':'avg_shs_school_fees',
                'average_of_category_a_top-up':'avg_catA_top_up',
                'average_of_category_b_top-up':'avg_catB_top_up',
                'average_of_category_c_top-up':'avg_catC_top_up',
                'average_of_category_d_top-up':'avg_catD_top_up',
                'average_of_category_e_top-up':'avg_catE_top_up',
                'average_of_category_f_top-up':'avg_catF_top_up',
                'average_of_average_top-up':'avg_top_up',
            }
        )

        return esc, shsvp

    def load_gastpe_dataset(self, fpath, sheet_name='Tuition'):
        # This dataframe pertains to the Excel file that contains information on the ESC Amount
        # designated to a ESC participation school
        excel_a = self.esc.copy()
    
        # This dataset, on the hand, matched BEIS School ID with ESC School ID from PEAC's IS
        fpath = '../datasets/private/ESC and SHSVP Tuition.xlsx'
        excel_b = pd.read_excel(fpath, sheet_name=sheet_name)
    
        df_gastpe = excel_b.copy()
        old_gastpe = excel_a.copy()
        
        # Lowercase all column names
        df_gastpe.columns = [str(col).lower() for col in df_gastpe.columns]
        # Change spaces to underscore
        df_gastpe.columns = ['_'.join(col.split(' ')) for col in df_gastpe.columns]
    
        df_gastpe['deped_school_id'] = df_gastpe['deped_school_id'].astype('string')
        df_gastpe['program'] = df_gastpe['program'].astype('string')
        df_gastpe['suc/luc'] = [1 if str(val) == 'SUC/LUC' else 0 for val in df_gastpe['suc/luc']]
        
        program_esc = [1 if str(val).lower() in ['esc','both'] else 0 for val in df_gastpe['program'].values]
        program_shsvp = [1 if str(val).lower() in ['shs','both'] else 0 for val in df_gastpe['program'].values]
        
        df_gastpe['esc_participating'] = program_esc
        df_gastpe['shsvp_participating'] = program_shsvp
        
        # Rearrange columns
        cols = df_gastpe.columns.tolist()
        order_cols = cols[-2:] + cols[:-2]
        df_gastpe = df_gastpe[order_cols]
        
        # JOIN other dataset to get ESC amount of ESC participating schools
        df_gastpe['esc_school_id'] = [str(int(val)) if str(val) not in ['NaN','nan'] else 'None' for val in df_gastpe['esc_school_id'].values]
        
        old_gastpe = old_gastpe[['esc_school_id_ref','esc_amount']]
        
        df_gastpe = df_gastpe.merge(
            old_gastpe,
            left_on='esc_school_id', right_on='esc_school_id_ref',
            how='left'
        )
        
        # Drop irrelevant columns
        ir_cols = ['billed_in_sy_2024-2025_(esc)','program','esc_school_id_ref','region_shs']
        rename_cols = {
            'deped_school_id':'school_id'
        }
        df_gastpe = (
            df_gastpe
            .drop(columns=ir_cols)
            .rename(columns=rename_cols)
            .set_index('school_id')
        )
    
        return df_gastpe

    def compile_public_datasets(self):
        # Enrollment of public and private
        enr_info = self.enrollment_info.copy()
        enr = self.enrollment.copy()
        pub_coords = self.public_coordinates.copy()
        pub_seats_info, pub_seats = self.public_seats_info.copy(), self.public_seats.copy()
        public_shifting = self.public_shifting_info.copy()
        shs_offerings = self.shs_offerings.copy()
        
        # Public coordinates
        rel_cols = [
            'region','division','school_id','school_name',
            'province','municipality','longitude', 'latitude',
        ]
        pub_cds = pub_coords[rel_cols]
        pub_cds = pub_cds.set_index('school_id')
        
        # Public seats
        pub_seats_pvt = pub_seats.pivot_table(
            index='school id',
            columns='level_of_education',
            values='count_seats',
            aggfunc='sum'
        )
        new_cols = ['seats_es','seats_jhs','seats_shs']
        pub_seats_pvt.columns = new_cols
        
        # Public shifting
        rel_cols = ['school_id','shifting schedule being implemented']
        pub_shifting = public_shifting[rel_cols].set_index('school_id')
        pub_shifting.columns = ['shifting_schedule']
        
        # Add CoC from enrollment for sanity check
        enr_info_ = enr_info[['school_id','modified coc']].set_index('school_id')
        
        # Actual enrollment figures by level of education
        pub_info = enr_info[enr_info['sector'] == 'Public']
        pub_ids = pub_info['school_id'].unique()
        pub_enr = enr[enr['school_id'].isin(pub_ids)]
        
        pub_enr_pvt = pub_enr.pivot_table(
            index='school_id',
            columns='level_of_education',
            values='count_enrollment',
            aggfunc='sum'
        )
        pub_enr_pvt.columns = ['enrollment_es','enrollment_jhs','enrollment_shs']

        shs_offerings.columns = ["shs_"+str(col) for col in shs_offerings.columns]

        # There are SUC/LUCs included in the gastpe database, namely institutions that
        # are SHS VP participating
        gastpe = self.gastpe.copy()
        
        mask = gastpe['suc/luc'] == 1
        gastpe = gastpe.loc[mask]
        
        rel_cols = [
            'shsvp_participating',
            'shsvp_(tuition)', 'shsvp_(other)',
            'shsvp_(misc)', 'shsvp_(total)',

        ]
        gastpe = gastpe[rel_cols]
        
        compiled_public = pub_cds.join(
            [
                pub_shifting,
                enr_info_,
                pub_seats_pvt,
                pub_enr_pvt,
                shs_offerings,
                # gastpe, # SUC/LUCs participating in SHS VP have no coordinates
            ],
            how='left'
        )
        # BANDAID solution to "sector"
        compiled_public['sector'] = ['Public' for _ in range(compiled_public.shape[0])]
        
        return compiled_public

    def compile_private_datasets(self):
        # Enrollment of public and private
        enr_info = self.enrollment_info.copy()
        enr = self.enrollment.copy()
        shs_offerings = self.shs_offerings.copy()

        rel_cols = ['region','division','school_id','school name','province','municipality','barangay','school type','modified coc']
        priv_info = enr_info[enr_info['sector'] == 'Private'].copy()
        priv_info = priv_info[rel_cols]
        
        priv_ids = priv_info['school_id'].unique()
        priv_enr = enr[enr['school_id'].isin(priv_ids)]
        
        # Enrollment
        priv_enr_pvt = priv_enr.pivot_table(
            index='school_id',
            columns='level_of_education',
            values='count_enrollment',
            aggfunc='sum'
        )
        priv_enr_pvt.columns = ['enrollment_es','enrollment_jhs','enrollment_shs']
        
        private_seats_info, private_seats = self.private_seats_info.copy(), self.private_seats.copy()
        
        # Seats
        priv_seats_pvt = private_seats.pivot_table(
            index='school id',
            columns='level_of_education',
            values='furniture_count',
            aggfunc='sum'
        )
        priv_seats_pvt.columns = ['seats_kinder','seats_es','seats_jhs','seats_shs']
        
        # GASTPE data
        gastpe = self.gastpe.copy()
        rel_cols = [
            'esc_participating', 'shsvp_participating',
            'esc_school_id', 'esc_(tuition)', 'esc_(other)',
            'esc_(misc)', 'esc_(total)', 'shsvp_(tuition)', 
            'shsvp_(other)', 'shsvp_(misc)', 'shsvp_(total)',
            'esc_amount'
        ]
        gastpe = gastpe[rel_cols]

        priv_coords = self.private_coordinates
        priv_cds = priv_coords.copy()
        
        df_dupes = priv_cds[priv_cds['school_id'].duplicated()]
        
        dupe_ids = df_dupes['school_id'].unique()
        first_dupes = []
        for d_id in dupe_ids:
            tmp = priv_cds[priv_cds['school_id'] == d_id]
            tmp_idxs = tmp.index
        
            for ix in tmp_idxs[1:]:
                priv_cds.drop(index=ix, inplace=True)
        
        rel_cols = ['school_id','latitude','longitude']
        priv_cds = priv_cds[rel_cols]

        shs_offerings.columns = ["shs_"+str(col) for col in shs_offerings.columns]
        
        compiled_private = priv_info.set_index('school_id').join(
            [
                priv_enr_pvt,
                priv_seats_pvt,
                gastpe,
                priv_cds.set_index('school_id'),
                shs_offerings,
            ], how='left'
        )
        compiled_private = compiled_private.rename(
            columns={
                'school name':'school_name'
            }
        )
        # BANDAID solution to "sector"
        compiled_private['sector'] = ['Private' for _ in range(compiled_private.shape[0])]

        return compiled_private