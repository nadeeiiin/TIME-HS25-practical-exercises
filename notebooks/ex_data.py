import gc
from typing import List, Iterable, Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


def get_sea_surface_temp() -> pd.DataFrame:
    
    # Load the data
    dataset = sm.datasets.elnino.load_pandas()
    data = dataset['data']
    
    # Convert wide to long
    months = [
        'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
        'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'
    ]
    df = data.melt(id_vars=['YEAR'], value_vars=months, var_name='MONTH')
    
    # Combine year and month into a single 'DATE' column
    df['YEAR'] = df['YEAR'].astype(int)
    df['MONTH'] = df['MONTH'].apply(lambda x: months.index(x) + 1)
    df['DATE'] = df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str)
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m')
    
    # Sort the data
    df = df.sort_values('DATE')
    

    return df



