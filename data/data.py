import gc
from typing import List, Iterable, Dict, Tuple

import numpy as np
import pandas as pd

import eurostat
import yfinance as yf
from statsmodels.datasets import co2
from scipy.datasets import electrocardiogram


def get_electrocardiogram() -> pd.DataFrame:
    ecg = electrocardiogram()
    fs = 360
    return pd.DataFrame({
        'time': np.arange(ecg.size) / fs,
        'ecg': ecg
    })


def get_mauna_loa_co2() -> pd.DataFrame:
    return co2.load().data


def get_apple_5y() -> pd.DataFrame:
    df = yf.download('AAPL', period='5y')
    return df


def get_france_death_rate_20y() -> pd.DataFrame:
    country, nyears = 'FR', 20
    df = (
        eurostat.get_data_df('demo_mmonth')
        .rename(columns={r'geo\TIME_PERIOD': 'country'})
        .drop(columns=['freq', 'unit'])
    )
    df = df[~df['month'].isin(['TOTAL', 'UNK'])]
    month_mapping = {
        'M01': 'jan', 'M02': 'feb', 'M03': 'mar', 'M04': 'apr',
        'M05': 'may', 'M06': 'jun', 'M07': 'jul', 'M08': 'aug',
        'M09': 'sep', 'M10': 'oct', 'M11': 'nov', 'M12': 'dec'
    }
    df['month'] = df['month'].map(month_mapping)
    df = df.melt(id_vars=['month', 'country'], var_name='year', value_name='value')
    df = df.dropna()
    df['year'] = df['year'].astype(int)
    df['month'] = pd.Categorical(df['month'], categories=list(month_mapping.values()), ordered=True)
    df = df.sort_values(by=['year', 'month'])
    df = df[df['country'] == country]
    df = df[df['year'] > df['year'].max() - nyears]
    df['time'] = pd.date_range(start=f"{df['year'].min()}-01-01", periods=len(df), freq='ME')
    assert all(df['time'].dt.strftime('%b').str.lower() == df['month']), "Mismatch between 'time' and 'month' columns"
    df = df.reset_index(drop=True)
    return df


def get_switzerland_temperature():
    country, nyears = 'Switzerland', 20
    df = pd.read_csv('../data/GlobalLandTemperaturesByCountry.csv')
    df = df[df['Country'] == country]
    df['dt'] = pd.to_datetime(df['dt'])
    df = df.set_index('dt').resample('ME').agg({
        'AverageTemperature': 'mean',
        'AverageTemperatureUncertainty': 'mean',
        'Country': 'first',
    }).reset_index()
    cutoff_date = df['dt'].max() - pd.DateOffset(years=nyears)
    df = df[(df['dt'] >= cutoff_date) & (df['dt'].dt.year < 2013)]
    df = df.reset_index(drop=True)
    return df


def get_victoria_electricity_demand():
    df = pd.read_csv('../data/victoria_electricity_demand.csv')
    df['Date'] = df['Date'].apply(
        lambda x: pd.Timestamp('1899-12-30') + pd.Timedelta(x, unit='days')
    )
    df['dt'] = df['Date'] + pd.to_timedelta((df['Period'] - 1) * 30, unit='m')
    df = df[['dt', 'OperationalLessIndustrial']]
    df = df.set_index('dt').resample('ME').sum()
    return df.iloc[:-1]        # last value is zero


def get_sensor_data():
    # inspired from https://www.kaggle.com/code/dbogdanov/load-resample-data-and-basic-eda
    data_path = "../data/condition-monitoring-of-hydraulic-systems/"
    number_of_profiles = 2205
    sample_rate = 6000
    sensor_files_config = [
        {"name": "CE", "upsample_coeff": 100},
        {"name": "CP", "upsample_coeff": 100},
        {"name": "EPS1", "upsample_coeff": 1},
        {"name": "FS1", "upsample_coeff": 10},
        {"name": "FS2", "upsample_coeff": 10},
        {"name": "PS1", "upsample_coeff": 1},
        {"name": "PS2", "upsample_coeff": 1},
        {"name": "PS3", "upsample_coeff": 1},
        {"name": "PS4", "upsample_coeff": 1},
        {"name": "PS5", "upsample_coeff": 1},
        {"name": "PS6", "upsample_coeff": 1},
        {"name": "SE", "upsample_coeff": 100},
        {"name": "TS1", "upsample_coeff": 100},
        {"name": "TS2", "upsample_coeff": 100},
        {"name": "TS3", "upsample_coeff": 100},
        {"name": "TS4", "upsample_coeff": 100},
        {"name": "VS1", "upsample_coeff": 100},
    ]

    def get_files_with_resample(config: List[Dict]) -> Iterable[np.ndarray]:
        for file in config:
            data = np.genfromtxt(data_path + file["name"] + ".txt", dtype=float, delimiter='\t')
            yield np.repeat(data, file["upsample_coeff"], axis=1).flatten()

    def load_feature_dataframe(config: List[Dict]) -> pd.DataFrame:
        columns = [file["name"] for file in config]
        data = np.stack(list(get_files_with_resample(config)), axis=-1)
        data_df = pd.DataFrame(data, columns=columns)
        prodile_ids = np.repeat(range(1, number_of_profiles + 1), sample_rate)
        prodile_ids_df = pd.DataFrame(prodile_ids, columns=["profile_id"])
        return pd.concat([prodile_ids_df, data_df], axis=1, sort=False)

    feature_df = load_feature_dataframe(sensor_files_config)
    _ = gc.collect()
    return feature_df


def get_random_walk(seed: int, npoints: int = 100, variance: float = 1, drift: np.ndarray = None, seasonal: np.ndarray = None) -> np.ndarray:
    np.random.seed(seed)
    w = np.random.normal(loc=0, scale=np.sqrt(variance), size=npoints)
    if drift is not None:
        w += drift
    if seasonal is not None:
        w += seasonal
    return np.cumsum(w)


def get_periodic_process(seed: int, npoints: int = 100, sd: float = 1., period: float = 12.) -> Tuple[np.ndarray, str]:
    np.random.seed(seed)
    t = np.arange(npoints)
    c_t = np.cos(2 * np.pi * t / period)
    s_t = np.sin(2 * np.pi * t / period)
    a = np.random.normal(loc=0, scale=sd)
    b = np.random.normal(loc=0, scale=sd)
    name = f'Periodic process ($a={a:.2f}, b={b:.2f}, \omega=1/{period}$)'
    x_t = a * c_t + b * s_t
    return x_t, name
