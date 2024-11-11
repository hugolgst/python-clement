import pandas as pd
import json
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import plotly.express as px

def merge_csvs(country):
    CCI = pd.read_csv(f'data/CCI_{country}.csv')
    CCI = CCI[CCI['VALUE'] != "."] # some rows have dots
    CCI['VALUE'] = CCI['VALUE'].astype('float')
    CCI['DATE'] = pd.to_datetime(CCI['DATE']).dt.to_period('M')
    CCI = CCI.groupby('DATE').mean()

    Housing = pd.read_csv(f'data/Housing_{country}.csv')
    Housing['VALUE'] = Housing['VALUE'].astype('float')
    Housing['DATE'] = pd.to_datetime(Housing['DATE']).dt.to_period('M')
    Housing = Housing.groupby('DATE').mean()

    IPI = pd.read_csv(f'data/IndustrialPriceIndex_{country}.csv')
    IPI['VALUE'] = IPI['VALUE'].astype('float')
    IPI['DATE'] = pd.to_datetime(IPI['DATE']).dt.to_period('M')
    IPI = IPI.groupby('DATE').mean()

    PMI = pd.read_csv(f'data/PMI_{country}.csv')
    PMI['VALUE'] = PMI['VALUE'].astype('float')
    PMI['DATE'] = pd.to_datetime(PMI['DATE']).dt.to_period('M')
    PMI = PMI.groupby('DATE').mean()

    df = pd.concat(
        [ CCI, Housing, IPI, PMI ], 
        axis=1, 
        join="inner",
    )
    df.columns = [ 'CCI', 'Housing', 'IPI', 'PMI' ]

    return df

# Compute standard deviation given a column name
def compute_std_dev(df, column):
    X = np.arange(len(df)).reshape(-1, 1)
    Y = df[column]

    model = LinearRegression().fit(X, Y)
    std_dev = np.sqrt(mean_squared_error(Y, model.predict(X)))

    return float(std_dev)

def compute_composite(df):
    sum = 0

    # Iterate through all the indeces
    for col in df.columns:
        std_dev_inverse = 1/compute_std_dev(df, col)

        df[col] = df[col].apply(lambda x: float(x) * std_dev_inverse)
        sum += std_dev_inverse

    df['COMPOSITE'] = df.sum(axis=1)
    df['COMPOSITE'] = df['COMPOSITE'].apply(lambda x: x / sum)
    df['COMPOSITE'] = df['COMPOSITE'] - df.iloc[0]['COMPOSITE'] + 100

    return df


def plot_composite(country):
    df = merge_csvs(country)
    df = compute_composite(df)
    fig = px.line(df, x=df.index.astype('str'), y="COMPOSITE")
    fig.show()

plot_composite('china')
