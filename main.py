import pandas as pd
import json
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import plotly.express as px

CCI = pd.read_csv('data/CCI.csv')
CCI = CCI[CCI['VALUE'] != "."] # some rows have dots
CCI['VALUE'] = CCI['VALUE'].astype('float')
CCI['DATE'] = pd.to_datetime(CCI['DATE']).dt.to_period('M')
CCI = CCI.groupby('DATE').mean()

CPI = pd.read_csv('data/CPI.csv')
CPI['VALUE'] = CPI['VALUE'].astype('float')
CPI['DATE'] = pd.to_datetime(CPI['DATE']).dt.to_period('M')
CPI = CPI.groupby('DATE').mean()

Housing = pd.read_csv('data/Housing.csv')
Housing['VALUE'] = Housing['VALUE'].astype('float')
Housing['DATE'] = pd.to_datetime(Housing['DATE']).dt.to_period('M')
Housing = Housing.groupby('DATE').mean()

IPI = pd.read_csv('data/IndustrialPriceIndex.csv')
IPI['VALUE'] = IPI['VALUE'].astype('float')
IPI['DATE'] = pd.to_datetime(IPI['DATE']).dt.to_period('M')
IPI = IPI.groupby('DATE').mean()

PMI = pd.read_csv('data/PMI.csv')
PMI['VALUE'] = PMI['VALUE'].astype('float')
PMI['DATE'] = pd.to_datetime(PMI['DATE']).dt.to_period('M')
PMI = PMI.groupby('DATE').mean()

PPI = pd.read_csv('data/PPI.csv')
PPI['VALUE'] = PPI['VALUE'].astype('float')
PPI['DATE'] = pd.to_datetime(PPI['DATE']).dt.to_period('M')
PPI = PPI.groupby('DATE').mean()

df = pd.concat(
    [ CCI, CPI, Housing, IPI, PMI, PPI ], 
    axis=1, 
    join="inner",
)
df.columns = [ 'CCI', 'CPI', 'Housing', 'IPI', 'PMI', 'PPI' ]

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


df = compute_composite(df)
print(df)
fig = px.line(df, x=df.index.astype('str'), y="COMPOSITE")
fig.show()
