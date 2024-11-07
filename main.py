import pandas as pd
import json

CCI = pd.read_csv('data/CCI.csv')
CCI['DATE'] = pd.to_datetime(CCI['DATE'])

CPI = pd.read_csv('data/CPI.csv')
CPI['DATE'] = pd.to_datetime(CPI['DATE'])

Housing = pd.read_csv('data/Housing.csv')
Housing['DATE'] = pd.to_datetime(Housing['DATE'])

IPI = pd.read_csv('data/IndustrialPriceIndex.csv')
IPI['DATE'] = pd.to_datetime(IPI['DATE'])

PMI = pd.read_csv('data/PMI.csv')
PMI['DATE'] = pd.to_datetime(PMI['DATE'])

PPI = pd.read_csv('data/PPI.csv')
PPI['DATE'] = pd.to_datetime(PPI['DATE'])

print(PMI)
