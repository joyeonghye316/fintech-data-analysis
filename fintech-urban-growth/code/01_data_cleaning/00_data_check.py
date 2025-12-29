import pandas as pd
import numpy as np

# read data
df = pd.read_csv('panel_final_Fintech_national.csv')

# print information 
print("=" * 80)
print("数据基本信息")
print("=" * 80)
print(f"\n数据维度: {df.shape}")
print(f"\n列名: {df.columns.tolist()}")
print(f"\n前5行数据:")
print(df.head())
print(f"\n数据类型:")
print(df.dtypes)
print(f"\n缺失值:")
print(df.isnull().sum())
print(f"\n描述性统计:")
print(df.describe())


