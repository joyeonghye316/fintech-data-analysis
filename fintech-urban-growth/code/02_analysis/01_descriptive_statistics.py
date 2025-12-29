import pandas as pd
df = pd.read_csv('panel_final_Fintech_national.csv')

# 核心变量
df['Fintech'] = df['ln_fintech_lag1']
df['Fintech_sq'] = df['Fintech'] ** 2

# 描述性统计
desc = df[['ln_GRDP', 'Fintech', 'Fintech_sq', 'INT', 'GOV', 'OPE',
           'PC1', 'PC2', 'fintech_national', 'IS', 'ln_CAP']].describe()

# 美化
desc = desc.T
desc = desc[['mean', 'std', 'min', 'max', 'count']]
desc.columns = ['均值', '标准差', '最小值', '最大值', '观测数']
desc.round(4)

print(desc)


