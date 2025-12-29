# -*- coding: utf-8 -*-
"""
表3：稳健性检验（6列）
"""

import pandas as pd
from linearmodels.panel import PanelOLS, compare

# ================== 1. 读取数据 ==================
df = pd.read_csv('panel_final_Fintech_national.csv')   # 170×15的文件

# ================== 2. 核心变量提取 ==================
df['Fintech']       = df['ln_fintech_lag1']
df['Fintech_sq']    = df['Fintech'] ** 2
df['Fintech_national'] = df.groupby('year')['Fintech'].transform('mean')

# 是否首尔（稳健性用）
df['is_seoul'] = (df['city'] == '서울특별시').astype(int)

# 替换因变量需要的新变量
df['GRDP_growth']   = df.groupby('city')['ln_GRDP'].transform(lambda x: x.diff(1))

df = df.set_index(['city', 'year'])

# ================== 3. 稳健性检验6列 ==================
results = {}

results['(1) 主规格'] = PanelOLS.from_formula(
    'ln_GRDP ~ Fintech + Fintech_sq + INT + GOV + OPE + PC1 + PC2 + Fintech_national + EntityEffects', df
).fit(cov_type='clustered', cluster_entity=True)

results['(2) 排除首尔'] = PanelOLS.from_formula(
    'ln_GRDP ~ Fintech + Fintech_sq + INT + GOV + OPE + PC1 + PC2 + Fintech_national + EntityEffects',
    data=df[df.index.get_level_values('city') != '서울특별시']
).fit(cov_type='clustered', cluster_entity=True)

if 'ln_GRDP_total' in df.columns and df['ln_GRDP_total'].notna().any():
    results['(3) GRDP总量'] = PanelOLS.from_formula(
        'ln_GRDP_total ~ Fintech + Fintech_sq + INT + GOV + OPE + PC1 + PC2 + Fintech_national + EntityEffects', df
    ).fit(cov_type='clustered', cluster_entity=True)

# 重点：剔除COVID年份
results['(4) 剔除COVID'] = PanelOLS.from_formula(
    'ln_GRDP ~ Fintech + Fintech_sq + INT + GOV + OPE + PC1 + PC2 + Fintech_national + EntityEffects',
    data=df[~df.index.get_level_values('year').isin([2020, 2021])]
).fit(cov_type='clustered', cluster_entity=True)


print("\n=== 表3 稳健性检验 ===\n")
print(compare(results, stars=True, precision='std-errors'))


