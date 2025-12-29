# -*- coding: utf-8 -*-
import pandas as pd
from linearmodels.panel import PanelOLS, compare

df = pd.read_csv('panel_final_Fintech_national.csv')

# 直接读取表头
df['Fintech']       = df['ln_fintech_lag1']
df['Fintech_lag2']  = df['ln_fintech_lag2']
df['Fintech_lead1'] = df['ln_patent_lead1']        # 超前1期

# 用 lead1 再 shift(-1) 得到 lead2
df = df.sort_values(['city', 'year'])
df['Fintech_lead2'] = df.groupby('city')['Fintech_lead1'].shift(-1)   # ← 只掉2023年

# 平方项 + 全国控制
df['Fintech_sq']       = df['Fintech']**2
df['Fintech_lag2_sq']  = df['Fintech_lag2']**2
df['Fintech_lead1_sq'] = df['Fintech_lead1']**2
df['Fintech_lead2_sq'] = df['Fintech_lead2']**2

df['nat_main']  = df.groupby('year')['Fintech'].transform('mean')
df['nat_lead2'] = df.groupby('year')['Fintech_lead2'].transform('mean')

df = df.set_index(['city', 'year'])

# 最终五列表头（超前2期样本153）
results = {}

results['(1) 基准模型'] = PanelOLS.from_formula(
    'ln_GRDP ~ Fintech + Fintech_sq + INT + GOV + OPE + PC1 + PC2 + EntityEffects', df
).fit(cov_type='clustered', cluster_entity=True)

results['(2) 主规格\n+全国控制'] = PanelOLS.from_formula(
    'ln_GRDP ~ Fintech + Fintech_sq + INT + GOV + OPE + PC1 + PC2 + nat_main + EntityEffects', df
).fit(cov_type='clustered', cluster_entity=True)

results['(3) 滞后2期'] = PanelOLS.from_formula(
    'ln_GRDP ~ Fintech_lag2 + Fintech_lag2_sq + INT + GOV + OPE + PC1 + PC2 + nat_main + EntityEffects', df
).fit(cov_type='clustered', cluster_entity=True)

results['(4) 超前1期'] = PanelOLS.from_formula(
    'ln_GRDP ~ Fintech_lead1 + Fintech_lead1_sq + INT + GOV + OPE + PC1 + PC2 + nat_main + EntityEffects', df
).fit(cov_type='clustered', cluster_entity=True)

# 超前2期：153样本
results['(5) 超前2期'] = PanelOLS.from_formula(
    'ln_GRDP ~ Fintech_lead2 + Fintech_lead2_sq + INT + GOV + OPE + PC1 + PC2 + nat_lead2 + EntityEffects',
    data=df.dropna(subset=['Fintech_lead2'])
).fit(cov_type='clustered', cluster_entity=True)

print("\n=== 金融科技对城市经济增长的非线性影响：因果识别完整版（超前2期153样本）===\n")
print(compare(results, stars=True, precision='std-errors'))

print("\n" + "="*80)
print("超前2期详细结果（样本量=153）：")
print(results['(5) 超前2期'].summary)


res = results['(2) 主规格\n+全国控制']
cov_matrix = res.cov
cov_b1b2 = cov_matrix.loc['Fintech', 'Fintech_sq']
print("Cov(Fintech, Fintech_sq) =", cov_b1b2)

var_b1 = cov_matrix.loc['Fintech', 'Fintech']
var_b2 = cov_matrix.loc['Fintech_sq', 'Fintech_sq']

print("Var(Fintech) =", var_b1)
print("Var(Fintech_sq) =", var_b2)
print("Cov(Fintech, Fintech_sq) =", cov_b1b2)