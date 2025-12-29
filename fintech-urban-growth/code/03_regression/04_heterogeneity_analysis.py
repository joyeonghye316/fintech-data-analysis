
"""
表5：异质性分析 —— 四组对比
"""

import pandas as pd
from linearmodels.panel import PanelOLS, compare

# ================== 1. 读取CSV 文件 ==================
df = pd.read_csv('panel_final_Fintech_national.csv')

# ================== 2. 核心变量 ==================
df['Fintech'] = df['ln_fintech_lag1']
df['Fintech_sq'] = df['Fintech'] ** 2
df['Fintech_national'] = df.groupby('year')['Fintech'].transform('mean')

# 首都圈定义
capital_cities = ['서울특별시', '경기도', '인천광역시']
df['is_capital'] = df['city'].isin(capital_cities).astype(int)

# 设置面板索引
df = df.set_index(['city', 'year'])

# ================== 3. 异质性分析三列 ==================
results = {}

# (1) 首都圈（3市 × 10年 = 30个观测）
results['(1) 首都圈'] = PanelOLS.from_formula(
    'ln_GRDP ~ Fintech + Fintech_sq + INT + GOV + OPE + PC1 + PC2 + Fintech_national + EntityEffects',
    data=df[df['is_capital'] == 1]
).fit(cov_type='clustered', cluster_entity=True)

# (2) 非首都圈（14市 × 10年 = 140个观测）
results['(2) 非首都圈'] = PanelOLS.from_formula(
    'ln_GRDP ~ Fintech + Fintech_sq + INT + GOV + OPE + PC1 + PC2 + Fintech_national + EntityEffects',
    data=df[df['is_capital'] == 0]
).fit(cov_type='clustered', cluster_entity=True)

# (3) 全样本交互项
results['(3) 全样本\n交互项'] = PanelOLS.from_formula(
    'ln_GRDP ~ Fintech + Fintech_sq + Fintech:is_capital + Fintech_sq:is_capital + '
    'INT + GOV + OPE + PC1 + PC2 + Fintech_national + EntityEffects',
    data=df,
    drop_absorbed=True
).fit(cov_type='clustered', cluster_entity=True)

# ================== 4. 输出终稿表格 ==================
print("\n=== 表5-1 金融科技对城市经济增长的异质性影响：首都圈 vs 非首都圈 ===\n")
print(compare(results, stars=True, precision='std-errors'))

# 额外打印分组信息
print("\n【分组信息】")
print("首都圈城市（3个）：", df[df['is_capital']==1].index.get_level_values('city').unique().tolist())
print("非首都圈城市（14个）：", df[df['is_capital']==0].index.get_level_values('city').unique().tolist())


"""
表5-2：异质性分析 —— 按2014年金融科技发展水平分组
"""

import pandas as pd
from linearmodels.panel import PanelOLS, compare

df = pd.read_csv('panel_final_Fintech_national.csv')

# 核心变量
df['Fintech'] = df['ln_fintech_lag1']
df['Fintech_sq'] = df['Fintech'] ** 2
df['Fintech_national'] = df.groupby('year')['Fintech'].transform('mean')

# 计算2014年fintech水平并分组
df_2014 = df[df['year'] == 2014][['city', 'Fintech']].copy()
median_2014 = df_2014['Fintech'].median()

# 合并分组标签
df = df.merge(df_2014[['city', 'Fintech']], left_on='city', right_on='city', suffixes=['', '_2014'])
df['high_initial'] = (df['Fintech_2014'] >= median_2014).astype(int)

df = df.set_index(['city', 'year'])

# ================== 异质性分析：子样本回归 ==================
results = {}

# (1) 高初始fintech水平组（领先城市）
results['(1) 高初始水平组'] = PanelOLS.from_formula(
    'ln_GRDP ~ Fintech + Fintech_sq + INT + GOV + OPE + PC1 + PC2 + Fintech_national + EntityEffects',
    data=df[df['high_initial'] == 1]
).fit(cov_type='clustered', cluster_entity=True)

# (2) 低初始fintech水平组（落后城市）
results['(2) 低初始水平组'] = PanelOLS.from_formula(
    'ln_GRDP ~ Fintech + Fintech_sq + INT + GOV + OPE + PC1 + PC2 + Fintech_national + EntityEffects',
    data=df[df['high_initial'] == 0]
).fit(cov_type='clustered', cluster_entity=True)

# (3) 全样本（作为对比）
results['(3) 全样本（主规格）'] = PanelOLS.from_formula(
    'ln_GRDP ~ Fintech + Fintech_sq + INT + GOV + OPE + PC1 + PC2 + Fintech_national + EntityEffects', df
).fit(cov_type='clustered', cluster_entity=True)

print("\n=== 表5-2 异质性分析：按2014年金融科技发展水平分组（子样本回归）===\n")
print(compare(results, stars=True, precision='std-errors'))

# 打印分组统计
print("\n2014年金融科技水平分组统计：")
print("中位数 =", median_2014)
print(df_2014.sort_values('Fintech', ascending=False))



"""
表5-3：异质性分析 —— 按产业结构（第三产业占比 IS）分组
"""

import pandas as pd
from linearmodels.panel import PanelOLS, compare

df = pd.read_csv('panel_final_Fintech_national_IS.csv')

# 核心变量
df['Fintech'] = df['ln_fintech_lag1']
df['Fintech_sq'] = df['Fintech'] ** 2
df['Fintech_national'] = df.groupby('year')['Fintech'].transform('mean')

# ================== 用2014年第三产业占比（IS）分组 ==================
is_2014 = df[df['year'] == 2014][['city', 'IS']].drop_duplicates()
median_is = is_2014['IS'].median()

print(f"\n2014年第三产业占比中位数：{median_is:.3f}")
high_service = is_2014[is_2014['IS'] >= median_is]['city'].tolist()
low_service  = is_2014[is_2014['IS'] < median_is]['city'].tolist()
print("高服务业组（≥中位数）：", high_service)
print("低服务业组（<中位数）：", low_service)

# 合并分组标签
df = df.merge(is_2014[['city', 'IS']], on='city', suffixes=['', '_2014'])
df['high_service'] = (df['IS_2014'] >= median_is).astype(int)

df = df.set_index(['city', 'year'])

# ================== 三列回归 ==================
results = {}

results['(1) 高服务业城市'] = PanelOLS.from_formula(
    'ln_GRDP ~ Fintech + Fintech_sq + INT + GOV + OPE + PC1 + PC2 + Fintech_national + EntityEffects',
    data=df[df['high_service'] == 1]
).fit(cov_type='clustered', cluster_entity=True)

results['(2) 低服务业城市'] = PanelOLS.from_formula(
    'ln_GRDP ~ Fintech + Fintech_sq + INT + GOV + OPE + PC1 + PC2 + Fintech_national + EntityEffects',
    data=df[df['high_service'] == 0]
).fit(cov_type='clustered', cluster_entity=True)

results['(3) 全样本交互'] = PanelOLS.from_formula(
    'ln_GRDP ~ Fintech + Fintech_sq + Fintech:high_service + Fintech_sq:high_service + '
    'INT + GOV + OPE + PC1 + PC2 + Fintech_national + EntityEffects',
    data=df, drop_absorbed=True
).fit(cov_type='clustered', cluster_entity=True)

print("\n=== 表5-3 金融科技对城市经济增长影响的异质性：基于产业结构（第三产业占比） ===\n")
print(compare(results, stars=True, precision='std-errors'))



"""
表5-4：异质性分析 —— 按企业总资产规模（ln_CAP）分组
"""

import pandas as pd
from linearmodels.panel import PanelOLS, compare

df = pd.read_csv('panel_final_Fintech_national_CAP.csv')

df['Fintech'] = df['ln_fintech_lag1']
df['Fintech_sq'] = df['Fintech'] ** 2
df['Fintech_national'] = df.groupby('year')['Fintech'].transform('mean')

# ================== 用2014年 ln_CAP 分组 ==================
cap_2014 = df[df['year'] == 2014][['city', 'ln_CAP']].drop_duplicates()
median_cap = cap_2014['ln_CAP'].median()

print(f"\n2014年企业总资产对数（ln_CAP）中位数：{median_cap:.3f}")
high_cap_cities = cap_2014[cap_2014['ln_CAP'] >= median_cap]['city'].tolist()
low_cap_cities  = cap_2014[cap_2014['ln_CAP'] < median_cap]['city'].tolist()
print("高企业规模组：", high_cap_cities)
print("低企业规模组：", low_cap_cities)

df = df.merge(cap_2014[['city', 'ln_CAP']], on='city', suffixes=['', '_2014'])
df['high_cap'] = (df['ln_CAP_2014'] >= median_cap).astype(int)

df = df.set_index(['city', 'year'])

results = {}

results['(1) 高企业规模城市'] = PanelOLS.from_formula(
    'ln_GRDP ~ Fintech + Fintech_sq + INT + GOV + OPE + PC1 + PC2 + Fintech_national + EntityEffects',
    data=df[df['high_cap'] == 1]
).fit(cov_type='clustered', cluster_entity=True)

results['(2) 低企业规模城市'] = PanelOLS.from_formula(
    'ln_GRDP ~ Fintech + Fintech_sq + INT + GOV + OPE + PC1 + PC2 + Fintech_national + EntityEffects',
    data=df[df['high_cap'] == 0]
).fit(cov_type='clustered', cluster_entity=True)

results['(3) 全样本交互'] = PanelOLS.from_formula(
    'ln_GRDP ~ Fintech + Fintech_sq + Fintech:high_cap + Fintech_sq:high_cap + '
    'INT + GOV + OPE + PC1 + PC2 + Fintech_national + EntityEffects',
    data=df, drop_absorbed=True
).fit(cov_type='clustered', cluster_entity=True)

print("\n=== 表5-4 金融科技对城市经济增长影响的异质性：基于企业总资产规模 ===\n")
print(compare(results, stars=True, precision='std-errors'))


