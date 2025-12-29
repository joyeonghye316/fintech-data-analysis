
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS

# =============== 1. 跑一次基准回归 ===============
df = pd.read_csv('panel_final_Fintech_national.csv')

df['fintech'] = df['ln_fintech_lag1']
df['fintech_sq'] = df['fintech'] ** 2
df['fintech_national'] = df.groupby('year')['fintech'].transform('mean')

df = df.set_index(['city', 'year'])

res_baseline = PanelOLS.from_formula(
    'ln_GRDP ~ fintech + fintech_sq + INT + GOV + OPE + PC1 + PC2 + fintech_national + EntityEffects',
    df
).fit(cov_type='clustered', cluster_entity=True)

print(res_baseline.summary)

# =============== 2. 计算边际效应、拐点和标准误 ===============

b1 = res_baseline.params['fintech']
b2 = res_baseline.params['fintech_sq']

# 从协方差矩阵中直接取出方差和协方差
cov_mat = res_baseline.cov
var_b1 = cov_mat.loc['fintech', 'fintech']
var_b2 = cov_mat.loc['fintech_sq', 'fintech_sq']
cov_b1b2 = cov_mat.loc['fintech', 'fintech_sq']

# 拐点：-b1 / (2*b2)
turning = -b1 / (2 * b2)

# 构造 fintech 的取值区间（用样本的 min ~ max）
f_min, f_max = df['fintech'].min(), df['fintech'].max()
f_grid = np.linspace(f_min, f_max, 200)

# 边际效应：d ln_GRDP / d fintech = b1 + 2*b2*fintech
me = b1 + 2 * b2 * f_grid

# Delta method 计算每个点的标准误
# Var(me) = Var(b1) + (2x)^2 Var(b2) + 2*(2x)*Cov(b1,b2)
var_me = (
    var_b1
    + (2 * f_grid) ** 2 * var_b2
    + 2 * (2 * f_grid) * cov_b1b2
)
se_me = np.sqrt(var_me)
me_upper = me + 1.96 * se_me
me_lower = me - 1.96 * se_me

# =============== 3. 绘制边际效应曲线 ===============

plt.figure(figsize=(7, 4.5))

# 边际效应曲线
plt.plot(f_grid, me, linewidth=2, label='Marginal effect of fintech')

# 95% 置信区间阴影
plt.fill_between(f_grid, me_lower, me_upper, alpha=0.25, label='95% CI')

# 拐点竖线
plt.axvline(x=turning, linestyle='--', linewidth=1.5, label='Turning point')

# y=0 虚线
plt.axhline(y=0, linestyle='--', linewidth=1)

# 轴标签和标题
plt.xlabel('Fintech level (ln patents, lagged)')
plt.ylabel('Marginal effect on ln(GRDP)')
plt.title('Marginal effects of fintech on economic output')

# 图例放在左上角，避免遮住曲线
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
