# 这里把所有有可能用到的包都导了
from statsmodels.tsa.stattools import adfuller # 平稳性检测
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # 画acf, pacf图
from statsmodels.tsa.arima_model import ARIMA # ARIMA模型
from statsmodels.graphics.api import qqplot # 画qq图
from scipy.stats import shapiro # 正态检验
import statsmodels.tsa.stattools as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
import statsmodels
import seaborn as sns
import matplotlib.pylab as plt
from scipy import  stats

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
df = pd.read_csv('temperature_data.csv')
df = df.set_index('time')
df.index = pd.to_datetime(df.index)
# 过滤每年 8 月和 9 月的数据
df = df[df.index.month.isin([8, 9])]
# 一般做一阶或二阶差分可使序列平稳
diff_df = df.copy()
diff_df.index=df.index
# 一阶差分
diff_df['diff_1'] = diff_df.diff(1).dropna()
# 二阶差分
diff_df['diff_2'] = diff_df['diff_1'].diff(1).dropna()

# 作图
diff_df.plot(subplots=True,figsize=(18,20))
plt.show()
# adfuller单位根检验数据平稳性
from statsmodels.tsa.stattools import adfuller
print(adfuller(df))  # 原始数据
print(adfuller(diff_df['diff_1'].dropna()))  # 一阶差分
print(adfuller(diff_df['diff_2'].dropna()))  # 二阶差分

temperature_diff = diff_df['diff_1'].dropna()
from statsmodels.stats.diagnostic import acorr_ljungbox
print(acorr_ljungbox(temperature_diff, lags = 20))
# 画pacf图和acf图
"""from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
pacf = plot_pacf(temperature_diff, lags=40)
plt.title('PACF')
pacf.show()
acf = plot_acf(temperature_diff, lags=40)
plt.title('ACF')
acf.show()"""
import itertools

# 这里最大最小的参数可以自己调
p_min = 0
d_min = 0
q_min = 0
p_max = 8
d_max = 1
q_max = 8

# Initialize a DataFrame to store the results,，以BIC准则
results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max + 1)],
                           columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])

for p, d, q in itertools.product(range(p_min, p_max + 1),
                                 range(d_min, d_max + 1),
                                 range(q_min, q_max + 1)):
    if p == 0 and d == 0 and q == 0:
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue

    try:
        model = sm.tsa.ARIMA(df, order=(p, d, q),
                             # enforce_stationarity=False,
                             # enforce_invertibility=False,
                             )
        results = model.fit()
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
    except:
        continue
results_bic = results_bic[results_bic.columns].astype(float)
fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(results_bic,
                 mask=results_bic.isnull(),
                 ax=ax,
                 annot=True,
                 fmt='.2f',
                 )
ax.set_title('BIC')
plt.show()
from statsmodels.tsa.arima_model import ARIMA
# ARIMA(data, order=(p, d, q))
model = ARIMA(df, order=(8,1,6))
result = model.fit()
result.summary()
"""# 获取残差
resid = result.resid

# 画qq图
from statsmodels.graphics.api import qqplot
qqplot(resid, line='q', fit=True)
plt.show()
from scipy.stats import shapiro
shapiro(resid)
import statsmodels.api as sm
print(sm.stats.durbin_watson(resid.values))
r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1,21), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))"""
# 预测出来的数据也为一阶差分
# predict(起始时间，终止时间)
predict = result.predict('2024-09-8','2024-9-10')
plt.figure(figsize=(12, 8))
plt.plot(temperature_diff)
plt.plot(predict)
pred_all = result.predict('2010-08-01','2024-08-04')
plt.figure(figsize=(12, 8))
plt.plot(temperature_diff)
plt.plot(pred_all)
