import pandas as pd
import numpy as np
import matplotlib
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")  # 忽略输出警告
df = pd.read_csv('temperature_data.csv')
df = df.set_index('time')
df.index = pd.to_datetime(df.index)
# 过滤每年 8 月和 9 月的数据
df = df[df.index.month.isin([9]) & df.index.day.isin([8, 9, 10])]
matplotlib.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.figure()
plt.plot(df.index, df['temperature'], color='black', linewidth=0.5)
plt.xlabel('时间')
plt.ylabel('温度')
plt.title('每年9月8,9,10日的温度变化')
plt.show()
temperature = df['temperature']


# 创建一个函数来检查数据的平稳性
def test_stationarity(timeseries):
    # 执行Dickey-Fuller测试
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


# 检查原始数据的平稳性
test_stationarity(temperature)

# 将数据化为平稳数据
# 一阶差分
temperature_diff1 = temperature.diff(1)
# 24步差分
temperature_seasonal = temperature_diff1.diff(24)  # 非平稳序列经过d阶常差分和D阶季节差分变为平稳时间序列
# 十二步季节差分平稳性检验结果
test_stationarity(temperature_seasonal.dropna())
temperature_seasonal = temperature_seasonal.dropna()
# 可视化原始数据和差分后的数据
plt.figure()
plt.plot(temperature, label='Original')
plt.show()
plt.figure()
plt.plot(temperature_diff1, label='1st Order Difference')
plt.show()
plt.figure()
plt.plot(temperature_seasonal, label='seasonal Order Difference')
plt.show()
"""ACF,PACF图"""
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

pacf = plot_pacf(temperature_seasonal, lags=40)
plt.title('PACF')
pacf.show()
acf = plot_acf(temperature_seasonal, lags=40)
plt.title('ACF')
acf.show()

#LB白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox


def test_white_noise(data, alpha):
    lb_test_results = acorr_ljungbox(data, lags=1)
    lb = lb_test_results.iloc[0, 0]
    p = lb_test_results.iloc[0, 1]
    if p < alpha:
        print('LB白噪声检验结果：在显著性水平%s下，数据经检验为非白噪声序列' % alpha)
    else:
        print('LB白噪声检验结果：在显著性水平%s下，数据经检验为白噪声序列' % alpha)


#白噪声检验
test_white_noise(temperature_seasonal.dropna(), 0.01)

import itertools


# 搜索法定阶
def SARIMA_search(data):
    p = q = range(0, 3)
    s = [24]  # 周期为24
    d = [1]  # 做了一次季节性差分
    PDQs = list(itertools.product(p, d, q, s))  # itertools.product()得到的是可迭代对象的笛卡儿积
    pdq = list(itertools.product(p, d, q))  # list是python中是序列数据结构，序列中的每个元素都分配一个数字定位位置
    params = []
    seasonal_params = []
    results = []
    grid = pd.DataFrame()
    for param in pdq:
        for seasonal_param in PDQs:
            # 建立模型
            mod = sm.tsa.SARIMAX(data, order=param, seasonal_order=seasonal_param, enforce_stationarity=False,
                                 enforce_invertibility=False)
            # 实现数据在模型中训练
            result = mod.fit()
            print("ARIMA{}x{}-BIC:{}".format(param, seasonal_param, result.bic))
            # format表示python格式化输出，使用{}代替%
            params.append(param)
            seasonal_params.append(seasonal_param)
            results.append(result.aic)
    grid["pdq"] = params
    grid["PDQs"] = seasonal_params
    grid["bic"] = results
    print(grid[grid["bic"] == grid["bic"].min()])


#SARIMA_search(temperature)
#建立模型
model = sm.tsa.SARIMAX(temperature_seasonal, order=(1, 1, 2), seasonal_order=(0, 1, 2, 24))
SARIMA_m = model.fit()
print(SARIMA_m.summary())
#模型检验
test_white_noise(SARIMA_m.resid, 0.05)  #SARIMA_m.resid提取模型残差，并检验是否为白噪声
fig = SARIMA_m.plot_diagnostics(figsize=(15, 12))  #plot_diagnostics对象允许我们快速生成模型诊断并调查任何异常行为
#模型预测
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae


# 获取预测结果，自定义预测误差
def PredictionAnalysis(data, model, start, dynamic=False):
    pred = model.get_prediction(start=start, dynamic=dynamic, full_results=True)
    pci = pred.conf_int()  #置信区间
    pm = pred.predicted_mean  #预测值
    truth = data[start:]  #真实值
    pc = pd.concat([truth, pm, pci], axis=1)  #按列拼接
    pc.columns = ['true', 'pred', 'up', 'low']  #定义列索引
    print("1、MSE:{}".format(mse(truth, pm)))
    print("2、RMSE:{}".format(np.sqrt(mse(truth, pm))))
    print("3、MAE:{}".format(mae(truth, pm)))
    return pc


#绘制预测结果
def PredictonPlot(pc):
    plt.figure()
    plt.plot(pc['true'], label='base data')
    plt.plot(pc['pred'], label='prediction curve')
    plt.legend()
    plt.show()
    return True


pred = PredictionAnalysis(temperature_seasonal, SARIMA_m, start='2010-09-09 01:00:00')  # 预测全部
PredictonPlot(pred)

#预测未来
forecast = SARIMA_m.get_forecast(steps=72)
pre = forecast.predicted_mean
pre.index = pd.date_range(start='2024-09-08 00:00', end='2024-09-10 23:00', freq='H')
pre = np.array(pre)
temperature_copy = np.array(temperature.copy())
pre[0:24] += temperature_copy[-24:]
for i in range(24, 72):
    pre[i] += pre[i - 24]
pred = pd.DataFrame(pre, index=pd.date_range(start='2024-09-08 00:00', end='2024-09-10 23:00', freq='H'))
#预测整体可视化
plt.figure()
plt.plot(pred.index, pred)
plt.legend(['预测温度'], loc="best")
plt.xlabel("时间")
plt.ylabel("温度")
plt.show()
filltered_pred = pred.between_time(start_time='09:00', end_time='21:00')
filltered_pred.to_csv('9-21点预测温度.csv')
pred.to_csv('全时段预测温度.csv')
