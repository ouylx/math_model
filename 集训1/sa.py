import pandas as pd
import statsmodels.api as sm
import numpy as np
df = pd.read_csv('D:\jupyter_project\三并柜补全数据.csv', index_col=0)
df.index = pd.to_datetime(df.index)
df.index.freq='15s'
df1 = df.loc[df.index>='2023-07-19 00:00:00'] # 选取一个月数据
df1['FH'] = np.log(df['FH']+1)  # 处理数据让预测数据为正

#建立模型
model = sm.tsa.SARIMAX(df1, order=(1, 1, 1), seasonal_order=(1, 1, 1, 60))
SARIMA_m = model.fit()
print(SARIMA_m.summary())
