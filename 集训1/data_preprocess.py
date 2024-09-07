import pandas as pd
data = pd.read_csv(r'C:\Users\25492\Desktop\hangzhou_weather.csv')
print(data.head())
print(data.info())
temperature = pd.DataFrame(columns=['时间','温度'])
for i in range(0,data.shape[0],9):
    temperature.loc[i,'温度'] = data.iloc[i:i+9,3].mean()
    temperature.loc[i,'时间'] = data.iloc[i,0]
temperature = temperature.set_index('时间')
temperature.to_csv('10-23年7-10月份每小时温度.csv')