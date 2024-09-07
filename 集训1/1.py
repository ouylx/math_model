import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

data = pd.read_excel(r'C:\Users\25492\Desktop\研究生集训题1 泳池水质安全控制问题\附件1.xlsx')
x = np.array(data.iloc[1:, 0])
y = np.array(data.iloc[1:, 1])
x = x.astype(np.float64)
y = y.astype(np.float64)
#画出原始的序列
plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示字体
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('人数')
plt.ylabel('0.5小时后余氯浓度（mg/L）')
plt.plot(x, y, 'o')
plt.legend()

plt.show()

# 将原始序列分成多段,用一次函数拟合为函数f1和用三次函数拟合为函数f2
f1 = interp1d(x, y, kind='linear')
f2 = interp1d(x, y, kind='cubic')

#在原区间内均匀选取30个点,因为要插值到长度30.
x_pred = np.linspace(0, 260, num=30)

#用函数f1求出插值的30个点对应的值
y1 = f1(x_pred)
#在图中画出插值的30个点并连成曲线
plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示字体
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('人数')
plt.ylabel('0.5小时后余氯浓度（mg/L）')
plt.plot(x, y, 'bo')
plt.plot(x_pred, y1, '-rx', label='linear')
plt.legend()
plt.show()

#用函数f2求出插值的30个点对应的值
y2 = f2(x_pred)
#在图中画出插值的30个点并连成曲线
plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示字体
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('人数')
plt.ylabel('0.5小时后余氯浓度（mg/L）')
plt.plot(x, y, 'bo')
plt.plot(x_pred, y2, '-rx', label='cubic')
plt.legend()
plt.show()

y_pre = f2(255)
print(y_pre)