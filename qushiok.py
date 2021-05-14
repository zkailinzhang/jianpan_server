import math
import scipy.signal as signal
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# file = '/home/zkl/磨煤机爆燃.csv'
file = './磨煤机爆燃.csv'
data = pd.read_csv(file) 
'''
突升突降,主要还是要确定阈值
'''
#去除时间戳列
da = data[data.columns[1:]]

#获取第一列的数据
col1 = data["DCS1.10HFC30CT011"]

col11 = [col1[i] for i in col1.index]

col11 = []

col11 = [(col1[i]-col1[i-1])/math.fabs(col1[i-1]) for i in col1.index[1:]] 

tmp = range(len(col11))
tmp1 = range(len(col1))
plt.plot(tmp,col11) 
plt.show() 

print(np.argmax(col11))
print(np.max(col11))

col11.pop(np.argmax (col11))

col11.pop(np.argmin(col11))

tmp = range(len(col11))
plt.plot(tmp,col11) 
plt.show() 

#持续上升 持续下降
import scipy.signal as signal  
import numpy as np  
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit 
import pandas as pd 

#这是构造了两列数据啊
t = np.linspace(0, 5, 100) 
x = t + np.random.normal(size=100)
#x = t - np.random.normal(size=100)

z1 = np.polyfit(t,x,1) 
p1 = np.poly1d(z1)
z2 = np.polyfit(tmp1,col1,1)
p2 = np.poly1d(z2)

#解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.plot(t,x,label="原始值")
plt.plot(t,p1(t),label="原始值拟合")
plt.legend(loc='upper left')
plt.title('fitting')
plt.show()
plt.plot(tmp1,col1,label="原始值")
plt.plot(tmp1,p2(tmp1),label="原始值拟合")
plt.legend(loc='upper left')
plt.title('fitting')
plt.show()

#获取这个斜率对应的角度值，当角度值大于0°的时候就认为是增加的 趋势增加
angle = math.atan(p1[1])*180/math.pi
print(angle)
#todo 

