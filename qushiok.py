'''
import math
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from datetime import datetime
from datetime import timedelta

# file = '/home/zkl/磨煤机爆燃.csv'
file = './磨煤机爆燃.csv'
data = pd.read_csv(file)
'''
突升突降,主要还是要确定阈值
'''
#获取时间序列
dateSerise = list(data[data.columns[0]])

endTime = dateSerise[-1]
#将时间字符串装换成时间类型
endTime = datetime.strptime(endTime,'%Y/%m/%d %H:%M')
print(endTime)
frontTime = endTime - timedelta(minutes=10)
print(str(frontTime))
print(frontTime)
print(type(frontTime))

#去除时间戳列
da = data[data.columns[1:]]

#获取第一列的数据
col1 = data["DCS1.10HFC30CT011"]

col11 = [col1[i] for i in col1.index]

dataTest = col11[0:4]
dataTest.pop(0)

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
'''

'''
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
'''



import os
import math
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
# import paramiko
import pickle
import statsmodels
from statsmodels.sandbox.regression.predstd import wls_prediction_std
# import xlrd
from matplotlib import pyplot as plt
import statsmodels.api as sm  # 最小二乘
from statsmodels.formula.api import ols  # 加载ols模型
import wget
import json
import requests
import subprocess
# from config import Config
import logging
from enum import Enum
import redis
# import happybase
from datetime import datetime
from datetime import timedelta
import time
from concurrent.futures import ThreadPoolExecutor


data = pd.read_csv('磨煤机爆燃.csv')
y = data['DCS1.10HFC30CT011'].values


# 构造变量
datatime = data['DCS1.10HFC30CT011'].values
x = np.arange(0,len(y),1) # x值
X = sm.add_constant(x) # 回归方程添加一列 x0=1

mainKKS = 'DCS1.10HFC30CT011'
data = list(data[mainKKS])
# data = np.array(data)
# tmp = np.arange(0,len(data),1)
tmp = range(len(data))
tmp = sm.add_constant(tmp)

# 建回归方程
# OLS（endog,exog=None,missing='none',hasconst=None) (endog:因变量，exog=自变量）
# modle = sm.OLS(y,X) # 最小二乘法
modle = sm.OLS(data,tmp) # 最小二乘法
res = modle.fit()   # 拟合数据
beta = res.params   # 取系数
print(beta[1])
print(res.summary())  # 回归分析摘要


# 画图
Y = res.fittedvalues    # 预测值
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(x, y, '-', label='jz')  # 原始数据
ax.plot(x, Y, 'r--.',label='fit') # 拟合数据
ax.legend(loc='upper left') # 图例，显示label
plt.title('predict fund net value: ')
plt.xlabel('x')
plt.ylabel('jz')
plt.show()

