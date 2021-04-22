
import statsmodels
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import statsmodels.api as sm
import xlrd
from matplotlib import pyplot as plt
import pandas as pd

import statsmodels.api as sm #最小二乘
from statsmodels.formula.api import ols #加载ols模型


path ='/home/zkl/监盘/测试数据/EH油系统点信息分类.xlsx'

wb = xlrd.open_workbook(path)

sheet = wb.sheet_by_index(1)

path1 = '/home/zkl/监盘/出差/1.csv'
data= pd.read_csv(path1)

data_train,data_test = data[:37512],data[37512:]
#lm = ols('D04:SELMW ~ D04:FWF+ D04:SELPRESS + D04:SELRHPRS + D04:TFF + D04:FA',data=data_train)
#lm = ols('\'D04:SELMW\'~ \'D04:FWF\'+ \'D04:SELPRESS\' + \'D04:SELRHPRS\' + \'D04:TFF\' + \'D04:FA\'',data=data_train)


X = data.loc[:,('40XAV30CG101XQ01','40LAJ10AP002XQ01','40LAH10CG101XQ01')]

y = data.loc[:,'D04:MDFPF']


#完美 都可以适用 自适应，默认一元一次
model = sm.OLS(y,X ).fit() 
model.summary()
#遍历即可
model.conf_int(0.05)

#显示的定义指定 拟合方程
#若像之前那种方式呢，若实际
data_train.columns = ['date','y','x1','x2','x3','x4','x5']
lm = ols('y~x1+x2+x3+x4+x5',data=data_train)

# todo
# 通用性，一元 二元  三元
# 不显示写出方程式，


lmm = lm.fit()
lmm.summary()

#多元线性回归，的显示， 6个子图，截断与y的图，x1与y的图，等等
fig = sm.graphics.plot_partregress_grid(lmm)
fig.tight_layout(pad=1.0)
plt.show()

data_test.columns = ['date','y','x1','x2','x3','x4','x5']

ypred = lmm.predict(data_test)

y_true = data_test['y']

plt.plot(range(len(ypred)), ypred, 'b', label="predict") 
plt.plot(range(len(ypred)), y_true, 'r', label="test") 
plt.legend(loc="upper right",fontsize='x-large')

plt.show()



import pickle

filepath = r'model/model.pkl'
with open(filepath, 'wb') as f:
    pickle.dump(model, f)