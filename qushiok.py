import math
import scipy.signal as signal
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

file = '/home/zkl/磨煤机爆燃.csv'
data = pd.read_csv(file) 
'''
突升突降
'''
#去除时间戳列
da = data[data.columns[1:]]

col1 = data["DCS1.10HFC30CT011"]

col11 = [col1[i] for i in col1.index]

col11 = []
col11 = [(col1[i]-col1[i-1])/col1[i-1] for i in col1.index[1:]] 

col11 = [(col1[i]-col1[i-1])/math.fabs(col1[i-1]) for i in col1.index[1:]] 

tmp = range(len(col11))
plt.plot(tmp,col11) 
plt.show() 

col11.pop(np.argmax(col11))

col11.pop(np.argmax(col11))

tmp = range(len(col11))
plt.plot(tmp,col11) 
plt.show() 

#持续上升 持续下降
import scipy.signal as signal  
import numpy as np  
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit 
import pandas as pd 


t = np.linspace(0, 5, 100) 
x = t + np.random.normal(size=100)

z1 = np.polyfit(t,x,1) 
p1 = np.poly1d(z1)

plt.plot(t,x,label="原始值") 
plt.plot(t,p1(t),label="原始值拟合") 


angle = math.atan(p1[1])*180/math.pi