import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

step=100
delta=0.001

#计算保存记忆矩阵的Temp矩阵
def Temp_MemMat(memorymat,Temp_name):
    memorymat_row = memorymat.shape[0]
    Temp = np.zeros((memorymat_row, memorymat_row))
    for i in range(memorymat_row):
        for j in range(memorymat_row):
            Temp[i, j] = np.linalg.norm(memorymat[i] - memorymat[j])
    np.save(Temp_name,Temp)

#MSET计算，被MSETs调用
def MSET(memorymat,Kobs,Temp_name):#Temp为临时计算的矩阵
    memorymat_row=memorymat.shape[0]
    Kobs_row=Kobs.shape[0]
    Temp=np.load(Temp_name)
    Temp1=np.zeros((memorymat_row,Kobs_row))
    for m in range(memorymat_row):
        for n in range(Kobs_row):
            Temp1[m,n]=np.linalg.norm(memorymat[m] - Kobs[n])
    Kest=np.dot(np.dot(memorymat.T,(np.linalg.pinv(Temp))),Temp1)
    Kest=Kest.T
    return Kest

#判断输入的观测向量，再传到相应记忆矩阵中，得到估计值
def MSETs(memorymat1_name,memorymat2_name,memorymat3_name,Kobs):
    row_Kobs=Kobs.shape[0]
    col_Kobs = Kobs.shape[1]
    Kest=np.zeros((row_Kobs,col_Kobs))
    for t in range(row_Kobs):
        if Kobs[t,col_Kobs-1]<1/3:
            Kest[t] = MSET(memorymat1_name,Kobs[t:t+1,:],'Temp_low.npy')
        elif Kobs[t,col_Kobs-1]>2/3:
            Kest[t] = MSET(memorymat3_name, Kobs[t:t+1,:],'Temp_hig.npy')
        else:
            Kest[t] = MSET(memorymat2_name,Kobs[t:t+1,:],'Temp_med.npy')
    return Kest

#基于融合距离的相似度计算
def Cal_sim(Kobs,Kest):
    dist_norm = np.zeros((Kobs.shape[0],1))
    dist_cos = np.zeros((Kobs.shape[0], 1))
    for i in range(Kobs.shape[0]):
        dist_norm[i]=np.linalg.norm(Kobs[i, :] - Kest[i, :]) # 欧式距离
        dist_cos[i]= np.dot(Kobs[i, :], Kest[i, :]) /\
                     (np.linalg.norm(Kobs[i, :]) * np.linalg.norm(Kest[i, :]))  # dot向量内积，norm向量二范数
    dist_cos= dist_cos* 0.5 + 0.5  # 余弦距离平移至[0,1]
    sim = (1 / (1 + dist_norm / dist_cos))  # 相似度公式
    return sim

#各变量及其误差的可视化
def pic_vars(label,Kobs,Kest,np_Dmax,np_Dmin):
    Kobs = Kobs * (np_Dmax - np_Dmin) + np_Dmin  # 反归一化
    Kest = Kest * (np_Dmax - np_Dmin) + np_Dmin  # 反归一化
    col_num=Kobs.shape[1]
    e=np.ones((Kobs.shape[0],Kobs.shape[1]))
    plt.ion()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 图片显示中文
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(col_num):
        plt.figure()
        plt.subplot(211)
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        plt.plot(Kobs[:, i], 'steelblue', label='观测值', lw=1.5)
        plt.plot(Kest[:, i], 'indianred', label='估计值', lw=1.5)
        plt.legend(loc='upper right', fontsize=13)
        plt.xlabel('样本序号', fontsize=20)
        plt.ylabel(label[i], fontsize=20, verticalalignment='bottom')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.subplot(212)
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        e[:, i] = (np.abs(Kobs[:, i] - Kest[:, i]) / Kobs[:, i]) * 100
        plt.plot(e[:, i], 'peru', lw=1)  # 偏离度
        plt.xlabel('样本序号', fontsize=20)
        plt.ylabel('相对误差/%', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=18)
        plt.show()
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

#误差贡献率
def error_contribution(Kobs,Kest,momtent,label):
    error=(Kobs - Kest)**2
    error_cont =[]
    for row in error:
        error_cont.append(row/row.sum())
    plt.ion()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 图片显示中文
    plt.rcParams['axes.unicode_minus'] = False
    plt.bar(np.arange(1,Kobs.shape[1]+1,1), error_cont[momtent])
    plt.xticks(range(1, Kobs.shape[1]+1, 1), label, rotation=80)
    plt.title('1min内第%d时刻各变量的误差贡献率'%(momtent+1))
    plt.show()

# 累计误差贡献率
def Accumu_errorContirbution(Kobs,Kest,momtent,time_range,label):
    if time_range==0:
        print('Warning:time_range cannot be zero')
        return
    else:
        error = (Kobs - Kest) ** 2
        error_cont = np.zeros((1,Kobs.shape[1]))
        for i in range(time_range):
            error_cont += error[momtent+i]/error[momtent+i].sum()
        error_cont = np.squeeze(error_cont/time_range)
        plt.ion()
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 图片显示中文
        plt.rcParams['axes.unicode_minus'] = False
        plt.bar(np.arange(1,Kobs.shape[1]+1,1) ,error_cont)
        plt.xticks(range(1,Kobs.shape[1]+1,1),label,rotation=80)
        plt.title('1min内第%d个时刻发出预警起的累计误差贡献率' % (momtent+1))
        plt.show()

#更新记忆矩阵，暂未用
def Mat_update(Kobs,sim,thres,memorymat_name,Temp_name):
    Kobs_row=Kobs.shape[0]
    Kobs_col = Kobs.shape[1]
    k_index=np.arange(201,301,1)
    break_flag=False
    mat_temp = []
    for i in range(Kobs_row):
        if sim[i]>thres :#判断观测向量是否正常
            for k,k_in in enumerate(k_index):
                for j in range(Kobs_col):
                    if np.abs(Kobs[i,j] - k_in * (1 / step)) < delta:
                        mat_temp.append(Kobs[i])
                        print('add state')
                        break_flag=True
                        break
                if break_flag==True:
                    break
    mat_temp= np.array(mat_temp, dtype=float)
    print('size of mat_temp:',mat_temp.shape)
    # Temp_MemMat(memorymat,Temp_name)
    # print('size of memorymat',memorymat.shape)
    # np.save(memorymat_name,memorymat)
    return mat_temp




