from MSET import  *
from DPC import *

label1=('测试1','测试2','测试3',)
label2=('测试1','测试2','测试3',)
N=3#每次输入3个观测向量
train = Z_Score('test_data21.csv')#加载训练集
train_centers = union_func('test_data21.csv')#通过DPC得到聚类中心，作为记忆矩阵
Temp_MemMat(train_centers,'Temp1.npy')#保存MSET计算用的临时矩阵
#test_centers = Z_Score('test_data22.csv')#加载测试集
fault_centers = Z_Score('test_data23.csv')#加载故障集
sim=np.zeros((fault_centers.shape[0],1)) #np.zeros()返回来一个给定形状和类型的用0填充的数组  shape:查看矩阵或者数组的维数
thres=np.zeros((fault_centers.shape[0],1))
Kest=np.zeros((fault_centers.shape[0],fault_centers.shape[1]))
for i in range(int(fault_centers.shape[0]/N)):
    # 加载记忆矩阵与临时矩阵，输入观测向量，计算对应估计向量
    Kest[i*N:(i+1)*N] = MSET(train_centers,fault_centers[i*N:(i+1)*N],'train.npy')
    sim[i*N:(i+1)*N]=Cal_sim(fault_centers[i*N:(i+1)*N],Kest[i*N:(i+1)*N])










