1.DPC.py：

  def Z_Score：数据预处理
  
  def getDistanceMatrix：计算距离矩阵
  
  def select_dc：计算dc
  
  def get_density：计算局部密度
  
  def get_deltas：计算密度距离
  
  def find_centers_K：(rho, deltas, K)：获取聚类中心索引
  
  def union_func：合数据处理和DPC各部分，返回聚类中心所在行所有数据
  
  
2.MSET.py：

  def Temp_MemMat：计算保存记忆矩阵的Temp矩阵
  
  def MSET：MSET计算，被MSETs调用
  
  def MSETs：判断输入的观测向量，再传到相应记忆矩阵中，得到估计值
  
  def Cal_sim：基于融合距离的相似度计算
  
  def pic_vars：各变量及其误差的可视化
  
  def error_contribution：误差贡献率
  
  def Accumu_errorContirbution：累计误差贡献率
  
  def Mat_update：更新记忆矩阵
  
  
3.example.py：写入数据测试
  
  
  
  
  
  
  
