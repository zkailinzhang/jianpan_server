# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np 
from flask import Flask, jsonify, request
#import paramiko
import pickle
import statsmodels
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import xlrd
from matplotlib import pyplot as plt
import statsmodels.api as sm #最小二乘
from statsmodels.formula.api import ols #加载ols模型
import wget
import json
import requests
import subprocess
from config import Config
import logging 
from enum import Enum
import redis 
import happybase
from concurrent.futures import ThreadPoolExecutor
import shutil


executor = ThreadPoolExecutor(8)

pathcwd = os.path.dirname(__file__)
app = Flask(__name__)

class STATES(Enum):
    
    XUNLIAN_ZHONG = 1
    XL_WANCHENG =2
    
    PINGGU_ZHONG =3
    PG_WANCHENG=4
    PG_QUXIAO = 5
    
    FABU = 6
    FB_QUXIAO =7  
    

logpath = 'log/serving.std.out'
logging.basicConfig(filename=logpath,filemode='a',format='%(asctime)s %(name)s:%(levelname)s:%(message)s',datefmt="%d-%m-%Y \
    %H:%M:%S",level=logging.DEBUG)
'''
#STATES_ENUM ={"初始化":0,"待训练":1,"训练中":2,'待发布':3,'已发布':4,"训练失败":5}
# 状态管理，没有取消训练， 但有取消训练接口， 就直接remove，key，
状态管理 有取消评估，接口，  状态置 训练完成  模型内存保存
状态管理 有取消发布，接口，  状态置 训练完成  模型内存保存
"id":{"status":,"model":，"modelid":,"firstConfidence"，"secondConfidence":,}
'''

MODELS_STATUS = {} #模型状态管理字典
# "id":{"status":,"model":，"modelid":,"firstConfidence"，"secondConfidence":,}


# # "train_version":0123,"release_verison":0123}  
#评分的线性回归



@app.route('/predict', methods=['POST','GET'])
def predict():
    try:
        request_json = request.get_json()   
        model_id = request_json["modelId"]
        delta1 = request_json["firstConfidence"]
        delta2 = request_json["secondConfidence"]
        logging.info("******predicting modelid {},".format(model_id))
 
        if (delta1>=0.98 or delta1 <0.95 or delta2>=1.0 or delta2 <0.98): return (bad_request(504)) 

        print("Loading the model...")
        loaded_model = None
        clf = 'model.pkl'
        #if str(model_id) in request_json.keys():
        #    loaded_model = request_json[str(model_id)]
        local_path = './model/' + str(model_id)+'/'
        if not os.path.exists(local_path):return(bad_request(502))
        #if str(model_id) not in MODELS_MAP.keys():
        with open(local_path + clf,'rb') as f:
            loaded_model = pickle.load(f)
        
            
        #logging.info("******predicting summary {},".format(loaded_model.summary()  )) 
        params  = loaded_model.params.index
        columns = list(params[1:])
        for col in columns:
            if col not in request_json.keys():
                return(bad_request(501))
                      
    except Exception as e:
        logging.info("******predicting modelid {},excp:{}".format(model_id,e))
        raise e 
    data=[]
    for i in range(1,len(params)):
        
        data.append(request_json[params[i]]) 
    
    if len(data)==0 or '' in data:
        return(bad_request(400))
    else:
        data =  np.expand_dims(data,1)       
        df = pd.DataFrame(dict(zip(columns,data)))

        print("The model has been loaded...doing predictions now...")
        
        df.insert(0,'const',1.0)
        df_const = df
        predictions = loaded_model.predict(df_const)
             
        paras  = loaded_model.conf_int(Config.confidence)
        d_pre = paras[0][0]
        up_pre = paras[1][0]
        
        for i in range(1,len(paras)):
            d_pre += df_const.loc[0][i] * paras[0][i] 
            up_pre += df_const.loc[0][i] * paras[1][i] 

        paras  = loaded_model.conf_int(Config.confidence_second)
        d_pre2 = paras[0][0]
        up_pre2 = paras[1][0]
        
        for i in range(1,len(paras)):
            d_pre2 += df_const.loc[0][i] * paras[0][i] 
            up_pre2 += df_const.loc[0][i] * paras[1][i] 
        up_pre += up_pre* (Config.K1*delta1-Config.B1)
        up_pre2 += up_pre2 *(Config.K2*delta2-Config.B2)
        d_pre -=d_pre* (Config.K1*delta1-Config.B1)
        d_pre2-=d_pre2*(Config.K2*delta2-Config.B2)
        pred_interval = {"prediction":predictions.loc[0],"first_upper":up_pre,"first_lower":d_pre,"second_upper":up_pre2,"second_lower":d_pre2}
              
        message = {
			'status': True,
			'message': "请求成功",
            'data':pred_interval
	    }
        logging.info("******predicting finished modelid {},".format(model_id))
        responses = jsonify(message)
        
        responses.status_code = 200
    
    return (responses)

@app.route('/predict_publish', methods=['POST','GET'])
def predict_publish():
    try:
        request_json = request.get_json()   
        model_id = request_json["modelId"]
        delta1 = request_json["firstConfidence"]
        delta2 = request_json["secondConfidence"]
        if (delta1>=0.98 or delta1 <0.95 or delta2>=1.0 or delta2 <0.98): return (bad_request(504))

        print("Loading the model...")
        # loaded_model = None
        # clf = 'model.pkl'
        #判断模型是不是存在 模型是否已经发布
        if str(model_id) not in MODELS_STATUS.keys():
            return(bad_request(503))
        elif MODELS_STATUS[str(model_id)]['status'] != STATES.FABU:
            return(bad_request(503))
        loaded_model = MODELS_STATUS[str(model_id)]['model']

        #如果内存中不存在该模型，则认为该模型不存在 不再到磁盘加载模型，则下边的代码可以删除
        # local_path = './model/publist/' + str(model_id)+'/'
        # if not os.path.exists(local_path):return(bad_request(502))
        # #if str(model_id) not in MODELS_MAP.keys():
        # with open(local_path + clf,'rb') as f:
        #     loaded_model = pickle.load(f)
             
        params  = loaded_model.params.index
        columns = list(params[1:])
        for col in columns:
            if col not in request_json.keys():
                return(bad_request(501))
                      
    except Exception as e:
        raise e 
    data=[]
    for i in range(1,len(params)):
        print(request_json[params[i]])
        data.append(request_json[params[i]]) 
    
    if len(data)==0 or '' in data:
        return(bad_request(400))
    else:
        data =  np.expand_dims(data,1)       
        df = pd.DataFrame(dict(zip(columns,data)))

        print("The model has been loaded...doing predictions now...")
        
        df.insert(0,'const',1.0)
        df_const = df
        predictions = loaded_model.predict(df_const)
             
        paras  = loaded_model.conf_int(Config.confidence)
        d_pre = paras[0][0]
        up_pre = paras[1][0]
        
        for i in range(1,len(paras)):
            d_pre += df_const.loc[0][i] * paras[0][i] 
            up_pre += df_const.loc[0][i] * paras[1][i] 

        paras  = loaded_model.conf_int(Config.confidence_second)
        d_pre2 = paras[0][0]
        up_pre2 = paras[1][0]
        
        for i in range(1,len(paras)):
            d_pre2 += df_const.loc[0][i] * paras[0][i] 
            up_pre2 += df_const.loc[0][i] * paras[1][i] 
        up_pre += up_pre* (Config.K1*delta1-Config.B1)
        up_pre2 += up_pre2 *(Config.K2*delta2-Config.B2)
        d_pre -=d_pre* (Config.K1*delta1-Config.B1)
        d_pre2-=d_pre2*(Config.K2*delta2-Config.B2)
        pred_interval = {"prediction":predictions.loc[0],"first_upper":up_pre,"first_lower":d_pre,"second_upper":up_pre2,"second_lower":d_pre2}

        prediction_series = pd.DataFrame(pred_interval,index=[0])
              
        message = {
			'status': True,
			'message': "请求成功",
            'data':pred_interval
	    }
        
        responses = jsonify(message)
        
        responses.status_code = 200
    
    return (responses)


@app.errorhandler(400)
def bad_request(error=400):
    
    message = {
        'status': False,
		'message': '',
	    'data':'',
        }
    if error == 400:
        message.update( {
			'message': '-->数据为空错误，请检查相关测点数据',
	    })
        resp = jsonify(message)
        resp.status_code = 400
    elif error ==501:
        message.update( {
			'message': '-->请求参数不一致错误，请检查相关测点、模型ID、请求接口',
	    })
        resp = jsonify(message)
        resp.status_code = 501
    elif error ==502:
        message.update( {
			'message': '-->模型文件找不到',
	    })
        resp = jsonify(message)
        resp.status_code =502
    elif error ==503:
        message.update( {
			'message': '-->模型未发布',
	    })
        resp = jsonify(message)
        resp.status_code =503
    elif error ==504:
        message.update( {
			'message': '-->置信度超出范围',
	    })
        resp = jsonify(message)
        resp.status_code =504
    elif error ==505:
        message.update( {
			'message': '-->远程数据文件不存在',
	    })
        resp = jsonify(message)
        resp.status_code =505
    elif error ==506:
        message.update({
            'message': '-->模型不为训练完成状态'
        })
        resp = jsonify(message)
        resp.status_code = 506
    return resp



@app.route('/publish', methods=['POST'])
def publish():
    request_json = request.get_json()
    if 'modelId' not in request_json.keys:
        return(bad_request(501))
     
    model_id = request_json["modelId"]
    #若已经发布了，则直接返回模型已经发布过（实际上应该不存在这种情况吧，模型如果已经发布过，应该不会出现在模型未发布的列表中）
    path = ''
    
    clf = 'model.pkl'
    loaded_model = MODELS_STATUS[str(model_id)]['model']
    #不需要从磁盘加载模型，此时模型已经在内存中
    # if str(model_id) not in MODELS_STATUS.keys():
    #     with open('./model/publish/' + str(model_id)+'/' + clf,'rb') as f:
    #         loaded_model = pickle.load(f)
    #
    # MODELS_STATUS[str(model_id)]['model'] = loaded_model
    if str(model_id) in MODELS_STATUS.keys():
        MODELS_STATUS[str(model_id)]['status'] = STATES.FABU
    else:
        return bad_request(502)
    message = {
			'status': True,
			'message': request.url+'-->模型发布成功',
	}
    resp = jsonify(message)
    resp.status_code = 200
    return resp

@app.route('/publish_cancel', methods=['POST'])
def publish_cancel():
    request_json = request.get_json()
    if 'modelId' not in request_json.keys:
        # return(bad_request(401)) #此处为请求参数错误 应该为501错误码
        return (bad_request(501))
    model_id = request_json["modelId"]   
    
    if model_id not in MODELS_STATUS.keys:
        # return(bad_request(401)) #此处为模型文件找不到 应该为502错误码
        return (bad_request(502))
    #已经pop 在pop 报错

    # MODELS_STATUS.pop[str(model_id)]
    # 取消发布之后将模型的状态改为训练完成状态,同时将第一二置信度不变
    MODELS_STATUS[str(model_id)]['status'] = STATES.XL_WANCHENG
    # MODELS_STATUS[str(model_id)]['firstConfidence'] = None
    # MODELS_STATUS[str(model_id)]['secondConfidence'] = None

    message = {
			'status': True,
			'message': request.url+'-->模型取消发布成功',
	}
    resp = jsonify(message)
    resp.status_code = 200
    return resp

def evaluate_task(delta2,evaluationId,delta1,local_path_csv,assistKKS,model_id,loaded_model,epochs,chunks):
    
    logging.info("******evaluate_task  modelid {} ,".format(model_id))
    data = pd.read_csv(local_path_csv)
    X = data.loc[:,tuple(assistKKS)]
    #y = data.loc[:,mainKKS]
    X_const = sm.add_constant(X)
    times_start = data.iloc[:1,0].values[0]

    #df.insert(0,'const',1.0)
    
    #批量预测
    predictions = loaded_model.predict(X_const)
    df_const = X_const
    paras  = loaded_model.conf_int(Config.confidence)
    d_pre = paras[0][0]
    up_pre = paras[1][0]
    
    for i in range(1,len(paras)):
        d_pre += df_const[assistKKS[i-1]] * paras[0][i] 
        up_pre += df_const[assistKKS[i-1]] * paras[1][i] 

    paras  = loaded_model.conf_int(Config.confidence_second)
    d_pre2 = paras[0][0]
    up_pre2 = paras[1][0]
    
    for i in range(1,len(paras)):
        d_pre2 += df_const[assistKKS[i-1]] * paras[0][i] 
        up_pre2 += df_const[assistKKS[i-1]] * paras[1][i] 

    up_pre += (Config.K1*delta1-Config.B1)*up_pre

    up_pre2 += up_pre2 *(Config.K2*delta2-Config.B2)
    d_pre -=d_pre* (Config.K1*delta1-Config.B1)
    d_pre2-=d_pre2*(Config.K2*delta2-Config.B2)


    pred_interval = {"prediction":list(predictions.values),"first_upper":list(up_pre.values),\
        "first_lower":list(d_pre.values),"second_upper":list(up_pre2.values),\
            "second_lower":list(d_pre2.values)}
    #不带索引就 列值
    prediction_series = json.dumps(pred_interval)
    re = redis.StrictRedis(host=Config.redis_host,port=Config.redis_port,db=Config.redis_db,password=Config.redis_password)

    keys_redis = "evaluate_"+ str(evaluationId)+"_"+str(model_id) +"_"+ str(epochs) +"_"+ str(chunks)

    re.set(keys_redis,prediction_series)

    #修改模型的状态
    MODELS_STATUS[str(model_id)]['status'] = STATES.PG_WANCHENG

    message = {
        'status': True,
        'message': "评估完成",
        #'data':prediction_series
        "keys_redis": keys_redis,
        "times_start": times_start
    }
    logging.info("******evaluate_task finished modelid {} ,{},".format(model_id,keys_redis))
    #java 接口
    header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}  
    resp = requests.post(Config.java_host, \
                    data = json.dumps(message),\
                    headers= header) 



@app.route('/evaluate', methods=['POST'])
def evaluate():

    try:
        request_json = request.get_json()  
        model_id = request_json["modelId"]
        datasetUrl = request_json["datasetUrl"]
        mainKKS = request_json["mainKKS"]
        assistKKS = request_json["assistKKS"]        
        delta1 = request_json["firstConfidence"]
        delta2 = request_json["secondConfidence"]
        evaluationId = request_json["evaluationId"]
        epochs = request_json["epochs"]
        chunks = request_json["chunks"]

        #添加模型管理之后 应该是不需要再去磁盘读取模型文件的
        logging.info("******evaluating modelid {},".format(model_id))
        loaded_model = None

        if str(model_id) in MODELS_STATUS.keys() and MODELS_STATUS[str(model_id)]['status'] == STATES.XL_WANCHENG:
            loaded_model = MODELS_STATUS[str(model_id)]['model']
        elif str(model_id) not in MODELS_STATUS.keys():
            return bad_request(502)
        elif str(model_id) in MODELS_STATUS.keys() and MODELS_STATUS[str(model_id)]['status'] != STATES.XL_WANCHENG:
            return bad_request(506) #新定义了一个506错误 提示模型不是训练完成状态
        params  = loaded_model.params.index
        columns = list(params[1:])
        for col in columns:
            if col not in assistKKS:
                return(bad_request(401))

        filename = datasetUrl[datasetUrl.rindex('/') +1:-4]  
        local_path = os.path.join(pathcwd,'dataset/evaluate/' + str(model_id)+'/')
        if not os.path.exists(local_path):
            os.makedirs( local_path )
        
        #哪个是绝对路径 哪个是文件名
        local_path_csv = os.path.join(local_path,filename +'.csv')
        #filename_ = wget.download(datasetUrl, out=local_path)
        p=subprocess.Popen(['wget','-N',datasetUrl,'-P',local_path])
        if p.wait()==8:return(bad_request(505))
           
    except Exception as e:
        logging.info("******evaluating modelid {},excep:{}".format(model_id,e))
        raise e

    #修改模型状态 设置模型的第一置信度 和 第二置信度
    MODELS_STATUS[str(model_id)]['status'] = STATES.PINGGU_ZHONG
    MODELS_STATUS[str(model_id)]['firstConfidence'] = delta1
    MODELS_STATUS[str(model_id)]['secondConfidence'] = delta2

    

    executor.submit(evaluate_task,delta2,evaluationId,delta1,local_path_csv,assistKKS,model_id,loaded_model,epochs,chunks)
    

    message = {
        'status': True,
        'message': "评估开始",
        #'data':prediction_series

    }
    logging.info("******evaluating asycio modelid {} ,".format(model_id))


    responses = jsonify(message)
    
    responses.status_code = 200
    
    return (responses)


def evaluate_renew_task(delta2,evaluationId,delta1,local_path_csv,assistKKS,model_id,loaded_model,epochs,chunks):
    
    logging.info("******evaluate_renew_task  modelid {} ,".format(model_id))
    data = pd.read_csv(local_path_csv)
    X = data.loc[:,tuple(assistKKS)]
    #y = data.loc[:,mainKKS]
    X_const = sm.add_constant(X)
    times_start = data.iloc[:1,0].values[0]

    #df.insert(0,'const',1.0)
    
    #批量预测
    predictions = loaded_model.predict(X_const)
    df_const = X_const
    paras  = loaded_model.conf_int(Config.confidence)
    d_pre = paras[0][0]
    up_pre = paras[1][0]
    
    for i in range(1,len(paras)):
        d_pre += df_const[assistKKS[i-1]] * paras[0][i] 
        up_pre += df_const[assistKKS[i-1]] * paras[1][i] 

    paras  = loaded_model.conf_int(Config.confidence_second)
    d_pre2 = paras[0][0]
    up_pre2 = paras[1][0]
    
    for i in range(1,len(paras)):
        d_pre2 += df_const[assistKKS[i-1]] * paras[0][i] 
        up_pre2 += df_const[assistKKS[i-1]] * paras[1][i] 

    up_pre += (Config.K1*delta1-Config.B1)*up_pre

    up_pre2 += up_pre2 *(Config.K2*delta2-Config.B2)
    d_pre -=d_pre* (Config.K1*delta1-Config.B1)
    d_pre2-=d_pre2*(Config.K2*delta2-Config.B2)


    pred_interval = {"first_upper":list(up_pre.values),\
        "first_lower":list(d_pre.values),"second_upper":list(up_pre2.values),\
            "second_lower":list(d_pre2.values)}
    #不带索引就 列值
    prediction_series = json.dumps(pred_interval)
    re = redis.StrictRedis(host=Config.redis_host,port=Config.redis_port,db=Config.redis_db,password=Config.redis_password)

    keys_redis = "evaluate_"+ str(evaluationId)+"_"+str(model_id) +"_"+ str(epochs) +"_"+ str(chunks)

    re.set(keys_redis,prediction_series)

    message = {
        'status': True,
        'message': "重新评估完成",
        #'data':prediction_series
        "keys_redis": keys_redis,
        "times_start": times_start
    }
    logging.info("******evaluate_task finished modelid {} ,{},".format(model_id,keys_redis))
    #java 接口
    header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}  
    resp = requests.post(Config.java_host, \
                    data = json.dumps(message),\
                    headers= header) 



@app.route('/evaluate_renew', methods=['POST'])
def evaluate_renew():

    try:
        request_json = request.get_json()  
        model_id = request_json["modelId"]
        datasetUrl = request_json["datasetUrl"]
        mainKKS = request_json["mainKKS"]
        assistKKS = request_json["assistKKS"]        
        delta1 = request_json["firstConfidence"]
        delta2 = request_json["secondConfidence"]
        evaluationId = request_json["evaluationId"]
        epochs = request_json["epochs"]
        chunks = request_json["chunks"]

        clf = 'model.pkl'     
        
        logging.info("******evaluating modelid {},".format(model_id))
        loaded_model = None
        # with open('./model/' + str(model_id)+'/' + clf,'rb') as f:
        #     loaded_model = pickle.load(f)
        #如果是重新评估的话 则可以直接模型，不需要再判断模型的状态了
        loaded_model = MODELS_STATUS[str(model_id)]['model']
        params  = loaded_model.params.index
        columns = list(params[1:])
        for col in columns:
            if col not in assistKKS:
                return(bad_request(401))

        filename = datasetUrl[datasetUrl.rindex('/') +1:-4]  
        local_path = os.path.join(pathcwd,'dataset/evaluate/' + str(model_id)+'/')
        if not os.path.exists(local_path):
            os.makedirs( local_path )
        
        #哪个是绝对路径 哪个是文件名
        local_path_csv = os.path.join(local_path,filename +'.csv')
        #filename_ = wget.download(datasetUrl, out=local_path)
        p=subprocess.Popen(['wget','-N',datasetUrl,'-P',local_path])
        if p.wait()==8:return(bad_request(505))
           
    except Exception as e:
        logging.info("******evaluating modelid {},excep:{}".format(model_id,e))
        raise e
    #修改模型的状态 并更新第一二置信度
    MODELS_STATUS[str(model_id)]['status'] = STATES.PINGGU_ZHONG
    MODELS_STATUS[str(model_id)]['firstConfidence'] = delta1
    MODELS_STATUS[str(model_id)]['secondConfidence'] = delta2
    

    executor.submit(evaluate_task,delta2,evaluationId,delta1,local_path_csv,assistKKS,model_id,loaded_model,epochs,chunks)
    

    message = {
        'status': True,
        'message': "重新评估开始",
        #'data':prediction_series

    }
    logging.info("******evaluating renew asycio modelid {} ,".format(model_id))


    responses = jsonify(message)
    
    responses.status_code = 200
    
    return (responses)


'''
#这个接口的内容不是取消评估的内容
@app.route('/evaluate_cancel', methods=['POST'])
def evaluate_cancel():

    try:
        request_json = request.get_json()  
        model_id = request_json["modelId"]
        datasetUrl = request_json["datasetUrl"]
        mainKKS = request_json["mainKKS"]
        assistKKS = request_json["assistKKS"]        
        confidence = request_json["confidence"]
        confidence = 1 - confidence
        
        clf = 'model.pkl'     
        print("Loading the model...")
        loaded_model = None
        with open('./model/' + str(model_id)+'/' + clf,'rb') as f:
            loaded_model = pickle.load(f)
        params  = loaded_model.params.index
        columns = list(params[1:])
        for col in columns:
            if col not in assistKKS:
                return(bad_request(401))
                      
    except Exception as e:
        raise e
    
    print(datasetUrl)
    filename = datasetUrl[datasetUrl.rindex('/') +1:-4]  
    local_path = './dataset/evaluate/' + str(model_id)+'/'
    if not os.path.exists(local_path):
        os.makedirs( local_path )
    
    #哪个是绝对路径 哪个是文件名
    local_path = local_path+filename +'.csv'
    filename_ = wget.download(datasetUrl, out=local_path)

    data = pd.read_csv(local_path)
    X = data.loc[:,tuple(assistKKS)]
    y = data.loc[:,mainKKS]
    X_const = sm.add_constant(X)

    #df.insert(0,'const',1.0)
    
    #批量预测
    predictions = loaded_model.predict(X_const)
    
    #批量区间预测  这样写可以吗
    paras  = loaded_model.conf_int(confidence)
    d_pre = paras[0][0]
    up_pre = paras[1][0]
    
    for i in range(1,len(paras)):
        d_pre += X_const[assistKKS[i-1]] * paras[0][i] 
        up_pre += X_const[assistKKS[i-1]] * paras[1][i] 
    
    print(y)
    print(predictions)
    print(d_pre)
    print(up_pre)
    pred_interval = {"y_true":y.values,"predict":predictions.values,"upper":up_pre.values,"lower":d_pre.values}
    #prediction_series = pd.DataFrame(pred_interval,orient='records')
    prediction_series = pd.DataFrame(pred_interval)
    prediction_series = prediction_series.to_json()
    print(pred_interval)
    print(prediction_series)
    message = {
        'status': True,
        'message': "请求成功",
        'data':prediction_series
    }
    
    responses = jsonify(message)
    
    responses.status_code = 200
    
    return (responses)
 
'''

#取消评估接口 参数为modelId
@app.route('/evaluate_cancel',methods=['post'])
def evaluate_cancel():
    try:
        request_json = request.get_json()
        print(request_json)
        model_id = request_json['modelId']
        firstConfidence = request_json['firstConfidence']
        secondConfidence = request_json['secondConfidence']

        #删除评估文件
        local_path = os.path.join(pathcwd, 'dataset/evaluate/' + str(model_id) + '/')
        if os.path.exists(local_path):
            fileList = os.listdir(local_path)
            for file in fileList:
                filepath = os.path.join(local_path,filepath)
                if os.path.isfile(filepath):
                    os.remove(filepath)
            shutil.rmtree(local_path)

        #修改模型的状态为训练完成 并将模型的第一二置信度更新
        if str(model_id) not in MODELS_STATUS.keys():
            return bad_request(502)
        else:
            MODELS_STATUS[str(model_id)]['status'] = STATES.XL_WANCHENG
            MODELS_STATUS[str(model_id)]['firstConfidence'] = firstConfidence
            MODELS_STATUS[str(model_id)]['secondConfidence'] = secondConfidence

    except Exception as e:
        logging.info("******cancel evaluating modelid {},excep:{}".format(model_id,e))
        raise e
    message = {
                'status': True,
                'message': '-->模型取消评估',
        }
    resp = jsonify(message)
    resp.status_code = 200
    return resp

@app.route('/evaluate_interval', methods=['POST'])
def evaluate_interval():

    try:
        request_json = request.get_json()  
        model_id = request_json["modelId"]
        
        clf = 'model.pkl'     
        print("Loading the model...")
        loaded_model = None
        with open('./model/' + str(model_id)+'/' + clf,'rb') as f:
            loaded_model = pickle.load(f)
        params  = loaded_model.params.index
        columns = list(params[1:])
        for col in columns:
            if col not in request_json.keys():
                return(bad_request(401))
                      
    except Exception as e:
        raise e
    
    data=[]
    for i in range(1,len(params)):
        data.append(request_json[params[i]]) 
    
    if len(data)==0 or '' in data:
        return(bad_request())
    else:
        data =  np.expand_dims(data,1)  
        df = pd.DataFrame(dict(zip(columns,data)))   
        print("The model has been loaded...doing predictions now...")
        
        df.insert(0,'const',1.0)
        df_const = df
        predictions = loaded_model.predict(df_const)
        
        paras  = loaded_model.conf_int()
        d_pre = paras[0][0]
        up_pre = paras[1][0]
        
        for i in range(1,len(paras)):
            d_pre += df_const.loc[0][i] * paras[0][i] 
            up_pre += df_const.loc[0][i] * paras[1][i] 

        pred_interval = {"prediction":predictions.loc[0],"upper":up_pre,"lower":d_pre}
        prediction_series = pd.DataFrame(pred_interval,index=[0])
          
        message = {
			'status': True,
			'message': "请求成功",
            'data':pred_interval
	    }
        
        responses = jsonify(message)
        
        responses.status_code = 200
    
    return (responses)
 



def train_task(train_task,local_path,assistKKS,mainKKS,model_id,local_path_model):
    data = pd.read_csv(local_path)

    X = data.loc[:,tuple(assistKKS)]

    y = data.loc[:,mainKKS]

    X_const = sm.add_constant(X)
    model = sm.OLS(y,X_const ).fit() 
    #print(model.summary())

    #model.conf_int(0.05)
    
    header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}
        
    #训练完成之后 修改模型状态为训练完成，将训练完成的模型保存下来
    if str(model_id) in MODELS_STATUS.keys(): #如果不在模型状态管理里边 说明模型还没有训练完 就已经被取消了
        if not os.path.exists(local_path_model): os.makedirs(local_path_model)
        filepath = local_path_model + 'model.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        MODELS_STATUS[str(model_id)]['status'] = STATES.XL_WANCHENG
        MODELS_STATUS[str(model_id)]['model'] = model

    message = {
        'status': True,
        'message': "训练完成",
        #'data':prediction_series
        "model_id": model_id
    }
    resp = requests.post(Config.java_host, \
                    data = json.dumps(message),\
                    headers= header)


@app.route('/train', methods=['POST'])
def train():
    try:
        request_json = request.get_json()
        
        model_id = request_json["modelId"]
        datasetUrl = request_json["dataUrl"]
        mainKKS = request_json["mainKKS"]
        
        #列表
        assistKKS = request_json["assistKKS"]
        
        #MODELS_MAP[str(model_id)]["status"] = STATES.training
        local_path_data = './dataset/train/' + str(model_id)+'/'
        local_path_model = './model/train/' + str(model_id)+'/'

        if not os.path.exists(local_path_data):   os.makedirs( local_path_data )
        if not os.path.exists(local_path_model):    os.makedirs( local_path_model )  
        
        p= subprocess.Popen(['wget','-N',datasetUrl,'-P',local_path_data])
        if p.wait()==8:return(bad_request(505))
        filename = datasetUrl[datasetUrl.rindex('/') +1:-4] 
        local_path = os.path.join(pathcwd,'dataset/train/' + str(model_id)+'/'+filename + '.csv')
    except Exception as e:
        logging.info("******training modelid {},excep:{}".format(model_id,e))
        raise e
    # print(datasetUrl)
    # filename = datasetUrl[datasetUrl.rindex('/') +1:-4]  
    # local_path = './dataset/' + str(model_id)+'/'
    # if not os.path.exists(local_path):
    #     os.makedirs( local_path )
    
    # #哪个是绝对路径 哪个是文件名
    # local_path = local_path+filename +'.csv'
    # filename_ = wget.download(datasetUrl, out=local_path)
    # except Exception as e:
    #     logging.info("******training modelid {},excep:{}".format(model_id,e))
    #     raise e
    MODELS_MAP = {
        "status": STATES.XUNLIAN_ZHONG, #设置模型的状态为训练中
        "model": None, #此时的模型为空的
        "model_id": model_id, # 模型id为从页面传过来的id
        "firstConfidence": 0.95,# 此时的第一置信度和第二置信度都初始默认值
        "secondConfidence": 0.98
    }  # 模型状态管理内层字典
    MODELS_STATUS[str(model_id)] = MODELS_MAP

    executor.submit(train_task,local_path,assistKKS,mainKKS,model_id,local_path_model)
  
 
        
    #MODELS_MAP[str(model_id)]["status"] = STATES.training_finish
    
    message = {
			'status': True,
			'message': '-->模型开始训练',
	}
    resp = jsonify(message)
    resp.status_code = 200
    return resp


#取消训练接口 参数为取消训练的模型id:model_id
@app.route('/train_cancel', methods= ['post'])
def train_cancel():
    try:
        request_json = request.get_json()
        model_id = request_json["modelId"]
        MODELS_STATUS.pop(str[model_id]) #直接将模型的相关信息移除
        # 将对应的model_id 的训练文件都删除
        data_path = os.path.join(pathcwd, 'dataset/train/' + str(model_id) + '/')
        print(data_path)
        filelist = os.listdir(data_path)
        for filename in filelist:
            filePath = os.path.join(data_path,filename)
            if os.path.isfile(filePath):
                os.remove(filePath)
        # os.remove(data_path) #权限问题 拒绝访问
        shutil.rmtree(data_path)

        # 这里应该加上停止训练进程的代码 但是目前没有找到好的解决办法

    except Exception as e:
        logging.info("******train cancel modelid {},excep:{}".format(model_id,e))
        raise e
    message = {
                'status': True,
                'message': '-->模型取消训练',
        }
    resp = jsonify(message)
    resp.status_code = 200
    return resp

'''
@app.route('/train_batch', methods=['POST'])
def train_batch():
    
    request_json = request.get_json()
    
    #model_id = request_json["modelId"]
    datasetUrl_list = request_json["datasetUrlList"]
    #mainKKS = request_json["mainKKS"]
    
    #列表
    #assistKKS = request_json["assistKKS"]
    
    #MODELS_MAP[str(model_id)]["status"] = STATES.training
    
    for datasetUrl in datasetUrl_list:
        print(datasetUrl)

        filename = datasetUrl[datasetUrl.rindex('/') +1:-4]

        model_id = eval(filename.split('_')[1]) 
        logging.info("***start train modelid: {}".format(model_id))
        local_path_data = './dataset/train/' + str(model_id)+'/'
        local_path_model = './model/train/' + str(model_id)+'/'

        if not os.path.exists(local_path_data):   os.makedirs( local_path_data )
        if not os.path.exists(local_path_model):    os.makedirs( local_path_model )  
        
        p= subprocess.Popen(['wget','-N',datasetUrl,'-P',local_path_data])
        
        if p.wait()==8:return(bad_request(505))
        local_path = os.path.join(pathcwd,'dataset/train/' + str(model_id)+'/', filename + '.csv')
        
        logging.info("***start read data: {}".format(model_id))
        data = pd.read_csv(local_path)
        logging.info("***start train model: {}".format(model_id))

        data = data.dropna()
        for col in data.columns[1:]:
            data = data[np.abs(data[col]-data[col].mean()) <= (3*data[col].std())]
            
        assistKKS = data.columns[2:]
        mainKKS = data.columns[1]

        X = data.loc[:,tuple(assistKKS)]

        y = data.loc[:,mainKKS]

        X_const = sm.add_constant(X)
        model = sm.OLS(y,X_const ).fit() 
        
        modelpath = local_path_model+'model.pkl'
        with open(modelpath, 'wb') as f:
            pickle.dump(model, f)
        
        logging.info("***finish modelid: {}".format(model_id))
    message = {
			'status': True,
			'message': request.url+'-->模型训练成功',
	}

    resp = jsonify(message)
    resp.status_code = 200
    return resp
'''

# 批量训练执行的任务，传进来的参数主要的就是correctModelList：字典的list {“local_path”: str,"assistKKS": list,"mainKKS": str,"model_id": str,"local_path_model": str}
def train_batch_task(train_batch_task,correctModelList):
    model_finish = [] #训练成功的modelId的list
    model_fail = [] #训练失败的modelId的list
    for correctModel in correctModelList: #循环执行单个训练任务
        model_id = correctModel["model_id"]
        try:
            train_task('train_task',correctModel["local_path"],correctModel["assistKKS"],correctModel["mainKKS"],correctModel["model_id"],correctModel["local_path_model"])
            model_finish.append(model_id)
        except Exception as e:
            logging.info("******training modelid {},excep:{}".format(model_id, e))
            MODELS_STATUS.pop(str(model_id))
            model_fail.append(model_id)
    header = {'Content-Type': 'application/json', \
              'Accept': 'application/json'}
    message = {
        'status': True,
        'message': "训练完成",
        #'data':prediction_series
        'data':{
            'model_finish': model_finish,
            'model_fail': model_fail
        }
    }
    resp = requests.post(Config.java_host, \
                    data = json.dumps(message),\
                    headers= header)


@app.route('/train_batch',methods=['post'])
def train_batch():
    try:
        request_json = request.get_json() #获取传过来的所有参数

        para_list = request_json["paraList"] #
        datasetUrl_list = request_json["datasetUrlList"]
        # paraListLength = len(para_list) #获取传过来的list的长度
        # datasetUrListLength = len(datasetUrl_list) #获取传过来的urlList的长度

        errorModelIdList = []  # 有问题的modelId的数组，用来返还给前端展示
        correctModelIdList = []  # 正确的的modelId的数组，用来返还给前端展示
        correctModelList = [] #正确的model的信息的list list中的元素是字典 {“local_path”: str,"assistKKS": list,"mainKKS": str,"model_id": str,"local_path_model": str}


        for para in para_list:
            count = len(datasetUrl_list) #用来计数，判断datasetUrl_list是不是已经遍历到了最后了
            para_modelId = para["modelId"]#从字典中获取model_id
            for datasetUrl in datasetUrl_list:  # 对url进行解析取出其中的model_id
                count-=1    #每次循环都自动减一，当count==0时 说明循环到了最后一个元素
                filename = datasetUrl[datasetUrl.rindex('/') + 1:-4]
                data_modelId = eval(filename.split('_')[1])  # 把model_id切出来
                if para_modelId == data_modelId: #当modelId相等时可以进行下一步的判断
                    model_id = para_modelId
                    # para_dic = list(para.values())[0]
                    mainKKS = para["mainKKS"]  # 取出主测点
                    assistKKS = para["assistKKS"]  # 取出相关测点

                    #需要先将文件下载下来
                    local_path_data = './dataset/train/' + str(model_id) + '/'
                    if not os.path.exists(local_path_data):   os.makedirs(local_path_data)
                    p = subprocess.Popen(['wget', '-N', datasetUrl, '-P', local_path_data])
                    #获取数据集文件出现错误，就可以认为这个模型是有问题的
                    if p.wait() == 8:
                        errorModelIdList.append(model_id)
                        datasetUrl_list.remove(datasetUrl) #每次都将有问题的Url去除，下次循环就可以少循环一次
                        break #此模型出现错误，则可以直接退出这次循环
                    local_path = os.path.join(pathcwd, 'dataset/train/' + str(data_modelId) + '/', filename + '.csv')
                    data = pd.read_csv(local_path)
                    columns = list(data.columns)
                    columns.pop(0) #将第一个时间戳的标签删除出去
                    data_mainKKS = columns[1]
                    if data_mainKKS == mainKKS: #当判断主测点相同时 进一步判断相关测点是不是相同
                        #首先获取相关测点的数量 判断相关测点的数量是不是一样的,数量不一样肯定是有问题的
                        columns.pop(0) #将主测点的标识删除出去,剩下的都是相关测点的数据标识
                        para_assistKKSLength = len(assistKKS)
                        data_assistKKSLength = len(columns) # -2 是因为第一个为时间戳 第二个为主测点
                        if(para_assistKKSLength != data_assistKKSLength):
                            errorModelIdList.append(model_id)
                            datasetUrl_list.remove(datasetUrl)
                            break #出现错误直接退出循环
                        else: #当相关测点的数量相等的时候，就进行双向判断 判断两个列表是不是所有的元素都是相同的
                            temp_list = [temp for temp in assistKKS if temp not in columns]
                            if len(temp_list) != 0: #说明有元素是在assistKKS中存在而columns不存在，即相关测点对应不上
                                errorModelIdList.append(model_id)
                                datasetUrl_list.remove(datasetUrl)
                                break
                            else: #判断到此就能确定数据校验已经通过 即模型id能对应上 主测点能对应上 相关测点能对应上
                                local_path_model = './model/train/' + str(model_id)+'/'
                                model_data = {
                                    "local_path": local_path,
                                    "assistKKS": assistKKS,
                                    "mainKKS": mainKKS,
                                    "model_id": model_id,
                                    "local_path_model": local_path_model
                                }
                                correctModelIdList.append(model_id)
                                correctModelList.append(model_data) #将正确的加到正确的list中 之后可以将当前datasetUrl移除
                                datasetUrl_list.remove(datasetUrl)
                                break
                    else:
                        errorModelIdList.append(model_id) #主测点不相同时，则认为这个model是有问题的
                        datasetUrl_list.remove(datasetUrl)
                        break
                if count == 0 and datasetUrl == datasetUrl_list[-1]: #此时说明已经将datasetUrl_list循环到了最后一个元素，并且不匹配，则此model有问题
                    errorModelIdList.append(para_modelId)
                    break #这里break没有什么意义 本来就已经到了末尾了
    except Exception as e:
        logging.info("****** batch training excep:{}".format(e))
        raise e
    for correctModelId in correctModelIdList:
        model_id = correctModelId
        MODELS_MAP = {
            "status": STATES.XUNLIAN_ZHONG,  # 设置模型的状态为训练中
            "model": None,  # 此时的模型为空的
            "model_id": model_id,  # 模型id为从页面传过来的id
            "firstConfidence": 0.95,  # 此时的第一置信度和第二置信度都初始默认值
            "secondConfidence": 0.98
        }  # 模型状态管理内层字典
        MODELS_STATUS[str(model_id)] = MODELS_MAP

    executor.submit(train_batch_task,correctModelList) #开始异步执行训练过程


    message = {
        'status': True,
        'message': '-->模型开始训练',
        'data': {
            'correctModelIdList': correctModelIdList,
            'errorModelIdList': errorModelIdList
        }
    }
    resp = jsonify(message)
    resp.status_code = 200
    return resp


@app.route('/confidence', methods=['POST'])
def confidence():
    # lmm.conf_int(0.05)  默认 0.05
    # 0.05   即 0.025-0.975 即涵盖95%  置信度95
    # 0.1  即 0.05-0.95  即涵盖90%    置信度90
    # 0.02 即  0.01- 0.99 即涵盖98%   置信度 98
    request_json = request.get_json()
        
    model_id = request_json["modelId"]
    confidence = request_json["confidence"]
    confidence = 1 - confidence
    #是在什么时候修改的，是在评估，是在发布后，
    #置信度，预测还要带上  
    loaded_model = None
    if str(model_id) in MODELS_STATUS.keys():
        loaded_model = MODELS_STATUS[str(model_id)]['model']


    loaded_model.conf_int(0.05)


if __name__ == "__main__":
    # 将host设置为0.0.0.0，则外网用户也可以访问到这个服务
    app.run(host="0.0.0.0", port=8385, debug=True)
