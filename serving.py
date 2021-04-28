# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np 
from flask import Flask, jsonify, request
#import paramiko
import pickle
import statsmodels
from statsmodels.sandbox.regression.predstd import wls_prediction_std
#import xlrd
from matplotlib import pyplot as plt
import statsmodels.api as sm #最小二乘
from statsmodels.formula.api import ols #加载ols模型
import wget
import json
import requests
import subprocess
#from config import Config
import logging 
from enum import Enum
import redis 
#import happybase
from concurrent.futures import ThreadPoolExecutor
from config import Config 

executor = ThreadPoolExecutor(8)

pathcwd = os.path.dirname(__file__)
app = Flask(__name__)

class States(Enum):
    
    XUNLIAN_ZHONG = 1
    XL_WANCHENG = 2
    
    PINGGU_ZHONG = 3
    PG_CX_ZHONG = 4
    PG_WANCHENG= 5 
    PG_CX_WANCHENG = 6 
    PG_QUXIAO = 7
    
    FABU = 8
    FB_QUXIAO = 9  
    
header = {'Content-Type': 'application/json','Accept': 'application/json'} 

MODELS_STATUS = {}
logpath = 'log/serving.std.out'
logging.basicConfig(filename=logpath,filemode='a',format='%(asctime)s %(name)s:%(levelname)s:%(message)s',datefmt="%d-%m-%Y \
    %H:%M:%S",level=logging.DEBUG)
'''
#STATES_ENUM ={"初始化":0,"待训练":1,"训练中":2,'待发布':3,'已发布':4,"训练失败":5}
# 状态管理，没有取消训练， 但有取消训练接口， 就直接remove，key，
状态管理 有取消评估，接口，  状态置 训练完成  模型内存保存,异步执行的 pid 也可以吧该训练线程 kill
状态管理 有取消发布，接口，  状态置 训练完成  模型内存保存，异步执行的 pid 也可以吧该训练线程 kill

"status":States.XUNLIAN_ZHONG,"model":，"modelid":model_id,
            "firstConfidence":0.95，"secondConfidence":0.98,"train_future":,"evaluate_future":
MODELS_STATUS = {
"id":{"status":,"model":，"modelid":,"firstConfidence"，"secondConfidence":,}
'''

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
 
        if (delta1>=0.98 or delta1 <0.95 or delta2>=1.0 or delta2 <0.98):
            logging.info("******predicting modelid {},excp:{}".format(model_id,"置信度异常"))
            return (bad_request(504)) 
         
        if MODELS_STATUS[str(model_id)]["status"] != States.FABU : return(bad_request(506))
        
        if str(model_id) not in MODELS_STATUS.keys():return(bad_request(502))
        loaded_model = MODELS_STATUS[str(model_id)]["model"] 
        
            
        #logging.info("******predicting summary {},".format(loaded_model.summary()  )) 
        params  = loaded_model.params.index
        columns = list(params[1:])
        for col in columns:
            if col not in request_json.keys():
                logging.info("******predicting modelid {},excp:{}".format(model_id,"相关测点异常"))
                return(bad_request(501))
                      

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
    except Exception as e:
        logging.info("******predicting modelid {},excp:{}".format(model_id,e))
        raise e 
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
        message.update( {
			'message': '-->模型未发布状态预测',
	})
        resp = jsonify(message)
        resp.status_code =506
    elif error ==507:
        message.update( {
			'message': '-->模型训练数据少于一天',
	})
        resp = jsonify(message)
        resp.status_code =507      
    elif error ==508:
        message.update( {
			'message': '-->模型训练数据为空',
	})
        resp = jsonify(message)
        resp.status_code =508  
    return resp



@app.route('/publish', methods=['POST'])
def publish():
    request_json = request.get_json()
    if 'modelId' not in request_json.keys:
        return(bad_request(400))
    
    model_id = request_json["modelId"]   
    
    if str(model_id) not in MODELS_STATUS.keys():return(bad_request(502))
        
    MODELS_STATUS[str(model_id)]["status"] = States.FABU    
    logging.info("******publish  modelid {} ,".format(model_id))
    
    message = {
			'status': True,
			'message': '-->模型发布成功',
	}
    resp = jsonify(message)
    resp.status_code = 200
    return resp




@app.route('/publish_cancel', methods=['POST'])
def publish_cancel():
    request_json = request.get_json()
    if 'modelId' not in request_json.keys:
        return(bad_request(400))
    
    model_id = request_json["modelId"]   
    
    if str(model_id) not in MODELS_STATUS.keys():return(bad_request(502))
        
    MODELS_STATUS[str(model_id)]["status"] = States.XL_WANCHENG   
     
    logging.info("******publish_cancel  modelid {} ,".format(model_id))
    
    message = {
			'status': True,
			'message': '-->模型取消发布成功',
	}
    resp = jsonify(message)
    resp.status_code = 200
    return resp



def evaluate_task(delta2,evaluationId,delta1,local_path_csv,assistKKS,model_id,loaded_model,epochs,chunks):
    
    try:
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
    except Exception as e:
        logging.info("******evaluatet task modelid {},excep:{}".format(model_id,e))
        message = {
        'status': False,
        'message': "评估中异常excep: " + str(e),
        #'data':prediction_series
        "model_id": model_id

        }
        resp = requests.post(Config.java_host_evaluate, \
                    data = json.dumps(message),\
                    headers= header)

        raise e
    
    message = {
        'status': True,
        'message': "评估完成",
        #'data':prediction_series
        "keys_redis": keys_redis,
        "times_start": times_start
    }
    logging.info("******evaluate_task finished modelid {} ,{},".format(model_id,keys_redis))

    resp = requests.post(Config.java_host_evaluate, \
                    data = json.dumps(message),\
                    headers= header) 



@app.route('/evaluate', methods=['POST'])
def evaluate():

    try:
        request_json = request.get_json()  
        model_id = request_json["modelId"]
        datasetUrl = request_json["dataUrl"]
        mainKKS = request_json["mainKKS"]
        assistKKS = request_json["assistKKS"]        
        delta1 = request_json["firstConfidence"]
        delta2 = request_json["secondConfidence"]
        evaluationId = request_json["evaluationId"]
        epochs = request_json["epochs"]
        chunks = request_json["chunks"]

        clf = 'model.pkl'     
        
        logging.info("******evaluating modelid {},".format(model_id))
        
        MODELS_STATUS[str(model_id)]["firstConfidence"] = delta1
        MODELS_STATUS[str(model_id)]["secondConfidence"] = delta2
            
        if str(model_id) not in MODELS_STATUS.keys():return(bad_request(502))
        loaded_model = MODELS_STATUS[str(model_id)]["model"]      
        
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
        p = subprocess.Popen(['wget','-N',datasetUrl,'-P',local_path])
        if p.wait()==8:return(bad_request(505))
           
    except Exception as e:
        logging.info("******evaluating modelid {},excep:{}".format(model_id,e))
        message = {
        'status': False,
        'message': "评估预处理异常excep: " + str(e),
        #'data':prediction_series
        "model_id": model_id

        }
        resp = requests.post(Config.java_host_evaluate, \
                    data = json.dumps(message),\
                    headers= header)
        raise e
    
    #state = {"status":States.PINGGU_ZHONG,"modelid":model_id,"firstConfidence":0.95,"secondConfidence":0.98}
    MODELS_STATUS[str(model_id)]["status"] = States.PINGGU_ZHONG
        
    evaluate_future =  executor.submit(evaluate_task,delta2,evaluationId,delta1,local_path_csv,assistKKS,model_id,loaded_model,epochs,chunks)
    MODELS_STATUS[str(model_id)+"_future"]["evaluate"]=evaluate_future 


    message = {
        'status': True,
        'message': "评估开始",
        #'data':prediction_series

    }
    logging.info("******evaluating asycio modelid {} ,".format(model_id))
    responses = jsonify(message) 
    responses.status_code = 200
    
    return (responses)

@app.route('/evaluate_cancel', methods=['POST'])
def evaluate_cancel():

    request_json = request.get_json()       
    model_id = request_json["modelId"]
    delta1 = request_json["firstConfidence"]
    delta2 = request_json["secondConfidence"]
        
        #若已经运行，线程有取消接口，若已经运行，就取消不了了，，那就会回调java，所以，
        #所以这边不能完全保证，不再回调， 还是会回调，
        #所以java那边，要确保，若页面取消了，状态就改为取消， 那就我这边即使回调显示训练完成，也不要考虑了，
    evaluate_future = MODELS_STATUS[str(model_id)+"_future"]["evaluate"]
    MODELS_STATUS[str(model_id)]["status"] = States.XL_WANCHENG
    MODELS_STATUS[str(model_id)]["firstConfidence"] = delta1
    MODELS_STATUS[str(model_id)]["secondConfidence"] = delta2
    
    if evaluate_future.cancel():
        message = {
                'status': True,
                'message': '-->模型评估取消成功',
        }
        logging.info("******evaluate cancel modelid {} success".format(model_id))
    else:
        message = {
                'status': False,
                'message': '-->模型评估取消失败',
        }
        logging.info("******evaluate cancel modelid {} failed".format(model_id))
    resp = jsonify(message)
    resp.status_code = 200
    return resp




def evaluate_renew_task(delta2,evaluationId,delta1,local_path_csv,assistKKS,model_id,loaded_model,epochs,chunks):
    
    try:
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
    except Exception as e:
        logging.info("******evaluating renew modelid {},excep:{}".format(model_id,e))
        message = {
        'status': False,
        'message': "重新评估中异常excep: " + str(e),
        #'data':prediction_series
        "model_id": model_id

        }
        resp = requests.post(Config.java_host_evaluate, \
                    data = json.dumps(message),\
                    headers= header)

        raise e

    message = {
        'status': True,
        'message': "重新评估完成",
        #'data':prediction_series
        "keys_redis": keys_redis,
        "times_start": times_start
    }
    logging.info("******evaluate_task finished modelid {} ,{},".format(model_id,keys_redis))

    resp = requests.post(Config.java_host_evaluate, \
                    data = json.dumps(message),\
                    headers= header) 




@app.route('/evaluate_renew', methods=['POST'])
def evaluate_renew():

    try:
        request_json = request.get_json()  
        model_id = request_json["modelId"]
        datasetUrl = request_json["dataUrl"]
        mainKKS = request_json["mainKKS"]
        assistKKS = request_json["assistKKS"]        
        delta1 = request_json["firstConfidence"]
        delta2 = request_json["secondConfidence"]
        evaluationId = request_json["evaluationId"]
        epochs = request_json["epochs"]
        chunks = request_json["chunks"]

        clf = 'model.pkl'     
        
        logging.info("******evaluating modelid {},".format(model_id))
        
        MODELS_STATUS[str(model_id)]["firstConfidence"] = delta1
        MODELS_STATUS[str(model_id)]["secondConfidence"] = delta2
        
        if str(model_id) not in MODELS_STATUS.keys():return(bad_request(502))
        loaded_model = MODELS_STATUS[str(model_id)]["model"]
        
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
        message = {
        'status': False,
        'message': "重新评估预处理异常excep: " + str(e),
        "model_id": model_id

        }
        resp = requests.post(Config.java_host_evaluate, \
                    data = json.dumps(message),\
                    headers= header)
        raise e
    
    MODELS_STATUS[str(model_id)]["status"] = States.PG_CX_ZHONG
    
    evaluate_future =  executor.submit(evaluate_task,delta2,evaluationId,delta1,local_path_csv,assistKKS,model_id,loaded_model,epochs,chunks)
    MODELS_STATUS[str(model_id)+"_future"]["evaluate"]=evaluate_future 
    
    message = {
        'status': True,
        'message': "重新评估开始",
        #'data':prediction_series
    }
    logging.info("******evaluating renew asycio modelid {} ,".format(model_id))
    responses = jsonify(message)  
    responses.status_code = 200
    
    return (responses)



def train_task(state,local_path_data,assistKKS,mainKKS,model_id,local_path_model):
    
    try:
       
        data = pd.read_csv(local_path_data)
        if(len(data)==0):
            message = {
            'status': False,
            'message': "训练数据为空",
            "model_id": model_id
    
            }
            resp = requests.post(Config.java_host_train,data = json.dumps(message),
                        headers= header)
            #continue
            #return(bad_request(508))
            #break
            #return 
            raise ValueError("训练数据为空")
        if(len(data)<=1*86400):
            message = {
            'status': False,
            'message': "训练数据至少一天",
            "model_id": model_id
    
            }
            resp = requests.post(Config.java_host_train,data = json.dumps(message),
                        headers= header)          
            #return(bad_request(507))
            raise ValueError("训练数据至少一天")
        
        X = data.loc[:,tuple(assistKKS)]

        y = data.loc[:,mainKKS]

        X_const = sm.add_constant(X)
        model = sm.OLS(y,X_const ).fit() 
        
        #model.conf_int(0.05)
        logging.info("******train finished modelid {} ".format(model_id))
 

        filepath = local_path_model+'model.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    
    except Exception as e:
        logging.info("******training modelid {},excep:{}".format(model_id,e))
        message = {
        'status': False,
        'message': "训练中异常excep: " + str(e),
        #'data':prediction_series
        "model_id": model_id
 
        }
        resp = requests.post(Config.java_host_train, \
                    data = json.dumps(message),\
                    headers= header)
        raise e
    
    
    logging.info("******train finished modelid {} saved".format(model_id))
    state["model"] =   model
    state["status"] = States.XL_WANCHENG
    #如果已有key呢，会直接覆盖吗，还是要先remove 我记得
    MODELS_STATUS[str(model_id)] = state 
    #state = {'modelId':1,"state":0}
    #java 接口
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
        path_data = './dataset/train/' + str(model_id)+'/'
        path_model = './model/train/' + str(model_id)+'/'

        if not os.path.exists(path_data):   os.makedirs( path_data )
        if not os.path.exists(path_model):    os.makedirs( path_model )  
        
        p= subprocess.Popen(['wget','-N',datasetUrl,'-P',path_data])
        if p.wait()==8:return(bad_request(505))
        filename = datasetUrl[datasetUrl.rindex('/') +1:-4] 
        local_path_data = os.path.join(pathcwd,'dataset/train/' + str(model_id)+'/'+filename + '.csv')
        
        local_path_model = os.path.join(pathcwd,'model/train/' + str(model_id)+'/')
    except Exception as e:
        logging.info("******training modelid {},excep:{}".format(model_id,e))
        message = {
        'status': False,
        'message': "训练预处理异常excep: " + str(e),
        #'data':prediction_series
        "model_id": model_id

        }
        resp = requests.post(Config.java_host_train, \
                    data = json.dumps(message),\
                    headers= header)
        raise e
    # print(datasetUrl)
    # filename = datasetUrl[datasetUrl.rindex('/') +1:-4]  
    # local_path = './dataset/' + str(model_id)+'/'
    # if not os.path.exists(local_path):
    #     os.makedirs( local_path )
    
    # #哪个是绝对路径 哪个是文件名
    # local_path = local_path+filename +'.csv'
    # filename_ = wget.download(datasetUrl, out=local_path)
    except Exception as e:
        logging.info("******training modelid {},excep:{}".format(model_id,e))
        raise e

    #state = dict()
    state = {"status":States.XUNLIAN_ZHONG,"modelid":model_id,"firstConfidence":0.95,"secondConfidence":0.98}
        
    train_future = executor.submit(train_task,state,local_path_data,assistKKS,mainKKS,model_id,local_path_model)
    MODELS_STATUS[str(model_id)+"_future"] = {}
    MODELS_STATUS[str(model_id)+"_future"]["train"]=train_future
        
    #MODELS_MAP[str(model_id)]["status"] = STATES.training_finish
    
    message = {
			'status': True,
			'message': '-->模型开始训练',
	}
    resp = jsonify(message)
    resp.status_code = 200
    return resp



@app.route('/train_cancel', methods=['POST'])
def train_cancel():

    request_json = request.get_json()       
    model_id = request_json["modelId"]
        
        #若已经运行，线程有取消接口，若已经运行，就取消不了了，，那就会回调java，所以，
        #所以这边不能完全保证，不再回调， 还是会回调，
        #所以java那边，要确保，若页面取消了，状态就改为取消， 那就我这边即使回调显示训练完成，也不要考虑了，
    train_future = MODELS_STATUS[str(model_id)+"_"+"future"]["train"]
    MODELS_STATUS[str(model_id)]["status"] = States.XL_WANCHENG
    
    del MODELS_STATUS[str(model_id)] 
    del MODELS_STATUS[str(model_id)+"_"+"future"] 
    
    if train_future.cancel():
        message = {
                'status': True,
                'message': '-->模型训练取消成功',
        }
        logging.info("******train cancel modelid {} success".format(model_id))
    else:
        message = {
                'status': False,
                'message': '-->模型训练取消失败',
        }
        logging.info("******train cancel modelid {} failed".format(model_id))
    resp = jsonify(message)
    resp.status_code = 200
    return resp


def train_batch_task(modelIdKKS,datasetUrlList):
    
    try:
        result_bools = [False] * len(datasetUrlList)
        result_ids = []
        for i in range(len(datasetUrlList)):
            datasetUrl = datasetUrlList[i]
            print(datasetUrl)

            filename = datasetUrl[datasetUrl.rindex('/') +1:-4]

            model_id = eval(filename.split('_')[1]) 
            result_ids.append(model_id)
            state = {"status":States.XUNLIAN_ZHONG,"modelid":model_id,"firstConfidence":0.95,"secondConfidence":0.98}
            
            logging.info("***start trainbatch modelid: {}".format(model_id))
            local_path_data = './dataset/train/' + str(model_id)+'/'
            local_path_model = './model/train/' + str(model_id)+'/'
              
            if not os.path.exists(local_path_data):   os.makedirs( local_path_data )
            if not os.path.exists(local_path_model):    os.makedirs( local_path_model )  
            
            p= subprocess.Popen(['wget','-N',datasetUrl,'-P',local_path_data])
            
            if p.wait()==8:return(bad_request(505))
            local_path = os.path.join(pathcwd,'dataset/train/' + str(model_id)+'/', filename + '.csv')
                    
            data = pd.read_csv(local_path)

            data = data.dropna()
            for col in data.columns[1:]:
                data = data[np.abs(data[col]-data[col].mean()) <= (3*data[col].std())]
                
            assistKKS = data.columns[2:]
            mainKKS = data.columns[1]

            #双向校验
            if str(model_id) not in modelIdKKS.keys():continue
            
            kks = modelIdKKS[str(model_id)]
            if kks["mainKKS"] != mainKKS:continue
            
            A2B = [True if i in list(assistKKS) else False for i in kks["assistKKS"]] 
            B2A = [True if i in kks["assistKKS"] else False for i in list(assistKKS)] 
            if sum(A2B) != len(assistKKS) or sum(B2A) != len(assistKKS) :continue
            
            result_bools[i]=True       
            
            X = data.loc[:,tuple(assistKKS)]

            y = data.loc[:,mainKKS]

            X_const = sm.add_constant(X)
            model = sm.OLS(y,X_const ).fit() 
            
            modelpath = local_path_model+'model.pkl'
            with open(modelpath, 'wb') as f:
                pickle.dump(model, f)
            state["model"] =   model
            state["status"] = States.XL_WANCHENG
            MODELS_STATUS[str(model_id)] = state 
            
            logging.info("***finish trainbatch modelid: {}".format(model_id))
        
    except Exception as e:
        logging.info("******training_batch modelid {},excep:{}".format(model_id,e))
        message = {
        'status': False,
        'message': "训练中异常excep: " + str(e),
        #'data':prediction_series
        "model_id": model_id
 
        }
        pass
        #continue
        # resp = requests.post(Config.java_host_train_batch, \
        #             data = json.dumps(message),\
        #             headers= header)
        # raise e
    
    logging.info("******train_batch finished result:\n{},\n{}".format(result_ids,result_bools))

    message = {
        'status': True,
        'message': "批量训练完成",
        "train_results": result_bools,
        "train_models": result_ids
    }
    resp = requests.post(Config.java_host_train_batch, \
                    data = json.dumps(message),\
                    headers= header)





@app.route('/train_batch', methods=['POST'])
def train_batch():
    
    request_json = request.get_json()
    #{"id":{""}}
    modelIdKKS = request_json["modelIdKKSDict"]
    datasetUrlList = request_json["datasetUrlList"]
    
    #MODELS_MAP[str(model_id)]["status"] = STATES.training

    train_batch_future = executor.submit(train_batch_task,modelIdKKS,datasetUrlList)
    
    message = {
			'status': True,
			'message': '-->模型批量开始训练',
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
    if str(model_id) in MODELS_MAP.keys():
        loaded_model = MODELS_MAP[str(model_id)]


    loaded_model.conf_int(0.05)
    
    

if __name__ == "__main__":
    # 将host设置为0.0.0.0，则外网用户也可以访问到这个服务
    app.run(host="0.0.0.0", port=8383, debug=True)

