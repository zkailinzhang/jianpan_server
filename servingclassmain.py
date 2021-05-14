# -*- coding: utf-8 -*-

from flask import Flask, jsonify, request
import requests
from servingclass import ModelService




app = Flask(__name__)


modelservice = ModelService()


@app.route("/predict",methods=['POST','GET'])
def predict():
    return modelservice.predict()



@app.route("/publish",methods=['POST','GET'])
def publish():
    return modelservice.predict()


@app.route("/publish_cancel",methods=['POST','GET'])
def publish_cancel():
    return modelservice.publish_cancel()


@app.route("/evaluate",methods=['POST','GET'])
def evaluate():
    return modelservice.evaluate()


@app.route("/evaluate_cancel",methods=['POST','GET'])
def evaluate_cancel():
    return modelservice.evaluate_cancel()


@app.route("/evaluate_renew",methods=['POST','GET'])
def evaluate_renew():
    return modelservice.evaluate_renew()


@app.route("/train",methods=['POST','GET'])
def train():
    return modelservice.train()


@app.route("/train_batch",methods=['POST','GET'])
def train_batch():
    return modelservice.train_batch()


@app.route("/train_cancel",methods=['POST','GET'])
def train_cancel():
    return modelservice.train_cancel()


@app.errorhandler(400)
def bad_request():
    return modelservice.bad_request(400)


#趋势预测zbx
@app.route("/trend_predict",methods=['POST','GET'])
def trend_predict():
    return modelservice.trend_predict()


@app.route("/suddenChange_predict",methods=['POST','GET'])
def suddenChange_predict():
    return modelservice.suddenChangePredict()

if __name__ == "__main__":
    # 将host设置为0.0.0.0，则外网用户也可以访问到这个服务
    app.run(host="0.0.0.0", port=8383, debug=True)
