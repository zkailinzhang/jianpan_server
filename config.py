# -*- coding: utf-8 -*-

import os 

class Config:
    K1 = 1.6667
    B1 = 1.5333
    K2 = 3.5
    B2 = 3.28
    confidence = 0.95
    confidence_second = 0.98
    #java_host = "http://172.17.231.79:30069/model/bizModel/getModelEvaluationResult"
    java_host = "http://192.168.18.51:30069/model/bizModel/getModelEvaluationResult"
    #redis_host = '172.17.224.171'
    redis_host = '192.168.18.13'
    redis_port = 30144 #6379
    redis_db = 0
    redis_password = 123456
    