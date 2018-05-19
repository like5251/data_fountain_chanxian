#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : config_online.py
# @Author   : Likew
# @Date     : 2018/5/18 19:58
# @Desc     : 
# @Solution :

import warnings

origin_features = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED", "CALLSTATE"]
origin_label = 'Y'

warnings.filterwarnings("ignore")

path_train = "/data/dm/train.csv"       # 训练文件
path_test = "/data/dm/test.csv"         # 测试文件
path_test_out = "model/pro_result.csv"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式
