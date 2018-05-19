#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : main.py
# @Author   : Likew
# @Date     : 2018/5/17 16:14
# @Desc     : 程序入口(线上线下配置不同路径和参数)
# @Solution : 配置参数→数据预处理→特征工程→模型选择→参数选择(早停、网格搜索)→预测输出

import time
import pre_process as pre
import process as pro

if __name__ == "__main__":
    online = False

    if online:
        from config_online import *
    else:
        from config_offline import *

    print("******************* Preprocess ***********************")
    start = time.time()

    train_origin = pre.read_csv(path_train)
    print('Loading Train_set Done! Time: %.3f s' % (time.time() - start))
    print('origin_train_shape = [%s,%s]' % (train_origin.shape))

    train_origin = pre.pre_process(train_origin)
    print('pre_process Train_set Done! Time: %.3f s' % (time.time() - start))

    train = pre.feature_engineering(train_origin, True)
    print('feature_engineering Train_set Done! Time: %.3f s' % (time.time() - start))
    print('train_shape = [%s,%s]' % (train.shape))

    del train_origin

    test_origin = pre.read_csv(path_test)
    print('Loading Test_set Done! Time: %.3f s' % (time.time() - start))
    print('origin_test_shape = [%s,%s]' % (test_origin.shape))

    test_origin = pre.pre_process(test_origin)
    print('pre_process Test_set Done! Time: %.3f s' % (time.time() - start))

    test = pre.feature_engineering(test_origin, False)
    print('feature_engineering Test_set Done! Time: %.3f s' % (time.time() - start))
    print('test_shape = [%s,%s]' % (test.shape))

    del test_origin

    result = pro.process(train, test, start, online)
    result.to_csv(path_test_out, header=True, index=False)

    print('\nDone! Time used: %.3f s' % (time.time() - start))
