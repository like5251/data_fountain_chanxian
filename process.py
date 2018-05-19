#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : process.py
# @Author   : Likew
# @Date     : 2018/5/18 16:14
# @Desc     : 模型选择-参数选择-预测输出
# @Solution : 调参之法（早停止确定最优迭代次数+网格搜索确定最优超参数）

import pandas as pd
import numpy as np
import lightgbm as lgb
import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer


def gini(actual, pred):
    """
    计算真实序列按照预测序列升序排列时的相对基尼系数，如果真实序列均为0，定义gini为1是合理的
    :param actual: 真实序列>=0
    :param pred: 预测序列
    :return: 基尼系数
    """
    n = len(pred)
    triple = np.c_[actual, pred, range(n)].astype(float)
    triple = triple[np.lexsort((triple[:, 2], triple[:, 1]))]
    cum_sum = triple[:, 0].cumsum()
    if cum_sum[-1] == 0:
        return 1
    else:
        x = cum_sum.sum() / cum_sum[-1]
        return (n + 1 - 2 * x) / n


def gini_normalized(y_true, y_pred):
    """
    自定义用于fit的eval_metric指标函数，此处为归一化的相对基尼系数
    :param y_true: 真实序列ndarray
    :param y_pred: 预测序列ndarray
    :return: eval_name, eval_result, is_bigger_better
    """
    gini_true = gini(y_true, y_true)
    gini_pred = gini(y_true, y_pred)
    # 如果真实值均匀分布，gini为0，定义此时的归一化为1是合理的
    res = gini_pred / gini_true if gini_true else 1
    return 'gini_normalized', res, True


def gini_feval(y_pred, train_data):
    """
    用于cv的自定义的feval指标函数
    :param y_pred: 预测值ndarray
    :param train_data: Dataset
    :return: (eval_name:string,eval_score,higher_is_better:bool)
    """
    labels = train_data.get_label()
    return 'gini', gini_normalized(labels, y_pred), True


def gini_grid(y_label, y_pred):
    """
    用于grid_search的scoring
    :param y_label: 真实序列ndarray
    :param y_pred: 预测序列ndarray
    :return: 归一化的基尼指数
    """
    return gini_normalized(y_label, y_pred)[1]


def report(results, top=5):
    """
    打印网格搜索结果中排名前10的参数组合和评估性能
    :param results: 网格搜索对象的结果属性cv_results_
    :param top: 排名前几的
    :return: None
    """
    print("GridSearchCV took %d candidate parameter settings." % len(results['params']))
    for i in range(1, top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print('model rank:%s' % i)
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def report_in_dataframe(results, top=10):
    """
    以DataFrame形式打印得分top_k的搜索结果：排名-参数-得分
    :param results: cv_results_
    :param top: int
    :return: DataFrame
    """
    result = pd.DataFrame(results['params'])
    result['test_score'] = results['mean_test_score']
    result['test_std'] = results['std_test_score']
    result['rank'] = results['rank_test_score']

    result = result.set_index('rank').sort_index()
    print(result[:top])
    return result[:top]


def early_stopping(lgb_reg, X_train, y_train, X_valid, y_valid, online):
    """
    通过早停确定最优迭代次数
    :param lgb_reg: 回归模型
    :param X_train: 部分训练集特征空间
    :param y_train: 部分训练集标签空间
    :param X_valid: 验证集特征空间
    :param y_valid: 验证集标签空间
    :param online: 是否线上
    :return: 设置了最优迭代次数的模型
    """
    lgb_reg.fit(X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_names=['gini'],
                eval_metric=gini_normalized,
                early_stopping_rounds=100,
                verbose=[1, 10][online])

    best_iteration = lgb_reg.best_iteration_
    lgb_reg.set_params(n_estimators=best_iteration)
    return lgb_reg


def random_search(lgb_reg, params_dist, X_train, y_train, n_iter=10, nfold=3):
    """
    随机网格搜索确定最优参数，返回使用最优参数全量refit后的模型
    :param lgb_reg: 模型
    :param params_dist: 参数网格
    :param X_train: 全部训练集的特征空间DataFrame
    :param y_train: 全部训练集的标记空间Series
    :param n_iter: 参数采样次数
    :param nfold: 交叉验证折数
    :return: refit后的模型
    """
    rs = RandomizedSearchCV(lgb_reg,
                            params_dist,
                            n_iter=n_iter,
                            scoring=make_scorer(gini_grid),
                            cv=nfold,
                            refit=True,
                            verbose=0,
                            return_train_score=False)
    rs.fit(X_train.values, y_train.values)

    report_in_dataframe(rs.cv_results_)
    return rs.best_estimator_


def process(train, test, start, online):
    """
    调参、训练、预测全过程
    :param train: 训练集DataFrame
    :param test: 测试集DataFrame
    :param start: 开始执行时间
    :param online: 是否线上
    :return: 预测结果DataFrame
    """
    # 特征和标签
    features = [c for c in train.columns if c not in ['label']]
    target = 'label'

    # 划分训练集和验证集:3:1
    ratio = int(train.shape[0] * 0.66)
    X_train = train[features].iloc[:ratio]
    y_train = train[target].iloc[:ratio]
    X_valid = train[features].iloc[ratio:]
    y_valid = train[target].iloc[ratio:]

    # 模型：leaf_wise生长策略，线上应该适当调大num_leaves，但是线下就跑不了
    if online:
        lgb_reg = lgb.LGBMRegressor(num_leaves=176,
                                    max_depth=6,

                                    learning_rate=0.01,
                                    n_estimators=1000,

                                    max_bin=255,
                                    subsample_for_bin=200000,

                                    min_split_gain=0.0,
                                    min_child_weight=0.001,
                                    min_child_samples=26,

                                    subsample=0.9,
                                    subsample_freq=3,
                                    colsample_bytree=0.5,

                                    reg_alpha=0.5,
                                    reg_lambda=5.0,

                                    random_state=999,
                                    n_jobs=-1,
                                    silent=True,
                                    verbose=-1)
    else:
        lgb_reg = lgb.LGBMRegressor(num_leaves=5,
                                    max_depth=-1,

                                    learning_rate=0.02,
                                    n_estimators=1000,

                                    max_bin=255,
                                    subsample_for_bin=50000,

                                    min_split_gain=0.0,
                                    min_child_weight=0.001,
                                    min_child_samples=5,

                                    subsample=0.8,
                                    subsample_freq=3,
                                    colsample_bytree=0.8,

                                    reg_alpha=0.0,
                                    reg_lambda=1.0,

                                    random_state=999,
                                    n_jobs=-1,
                                    silent=True,
                                    verbose=[0, -1][online])

    # 早停确定最优迭代次数
    print("\n******************* Early Stopping ***********************")
    lgb_reg = early_stopping(lgb_reg, X_train, y_train, X_valid, y_valid, online)
    print('Early Stopping Done! Time:%.3f s' % (time.time() - start))
    print('best n_estimators:%s' % lgb_reg.best_iteration_)

    # 网格搜索确定最优超参数
    print("\n******************* Random Search ***********************")
    if online:
        params_dist = {'num_leaves': range(2, 300, 2),
                       'max_depth': [-1, 6, 8, 10, 15],
                       'min_child_samples': range(2, 100, 2),

                       'subsample': [i / 10. for i in range(3, 11)],
                       'colsample_bytree': [i / 10. for i in range(3, 11)],
                       'subsample_freq': range(1, 10),

                       'reg_alpha': [0, 0.5, 1., 1.5, 2., 5., 10.],
                       'reg_lambda': [0, 0.5, 1., 1.5, 2., 5., 10.]}
    else:
        params_dist = {'num_leaves': range(2, 10),
                       'max_depth': [-1, 6, 8, 10],
                       'min_child_samples': range(2, 10),

                       'subsample': [i / 10. for i in range(5, 11)],
                       'colsample_bytree': [i / 10. for i in range(5, 11)],

                       'reg_alpha': [0, 0.2, 0.5, 1., 1.5, 2., 5., 10., 20.],
                       'reg_lambda': [0, 0.2, 0.5, 1., 1.5, 2., 5., 10., 20.]}

    lgb_reg = random_search(lgb_reg, params_dist, train[features], train[target], n_iter=200)

    print('RandomizedSearchCV Done! Time:%.3f s' % (time.time() - start))

    print('\nlast params:%s' % lgb_reg.get_params())
    print('feature importances:%s' % lgb_reg.feature_importances_)

    # 预测，注意DataFram中的列名和列序
    y_pred = lgb_reg.predict(test[features])
    result = pd.DataFrame({'Id': test.index, 'Pred': y_pred}, columns=['Id', 'Pred'])
    return result


if __name__ == '__main__':
    pass
