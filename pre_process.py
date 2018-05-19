#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : pre_process.py
# @Author   : Likew
# @Date     : 2018/5/18 15:58
# @Desc     : 数据预处理、特征工程
# @Solution : 缺失值填充、按用户、行程或时间统计聚合特征，尽可能多的生成新的特征，再做筛选

import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt


def read_csv(filename):
    """
    读取csv文件
    :return:
    """
    temp_data = pd.read_csv(filename)
    return temp_data


def pre_process(data):
    """
    对原始数据集进行预处理，包括填补缺失值，时间处理操作
    :param data: DataFrame
    :return: 处理后的数据集
    """
    # 插补速度和方向的缺失值
    data.loc[data.SPEED < 0, 'SPEED'] = np.nan
    data.loc[data.DIRECTION < 0, 'DIRECTION'] = np.nan
    data.loc[:, ['SPEED', 'DIRECTION']].fillna(method='ffill', inplace=True)
    data.loc[:, ['SPEED', 'DIRECTION']].fillna(method='bfill', inplace=True)

    # 处理时间特征
    data['datetime'] = pd.to_datetime(data.TIME, origin='unix', unit='s')
    data['date'] = data.datetime.apply(lambda x: x.date())
    data['hour'] = data.datetime.apply(lambda x: x.hour)
    return data


def feature_engineering(data, is_train):
    """
    对数据集data进行特征工程，返回新的数据集
    :param data: DataFrame数据集
    :param is_train: 是否为训练集
    :return: 新的DataFrame数据集
    """
    all_user = data.TERMINALNO.unique()

    # groupby会爆内存，索引用循环，虽然速度慢点但还可接受
    res = []
    for item in all_user:
        user = data.loc[data.TERMINALNO == item]
        user_dic = {'Id': item}

        # 记录数_按用户：表示用户驾驶总时长
        user_dic['user_record_num'] = user.shape[0]
        # 行程数_按用户：表示用户出行总次数
        user_dic['user_trip_num'] = user.TRIP_ID.nunique()
        # 天数_按用户：表示用户出行总天数
        user_dic['user_days_num'] = user.date.nunique()
        # 速度_按用户：表示用户出行整体速度均值
        user_dic['user_speed_avg'] = user.SPEED.mean()
        # 手机状态_按用户:表示用户出行时通话分布
        for i in range(5):
            user_dic['user_phone_%s' % i] = user.loc[user.CALLSTATE == i].shape[0] / float(user_dic['user_record_num'])
        # 经纬度_按用户：用户出行的经纬度均值与标准差，表示用户活动空间范围
        user_dic['user_longi_avg'] = user.LONGITUDE.mean()
        user_dic['user_lati_avg'] = user.LATITUDE.mean()
        user_dic['user_longi_var'] = user.LONGITUDE.var()
        user_dic['user_lati_var'] = user.LATITUDE.var()
        # 海拔_按用户：用户出行的海拔均值与标准差，表示用户活动海拔范围
        user_dic['user_height_avg'] = user.HEIGHT.mean()
        user_dic['user_height_var'] = user.HEIGHT.var()

        trip_group = user.groupby('TRIP_ID')
        # 记录数_按行程：表示行程持续时间的均值和最大值、标准差
        trip_record_size = trip_group.size()
        user_dic['trip_record_avg'] = trip_record_size.mean()
        user_dic['trip_record_max'] = trip_record_size.max()
        user_dic['trip_record_var'] = trip_record_size.var()
        # 速度_按行程：表示行程速度均值的均值、最大值、标准差（均速）；
        trip_speed_avg = trip_group['SPEED'].mean()
        user_dic['trip_speed_avg_avg'] = trip_speed_avg.mean()
        user_dic['trip_speed_avg_max'] = trip_speed_avg.max()
        user_dic['trip_speed_avg_var'] = trip_speed_avg.var()
        # 行程速度最大值(超速)的均值、最大值、标准差（最高速）；
        trip_speed_max = trip_group['SPEED'].max()
        user_dic['trip_speed_max_avg'] = trip_speed_max.mean()
        user_dic['trip_speed_max_max'] = trip_speed_max.max()
        user_dic['trip_speed_max_var'] = trip_speed_max.var()
        # 行程速度标准差的均值（稳定性）；
        user_dic['trip_speed_var_avg'] = trip_group['SPEED'].var().mean()
        # 方向_按行程求差分：表示行程路况弯道情况
        trip_dir_div = trip_group['DIRECTION'].apply(direction_div)
        user_dic['trip_dir_div_avg'] = trip_dir_div.mean()
        user_dic['trip_dir_div_max'] = trip_dir_div.max()
        user_dic['trip_dir_div_var'] = trip_dir_div.var()
        # 电话_按行程：表示用户开车时打电话习惯
        user_dic['trip_phone_move_ratio'] = trip_group[['SPEED', 'CALLSTATE']].apply(call_whiile_drive).sum() / float(
            user_dic['user_trip_num'])
        # 经纬度_按行程：行程始末距离的均值、标准差、行程平均经纬度的均值和标准差
        trip_distance = trip_group[['LONGITUDE', 'LATITUDE']].apply(cal_trip_distance)
        user_dic['trip_distance_avg'] = trip_distance.mean()
        user_dic['trip_distance_var'] = trip_distance.var()
        trip_longi_avg = trip_group['LONGITUDE'].mean()
        user_dic['trip_longi_avg'] = trip_longi_avg.mean()
        user_dic['trip_longi_var'] = trip_longi_avg.var()
        trip_lati_avg = trip_group['LATITUDE'].mean()
        user_dic['trip_lati_avg'] = trip_lati_avg.mean()
        user_dic['trip_lati_var'] = trip_lati_avg.var()
        # 海拔_按行程：行程海拔均值的均值、标准差；行程海拔标准差的均值、标准差;海拔的差分的均值、最大值、标准差
        trip_height_avg = trip_group['HEIGHT'].mean()
        user_dic['trip_height_avg_avg'] = trip_height_avg.mean()
        user_dic['trip_height_avg_var'] = trip_height_avg.var()
        trip_height_var = trip_group['HEIGHT'].var()
        user_dic['trip_height_var_avg'] = trip_height_var.mean()
        user_dic['trip_height_var_var'] = trip_height_var.var()
        trip_height_div = trip_group['HEIGHT'].apply(direction_div)
        user_dic['trip_height_div_avg'] = trip_height_div.mean()
        user_dic['trip_height_div_max'] = trip_height_div.max()
        user_dic['trip_height_div_var'] = trip_height_div.var()

        # 记录数_按天：单天出行记录均值、最值、标准差
        date_group = user.groupby('date')
        date_group_size = date_group.size()
        user_dic['date_record_avg'] = date_group_size.mean()
        user_dic['date_record_max'] = date_group_size.max()
        user_dic['date_record_var'] = date_group_size.var()
        # 记录数_按时：表示24小时各时段记录数
        # 速度_按时：表示用户不同时段的平均速度和最大速度(超速)
        hour_group = user.groupby('hour')
        hour_group_size = hour_group.size()
        hour_group_index = hour_group_size.index
        for j in range(24):
            if j in hour_group_index:
                user_dic['hour_record_%sh' % j] = hour_group_size.loc[j]
                user_dic['hour_speed_avg_%sh' % j] = hour_group['SPEED'].mean().loc[j]
                user_dic['hour_speed_max_%sh' % j] = hour_group['SPEED'].max().loc[j]
            else:
                user_dic['hour_record_%sh' % j] = 0
                user_dic['hour_speed_avg_%sh' % j] = 0
                user_dic['hour_speed_max_%sh' % j] = 0
        # 行程数_按天：单天出行行程均值、最值、标准差
        date_trip = date_group['TRIP_ID'].nunique()
        user_dic['date_trip_avg'] = date_trip.mean()
        user_dic['date_trip_max'] = date_trip.max()
        user_dic['date_trip_var'] = date_trip.var()

        # 标签
        if is_train:
            user_dic['label'] = user.Y.iat[0]
        res.append(user_dic)

    data_fe = pd.DataFrame(res)
    data_fe.set_index('Id', inplace=True)

    # 数据集中可能出现缺失值
    data_fe.fillna(-1, inplace=True)

    return data_fe


def direction_div(x):
    """
    方向在当前行程中的差分均值
    :param x: grouped单行程方向
    :return: 差分均值
    """
    if x.shape[0] < 2:
        return 0
    tmp = abs(x[1:].values - x[:-1].values)
    res = np.minimum(tmp,360-tmp)
    return res.mean()


def call_whiile_drive(x):
    """
    如果在当前行程中有接打电话行为，则返回1，否则返回0
    :param x: 单行程中的电话行为
    :return:行程中接打电话返回1，否则返回0
    """
    tmp = x.loc[(x.SPEED > 0) & (x.CALLSTATE.isin([1, 2, 3]))].shape[0]
    return 1 if tmp else 0


def cal_trip_distance(x):
    """
    计算该行程中的始末位置距离
    :param x: 单行程的经纬度
    :return: 行程始末位置的球面距离
    """
    lon1 = x.iloc[0, 0]
    lat1 = x.iloc[0, 1]
    lon2 = x.iloc[-1, 0]
    lat2 = x.iloc[-1, 1]
    return haversine1(lon1, lat1, lon2, lat2)


def haversine1(lon1, lat1, lon2, lat2):
    """
    计算两个地点的球面距离
    :param lon1: 出发点经度
    :param lat1: 出发点纬度
    :param lon2: 结束点经度
    :param lat2: 结束点纬度
    :return: 始末位置距离
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    # 地球平均半径，单位为公里
    r = 6371
    return c * r * 1000


if __name__ == '__main__':
    pass
