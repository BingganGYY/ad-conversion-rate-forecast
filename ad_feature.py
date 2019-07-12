import pandas as pd
import numpy as np
from pylab import mpl
from scipy import stats  # 求众数
from sklearn import preprocessing  # 求最小-最大标准化


import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split  # 将矩阵随机划分为训练子集和测试子集,并返回划分好的训练集、测试集样本和训练集、测试集标签
from sklearn.model_selection import StratifiedKFold  # 分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同
from sklearn.metrics import log_loss
from sklearn import preprocessing  # sklearn中的数据预处理preprocessing模块,它可以对数据进行标准化
# preprocessing.scale(X,axis=0, with_mean=True, with_std=True, copy=True)：将数据转化为标准正态分布（均值为0，方差为1）
# preprocessing.minmax_scale(X,feature_range=(0, 1), axis=0, copy=True)：将数据在缩放在固定区间，默认缩放到区间 [0, 1]

import warnings

warnings.filterwarnings("ignore")  # 把程序打包后，不显示告警信息

import time  # 处理时间戳
from itertools import product
import itertools  # 提供了非常有用的用于操作迭代对象的函数
import copy

import seaborn as sns  # 数据可视化
import matplotlib.pyplot as plt

data = pd.read_csv('C:/Users/Amber/Desktop/ad_data/ad_train.txt', sep=' ')  # 最开始最真实的train
data = data.drop_duplicates(subset='instance_id')  # 删除重复样本数，最终不重复：(478087, 27)
# print(data.shape)
# data.to_csv('C:/Users/Amber/Desktop/ad_data/data.csv')

###------------1、缺失值分析及处理
data_null_num = (data == -1).astype(int).sum(axis=0)  # 统计df里每一个变量取值-1的个数
print(data_null_num)
# 各个特征缺失比例图
data_null_percentage = data_null_num / len(data)
print('The null data percentage is:', data_null_percentage)
# 显示各特征缺失比例图像
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体，仿宋
data_null_percentage = data_null_percentage.reset_index()
data_null_percentage.columns = ['column_name', 'column_value']
ind = np.arange(data_null_percentage.shape[0])
fig, ax = plt.subplots(figsize=(6, 8))
rects = ax.barh(ind, data_null_percentage.column_value.values, color='lightskyblue')
ax.set_yticks(ind)
ax.set_yticklabels(data_null_percentage.column_name.values, rotation='horizontal')
ax.set_xlabel("各特征缺失数据比例")
plt.show()
# 缺失值处理---中位数(连续)、众数（离散）填充
print(data)
lianxu_columns = pd.Series(
    ['shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description'])
lisan_columns = pd.Series(
    ['item_brand_id', 'item_city_id', 'item_sales_level', 'user_gender_id', 'user_age_level', 'user_occupation_id',
     'user_star_level'])
for i in lianxu_columns:
    data.loc[data[i] == -1, i] = data.median()[i]
# 检验
# data_null_num = (data == -1).astype(int).sum(axis=0)
# print(data_null_num)
# data.to_csv('C:/Users/Amber/Desktop/data_lianxu.csv')
for i in lisan_columns:
    data.loc[data[i] == -1, i] = stats.mode(data[lisan_columns][i])[0][0]  # 求众数超棒的语句，一句结束

# 检验
# data_null_num = (data == -1).astype(int).sum(axis=0)
# print(data_null_num)
# data.to_csv('C:/Users/Amber/Desktop/data_lisan.csv')

'''
#标准化（最小-最大标准化）
norm_columns = pd.Series(['item_price_level','item_sales_level','item_collected_level','item_pv_level','user_age_level','user_star_level','context_page_id','shop_review_num_level','shop_star_level'])
#print(len(norm_columns))
for i in norm_columns:
    data[i]=(data[i]-data[i].min())/(data[i].max()-data[i].min())#最大最小标准化,对一列数据进行统一操作
#print(data[norm_columns].head(1))
#min_max_sacler = preprocessing.MinMaxScaler()#适合所有变量一起标准化
#min_max_sacler.fit(X_train)
#print(min_max_sacler.transform(X_train))
#data.to_csv('C:/Users/Amber/Desktop/data_norm.csv')
'''



#商品、用户、商铺特征与点击（转换率之间的关系）---便于分段
#1、商品

sns.pointplot(x='item_price_level', y='is_trade', data=data,color='lightskyblue')
sns.pointplot(x='item_sales_level', y='is_trade', data=data,color='lightskyblue')
sns.pointplot(x='item_collected_level', y='is_trade', data=data,color='lightskyblue')
sns.pointplot(x='item_pv_level', y='is_trade', data=data,color='lightskyblue')
#2、用户

sns.pointplot(x='user_gender_id', y='is_trade', data=data,color='lightskyblue')
sns.pointplot(x='user_age_level', y='is_trade', data=data,color='lightskyblue')
sns.pointplot(x='user_occupation_id', y='is_trade', data=data,color='lightskyblue')
sns.pointplot(x='user_star_level', y='is_trade', data=data,color='lightskyblue')

#3、商铺
sns.pointplot(x='shop_review_num_level', y='is_trade', data=data,color='lightskyblue')
sns.pointplot(x='shop_review_positive_rate', y='is_trade', data=data,color='lightskyblue')
sns.pointplot(x='shop_star_level', y='is_trade', data=data,color='lightskyblue')
sns.pointplot(x='shop_score_service', y='is_trade', data=data,color='lightskyblue')
sns.pointplot(x='shop_score_delivery', y='is_trade', data=data,color='lightskyblue')
sns.pointplot(x='shop_score_description', y='is_trade', data=data,color='lightskyblue')

#4、时间
grouped_df = data.groupby(["day", "hour"])["is_trade"].aggregate("mean").reset_index()
grouped_df = grouped_df.pivot('day', 'hour', 'is_trade')
plt.figure(figsize=(12,6))
sns.heatmap(grouped_df)
plt.title("CVR")
#plt.show()

###------------2、特征分段

# shop四个连续字段分段
# 在Python中如果想要对数据使用函数，可以借助apply(),applymap(),map() 来应用函数，括号里面可以是直接函数式，或者自定义函数（def）或者匿名函数（lambad）
# 疑问？分段的数据哪来的呢 我是通过聚类离散化找到分割点的,数据数据有改动
def shop_fenduan(data):
    data['shop_score_description0'] = data['shop_score_description'].apply(lambda x: 2 if x > 0.973 else x)
    data['shop_score_description0'] = data['shop_score_description0'].apply(lambda x: 1 if 0.973 >= x > 0.955 else x)
    data['shop_score_description0'] = data['shop_score_description0'].apply(lambda x: 0 if x <= 0.955 else x)
    data['shop_score_delivery0'] = data['shop_score_delivery'].apply(lambda x: 2 if x > 0.972 else x)
    data['shop_score_delivery0'] = data['shop_score_delivery0'].apply(lambda x: 1 if 0.959 >= x > 0.972 else x)
    data['shop_score_delivery0'] = data['shop_score_delivery0'].apply(lambda x: 0 if x <= 0.959 else x)
    data['shop_score_service0'] = data['shop_score_service'].apply(lambda x: 2 if x > 0.973 else x)
    data['shop_score_service0'] = data['shop_score_service0'].apply(lambda x: 1 if 0.959 >= x > 0.973 else x)
    data['shop_score_service0'] = data['shop_score_service0'].apply(lambda x: 0 if x <= 0.959 else x)
    data['shop_review_positive_rate0'] = data['shop_review_positive_rate'].apply(lambda x: 2 if x > 0.993 else x)
    data['shop_review_positive_rate0'] = data['shop_review_positive_rate0'].apply(
        lambda x: 1 if 0.993 >= x > 0.976 else x)
    data['shop_review_positive_rate0'] = data['shop_review_positive_rate0'].apply(lambda x: 0 if x <= 0.976 else x)
    return data

def baseprocess(data):
    print('1.baseprocess')
    print(
        '-----------------------------------------------------item_category_list等分解----------------------------------------------------')
    lbl = preprocessing.LabelEncoder()
    for i in range(1, 3):  # 1,2 2个
        data['item_category_list' + str(i)] = lbl.fit_transform(data['item_category_list'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))  # item_category_list的第0列全部都一样
    for i in range(10):  # 0,1...,9 10个
        data['item_property_list' + str(i)] = lbl.fit_transform(
            data['item_property_list'].map(lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
    for i in range(5):  # 0,1,...,4 5个
        data['predict_category_property' + str(i)] = lbl.fit_transform(data['predict_category_property'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))

    print(
        '--------------------------------------------------------------时间处理--------------------------------------------------------------')
    data['realtime'] = data['context_timestamp'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
    data['realtime'] = pd.to_datetime(data['realtime'])
    data['day'] = data['realtime'].dt.day
    data['hour'] = data['realtime'].dt.hour
    # print(
    #   '----------------------------------------------------------item_category_list等长度计算--------------------------------------------')
    # data['len_item_category'] = data['item_category_list'].map(lambda x: len(str(x).split(';')))
    # data['len_item_property'] = data['item_property_list'].map(lambda x: len(str(x).split(';')))
    # data['len_predict_category_property'] = data['predict_category_property'].map(lambda x: len(str(x).split(';')))
    return data


'''	
# 时间分段---检验过合理
def hour_fenduan(data):
    data['hour0'] = data['hour'].apply(lambda x: 1 if 7 <= x <= 12 else x)
    data['hour0'] = data['hour0'].apply(lambda x: 2 if 13<= x <=20 else x)
    data['hour0'] = data['hour0'].apply(lambda x: 3 )
'''
###----------3、单个特征点击转换率
'''  Descr:输入：带有日期特征的数据集data，特征名称col_name
       输出：带有两个新特征的data:
               该特征的历史点击次数:col_name+'_cnt'
              该特征的历史点击转换率:col_name+'_cntrate11'
              '''


def get_cnt(data, col_name):
    user_cnt = data.groupby(by=[col_name, 'day'])['is_trade'].agg({col_name + '_cnt': 'count'})
    user_cnt = user_cnt.unstack()
    user_cnt.fillna(0, inplace=True)
    user_cnt = user_cnt.reindex_axis(sorted(user_cnt.columns), axis=1)

    user_cnt = user_cnt.cumsum(axis=1)
    user_cnt = user_cnt.stack()
    user_cnt = user_cnt.reset_index()
    user_cnt['day'] = user_cnt['day'].map(lambda x: x + 1)

    user_cnt1 = data.groupby(by=[col_name, 'day', 'is_trade']).agg('size')
    user_cnt1 = user_cnt1.unstack(level=['is_trade', 'day'])
    user_cnt1 = user_cnt1[:][1]
    user_cnt1.fillna(0, inplace=True)
    user_cnt1 = user_cnt1.reindex_axis(sorted(user_cnt1.columns.tolist()), axis=1)
    user_cnt1 = user_cnt1.cumsum(axis=1)
    user_cnt1 = user_cnt1.stack()
    user_cnt1 = user_cnt1.reset_index()
    user_cnt1.columns = [col_name, 'day', col_name + '_cntrate11']
    user_cnt1['day'] = user_cnt1['day'].map(lambda x: x + 1)

    data = pd.merge(data, user_cnt, on=[col_name, 'day'], how='left')
    data = pd.merge(data, user_cnt1, on=[col_name, 'day'], how='left')
    data[col_name + '_cntrate11'] = data[col_name + '_cntrate11'] / data[col_name + '_cnt']
    del data[col_name + '_cnt']
    return data


def get_all_cnt(data):
    '''
       Descr:输入：数据data,统计函数get_cnt(data,col_name)
                   需要统计的特征列表（此处函数内部已定义）
               输出：包含每个特征的历史点击转换率、历史点击次数的data
       '''
    item_category_list_name = ['item_category_list' + str(i) for i in range(1, 3)]
    item_property_list_name = ['item_property_list' + str(i) for i in range(10)]
    predict_category_property_name = ['predict_category_property' + str(i) for i in range(5)]
    item_name = ['item_sales_level', 'item_price_level', 'item_collected_level', 'item_pv_level', 'item_brand_id',
                 'item_city_id']
    item_name = item_name + item_category_list_name + item_property_list_name + predict_category_property_name
    user_name = ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
    shop_name = ['shop_review_num_level', 'shop_review_positive_rate0', 'shop_star_level', 'shop_score_service0',
                 'shop_score_delivery0', 'shop_score_description0']
    id_name = ['user_id', 'item_id', 'shop_id', 'context_page_id']
    time_name = ['hour']


    all_name = item_name + id_name + user_name + shop_name + time_name

    for col_name in all_name:
        data = get_cnt(data, col_name)
        print('------第一个！单特征哦┏ (^ω^)=☞：' + col_name)
    return data

###------------4、交叉特征点击转换率
'''
#所有单个特征一起交叉，没有大类的区分
def get_cross_cnt(data, col_name, base_name):
    #    col_name = 'item_brand_id'
    #    base_name = 'item_id'
    cnt = data.groupby(by=[col_name, base_name, 'day'])['is_trade'].agg({col_name + '_' + base_name + '_cnt': 'count'})
    cnt = cnt.unstack()
    cnt.fillna(0, inplace=True)
    cnt = cnt.reindex_axis(sorted(cnt.columns), axis=1)
    cnt = cnt.cumsum(axis=1)
    cnt = cnt.stack()
    cnt = cnt.reset_index()
    cnt['day'] = cnt['day'].map(lambda x: x + 1)

    cnt1 = data.groupby(by=[col_name, base_name, 'day', 'is_trade']).agg('size')
    cnt1 = cnt1.unstack(level=['is_trade', 'day'])
    cnt1 = cnt1[:][1]
    cnt1.fillna(0, inplace=True)
    cnt1 = cnt1.reindex_axis(sorted(cnt1.columns.tolist()), axis=1)
    cnt1 = cnt1.cumsum(axis=1)
    cnt1 = cnt1.stack()
    cnt1 = cnt1.reset_index()
    cnt1.columns = [col_name, base_name, 'day', col_name + '_' + base_name + '_cross_cntrate11']
    cnt1['day'] = cnt1['day'].map(lambda x: x + 1)

    data = pd.merge(data, cnt, on=[col_name, base_name, 'day'], how='left')
    data = pd.merge(data, cnt1, on=[col_name, base_name, 'day'], how='left')

    #    col_name的单特征量
    cnt_col = data.groupby(by=[col_name, 'day'])['is_trade'].agg({col_name + '_cnt': 'count'})
    cnt_col = cnt_col.unstack()
    cnt_col.fillna(0, inplace=True)
    cnt_col = cnt_col.reindex_axis(sorted(cnt_col.columns), axis=1)
    cnt_col = cnt_col.cumsum(axis=1)
    cnt_col = cnt_col.stack()
    cnt_col = cnt_col.reset_index()
    cnt_col['day'] = cnt_col['day'].map(lambda x: x + 1)

    cnt_col1 = data.groupby(by=[col_name, 'day', 'is_trade']).agg('size')
    cnt_col1 = cnt_col1.unstack(level=['is_trade', 'day'])
    cnt_col1 = cnt_col1[:][1]
    cnt_col1.fillna(0, inplace=True)
    cnt_col1 = cnt_col1.reindex_axis(sorted(cnt_col1.columns.tolist()), axis=1)
    cnt_col1 = cnt_col1.cumsum(axis=1)
    cnt_col1 = cnt_col1.stack()
    cnt_col1 = cnt_col1.reset_index()
    cnt_col1.columns = [col_name, 'day', col_name + '_cntrate11']
    cnt_col1['day'] = cnt_col1['day'].map(lambda x: x + 1)

    data = pd.merge(data, cnt_col, on=[col_name, 'day'], how='left')
    data = pd.merge(data, cnt_col1, on=[col_name, 'day'], how='left')

    #    base_name的单特征量
    cnt_base = data.groupby(by=[base_name, 'day'])['is_trade'].agg({base_name + '_cnt': 'count'})
    cnt_base = cnt_base.unstack()
    cnt_base.fillna(0, inplace=True)
    cnt_base = cnt_base.reindex_axis(sorted(cnt_base.columns), axis=1)
    cnt_base = cnt_base.cumsum(axis=1)
    cnt_base = cnt_base.stack()
    cnt_base = cnt_base.reset_index()
    cnt_base['day'] = cnt_base['day'].map(lambda x: x + 1)

    cnt_base1 = data.groupby(by=[base_name, 'day', 'is_trade']).agg('size')
    cnt_base1 = cnt_base1.unstack(level=['is_trade', 'day'])
    cnt_base1 = cnt_base1[:][1]
    cnt_base1.fillna(0, inplace=True)
    cnt_base1 = cnt_base1.reindex_axis(sorted(cnt_base1.columns.tolist()), axis=1)
    cnt_base1 = cnt_base1.cumsum(axis=1)
    cnt_base1 = cnt_base1.stack()
    cnt_base1 = cnt_base1.reset_index()
    cnt_base1.columns = [base_name, 'day', base_name + '_cntrate11']
    cnt_base1['day'] = cnt_base1['day'].map(lambda x: x + 1)

    data = pd.merge(data, cnt_base, on=[base_name, 'day'], how='left')
    data = pd.merge(data, cnt_base1, on=[base_name, 'day'], how='left')

    #    col_name+'_'+base_name+'_cross_cntrate11'
    data[col_name + '_' + base_name + '_cb'] = data[col_name + '_' + base_name + '_cross_cntrate11'] / data[
        col_name + '_' + base_name + '_cnt']

    data[col_name + '_' + base_name + '_cb_c'] = data[col_name + '_' + base_name + '_cnt'] / data[col_name + '_cnt']
    data[col_name + '_' + base_name + '_cb_b'] = data[col_name + '_' + base_name + '_cnt'] / data[base_name + '_cnt']

    data[col_name + '_' + base_name + '_cb_c1'] = data[col_name + '_' + base_name + '_cross_cntrate11'] / data[
        col_name + '_cntrate11']
    data[col_name + '_' + base_name + '_cb_b1'] = data[col_name + '_' + base_name + '_cross_cntrate11'] / data[
        base_name + '_cntrate11']

    del data[col_name + '_' + base_name + '_cnt']
    del data[col_name + '_cnt']
    del data[base_name + '_cnt']
    del data[col_name + '_' + base_name + '_cross_cntrate11']
    del data[col_name + '_cntrate11']
    del data[base_name + '_cntrate11']
    return data
'''


# 大类和大类之间特征的交叉
def get_cross_cnt2(data, col_name, base_name):
    '''
       Descr:输入：数据data,需要组合的特征名col_name,base_name，无先后顺序
           输出：含有该组合特征的历史点击次数col_name+'_'+base_name+'_cnt'
                 历史点击转换率col_name+'_'+base_name+'_cross_cntrate11'
         example:
           col_name = 'item_brand_id'
           base_name = 'item_id'
       '''
    cnt = data.groupby(by=[col_name, base_name, 'day'])['is_trade'].agg({col_name + '_' + base_name + '_cnt': 'count'}) # 对大类1、大类2、天按组分类后进行count计数
    cnt = cnt.unstack()  # 将花括号结构---表格结构
    cnt.fillna(0, inplace=True)
    cnt = cnt.reindex_axis(sorted(cnt.columns), axis=1)
    cnt = cnt.cumsum(axis=1)  # 累计求和
    cnt = cnt.stack()  # 表格结构---花括号结构
    cnt = cnt.reset_index()
    cnt['day'] = cnt['day'].map(lambda x: x + 1)

    cnt1 = data.groupby(by=[col_name, base_name, 'day', 'is_trade']).agg('size')
    cnt1 = cnt1.unstack(level=['is_trade', 'day'])
    cnt1 = cnt1[:][1]
    cnt1.fillna(0, inplace=True)
    cnt1 = cnt1.reindex_axis(sorted(cnt1.columns.tolist()), axis=1)
    cnt1 = cnt1.cumsum(axis=1)
    cnt1 = cnt1.stack()
    cnt1 = cnt1.reset_index()
    cnt1.columns = [col_name, base_name, 'day', col_name + '_' + base_name + '_cross_cntrate11']
    cnt1['day'] = cnt1['day'].map(lambda x: x + 1)

    data = pd.merge(data, cnt, on=[col_name, base_name, 'day'], how='left')
    data = pd.merge(data, cnt1, on=[col_name, base_name, 'day'], how='left')
    data[col_name + '_' + base_name + '_cross_cntrate11'] = data[col_name + '_' + base_name + '_cross_cntrate11'] / \
                                                            data[col_name + '_' + base_name + '_cnt']

    #    新增删除cnt
    del data[col_name + '_' + base_name + '_cnt']
    return data


def get_all_cross_cnt(data):
    item_category_list_name = ['item_category_list' + str(i) for i in range(1, 3)]
    item_property_list_name = ['item_property_list' + str(i) for i in range(10)]
    predict_category_property_name = ['predict_category_property' + str(i) for i in range(5)]
    item_name = ['item_sales_level', 'item_price_level', 'item_collected_level', 'item_pv_level', 'item_brand_id',
                 'item_city_id']
    item_name = item_name + item_category_list_name + item_property_list_name + predict_category_property_name
    user_name = ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
    shop_name = ['shop_review_num_level', 'shop_review_positive_rate0', 'shop_star_level', 'shop_score_service0',
                 'shop_score_delivery0', 'shop_score_description0']
    all_name = item_name + user_name + shop_name

    for col_name in user_name:
        for base_name in item_name:
            data = get_cross_cnt2(data, col_name, base_name)
            print('------还在为主人统计user_item特征┏ (^ω^)=☞：' + col_name + base_name)
    for col_name in shop_name:
        for base_name in item_name:
            data = get_cross_cnt2(data, col_name, base_name)
            print('------还在为主人统计shop_item特征┏ (^ω^)=☞：' + col_name + base_name)
    for col_name in user_name:
        for base_name in shop_name:
            data = get_cross_cnt2(data, col_name, base_name)
            print('------还在为主人统计user_shop特征┏ (^ω^)=☞：' + col_name + base_name)

    for col_name in all_name:
        for base_name in ['hour']:
            data = get_cross_cnt2(data, col_name, base_name)
            print('------最后一个！all_hour特征(≖ω≖✿)：' + col_name + base_name)
    return data

# 调用前面的函数
data = shop_fenduan(data)
data = baseprocess(data)
data = get_all_cnt(data)
data = get_all_cross_cnt(data)
data.to_csv('C:/Users/Amber/Desktop/data_feature.csv')
'''
###------------5、Adaptive Lasso
import pandas as pd
import numpy as np
data_feature = pd.read_csv('C:/Users/Amber/Desktop/ad_data/data_feature.csv')
# print(data_feature)
# print(data_feature.columns.tolist())

data_feature.drop(['Unnamed: 0'], axis=1, inplace=True)  # 删除第一列原本生成的index。若设置参数inplace=True，则原数据发生改变
# print(data_feature.head(1))
# print(data_feature.shape) #(478087, 376)

data_feature.drop(['instance_id',
                   'item_id','item_category_list','item_property_list','item_brand_id','item_city_id',
				    'user_id',
					'context_id','context_timestamp','context_page_id','predict_category_property',
					'shop_id','shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description',
					'realtime'],axis=1,inplace=True)  # 删除不需要的特征17个 不删'is_trade'
y = data_feature['is_trade'].values
data_feature.drop(['is_trade'], axis=1, inplace=True)
x = data_feature
# lasso
from sklearn.linear_model import Lasso
factor_name = data_feature.columns.tolist()
x.fillna('0')
y.fillna('0')
model = Lasso()
model.fit(x, y)
result = model.coef_  # 各个特征的系数
factor_name = np.array(factor_name)  # list---np.array
data = pd.DataFrame(factor_name, result)

