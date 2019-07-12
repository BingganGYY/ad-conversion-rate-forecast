#数据可视化（不放入正文代码里）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import os #os模块就是对操作系统进行操作,使用该模块必须先导入模块: import os #getcwd() 获取当前工作目录
import arrow as ar #
import seaborn as sns #
from pyplotz.pyplotz import PyplotZ #一个优化matplotlib函数操作的package
plt.style.use('fivethirtyeight')
from palettable.colorbrewer.sequential import Blues_9,BuGn_9,Greys_3,PuRd_5#三种配色的调色板
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline
#os.chdir('H:\IJCAI')
pltz=PyplotZ()
import matplotlib
myfont = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')#输出图片有中文，不乱码
train = pd.read_csv('C:/Users/Amber/Desktop/ad_data/ad_train.txt',sep=' ')#最开始最真实的train

train = pd.read_csv('C:/Users/Amber/Desktop/ad_data/ad_train.txt',sep=' ')
#1、基础数据
#print(train.head())
#print(type(train))
#print(train.shape)#最开始最真实的train train：(478138, 27)
#print('训练数据集一共有'+str(len(train))+'个样本')
#print('训练标签的比例为'+str(len(train[train.is_trade==0])/len(train[train.is_trade==1])))
#print('数据中有'+str(len(train['item_id'].unique()))+'个不同的广告商品，'+str(len(train['user_id'].unique()))+'个不同的用户和'+str(len(train['shop_id'].unique()))+'个不同的商铺')
#探查下出现频率最高的各类型id
for x in ['instance_id','is_trade','item_id','user_id','context_id','shop_id']:
    print(train[x].value_counts().head())
#分析：
#使用饼图,看看样本正负比例
f,ax=plt.subplots(1,2,figsize=(14,6))
train['is_trade'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('')
ax[0].set_ylabel('')
ax[0].legend(fontsize=7.5)
sns.countplot('is_trade',data=train,ax=ax[1])
ax[1].set_title(u'trade正负个数', fontproperties = myfont)
plt.show()
#商品分布图
fig, axis1 = plt.subplots(1,1,figsize=(10,6))
item_num=pd.DataFrame({'item_id_num':train['item_id'].value_counts().values})
print(item_num)
sns.countplot(x='item_id_num',data=item_num[item_num['item_id_num']<50])
axis1.set_xlabel(u'商品出现的次数', fontproperties = myfont)
axis1.set_ylabel(u'出现n次的商品的数量', fontproperties = myfont)
axis1.set_title(u'商品分布', fontproperties = myfont)

fig, axis1 = plt.subplots(1,1,figsize=(26,6))

item_value=pd.DataFrame(train.item_id.value_counts()).reset_index().head(20)
axis1.set_xlabel('item_id')
axis1.set_ylabel(u'出现次数',fontproperties = myfont)
axis1.set_title(u'top20出现次数的商品',fontproperties = myfont)
y_pos = np.arange(len(item_value))
plt.bar(y_pos, item_value['item_id'], color=(0.2, 0.4, 0.6, 0.6))
pltz.xticks(y_pos, item_value['item_id'])
pltz.show()
#店铺分布图
fig, axis1 = plt.subplots(1, 1, figsize=(14, 6))
shop_num = pd.DataFrame({'shop_id_num': train['shop_id'].value_counts().values})
sns.countplot(x='shop_id_num', data=shop_num[shop_num['shop_id_num'] < 50])
axis1.set_xlabel(u'店铺出现的次数',fontproperties = myfont)
axis1.set_ylabel(u'出现n次的店铺的数量',fontproperties = myfont)
axis1.set_title(u'店铺分布',fontproperties = myfont)

fig, axis1 = plt.subplots(1, 1, figsize=(26, 6))

shop_value = pd.DataFrame(train.shop_id.value_counts()).reset_index().head(20)
axis1.set_xlabel('shop_id')
axis1.set_ylabel(u'出现次数',fontproperties = myfont)
axis1.set_title(u'top20出现次数的店铺',fontproperties = myfont)
y_pos = np.arange(len(shop_value))
plt.bar(y_pos, shop_value['shop_id'], color=(0.2, 0.4, 0.6, 0.6))
pltz.xticks(y_pos, shop_value['shop_id'])
pltz.show()

#2、商品数据
for x in ['item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level']:
    print(train[x].value_counts())
f,ax=plt.subplots(1,1, figsize=(6,6.5))

item_brand_id_num=pd.DataFrame({'brand_id_num':train['item_brand_id'].value_counts()}).reset_index()
brand_value=pd.DataFrame({'brand_id_num':item_brand_id_num['brand_id_num'][item_brand_id_num['brand_id_num']<5000].sum()},index=[0])
brand_value['index']='below_5000_counts_brand'
brand_value=pd.concat([brand_value,item_brand_id_num[item_brand_id_num['brand_id_num']>=5000]])
pd.Series(data=brand_value.set_index('index')['brand_id_num']).plot.pie(autopct='%1.1f%%',ax=ax,shadow=True,colors=Blues_9.hex_colors)
ax.set_title(u'广告商品的品牌分布',fontproperties = myfont)

f,ax=plt.subplots(1,1, figsize=(6,6.5))
item_city_id_num=pd.DataFrame({'city_id_num':train['item_city_id'].value_counts()}).reset_index()
city_value=pd.DataFrame({'city_id_num':item_city_id_num['city_id_num'][item_city_id_num['city_id_num']<5000].sum()},index=[0])
city_value['index']='below_5000_count_city'
city_value=pd.concat([city_value,item_city_id_num[item_brand_id_num['brand_id_num']>=5000]])
pd.Series(data=city_value.set_index('index')['city_id_num']).plot.pie(autopct='%1.1f%%',ax=ax,shadow=True,colors=Blues_9.hex_colors)
ax.set_title(u'广告商品的城市分布',fontproperties = myfont)

f,ax=plt.subplots(2,2, figsize=(6,6.5))
sns.countplot('item_price_level',data=train,ax=ax[0][0])
ax[0][0].set_title(u'广告商品的价格等级分布',fontproperties = myfont)

sns.countplot('item_sales_level',data=train,ax=ax[0][1])
ax[0][1].set_title(u'广告商品的销量等级分布',fontproperties = myfont)

sns.countplot('item_collected_level',data=train,ax=ax[1][0])
ax[1][0].set_title(u'广告商品被收藏次数的等级分布',fontproperties = myfont)

sns.countplot('item_pv_level',data=train,ax=ax[1][1])
ax[1][1].set_title(u'广告商品被展示次数的等级分布',fontproperties = myfont)

plt.show()
#3、用户数据
#这个为准 四个字段全部改成粉红色 哈哈
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
f,ax=plt.subplots(1,2,figsize=(14,6))
train['user_gender_id'].value_counts().sort_index().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True,colors=PuRd_5.hex_colors)
ax[0].set_title(u'用户的预测性别编号分布',fontproperties = myfont)
train['user_age_level'].value_counts().sort_index().plot.pie(autopct='%1.1f%%',ax=ax[1],shadow=True,colors=PuRd_5.hex_colors)
ax[1].set_title(u'用户的预测年龄等级分布',fontproperties = myfont)
f,ax=plt.subplots(1,2,figsize=(14,6))
train['user_occupation_id'].value_counts().sort_index().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True,colors=PuRd_5.hex_colors)
ax[0].set_title(u'用户的预测职业编号分布',fontproperties = myfont)
train['user_star_level'].value_counts().sort_index().plot.pie(autopct='%1.1f%%',ax=ax[1],shadow=True,colors=PuRd_5.hex_colors)
ax[1].set_title(u'用户的星级编号分布',fontproperties = myfont)
plt.show()

#4、店铺信息数据
for x in ['shop_review_num_level','shop_star_level']:
    print(train[x].value_counts())

f,ax=plt.subplots(1,1,figsize=(14,6))
sns.countplot('shop_review_num_level',data=train,ax=ax)
ax.set_xlabel(u'店铺的评价数量等级',fontproperties = myfont)
ax.set_ylabel(u'计数',fontproperties = myfont)
ax.set_title(u'店铺的评价数量等级分布',fontproperties = myfont)

f,ax=plt.subplots(1,1,figsize=(14,6))
sns.countplot('shop_star_level',data=train,ax=ax)
ax.set_xlabel(u'星级编号',fontproperties = myfont)
ax.set_ylabel(u'计数',fontproperties = myfont)
ax.set_title(u'店铺的星级编号分布',fontproperties = myfont)

f,ax=plt.subplots(2,2,figsize=(6,6.5))
plt.style.use('ggplot')
plt.tight_layout(5)

sns.distplot(train['shop_review_positive_rate'][train['shop_review_positive_rate']>0.9],ax=ax[0][0])
ax[0][0].set_xlabel(u'好评率',fontproperties = myfont)
ax[0][0].set_ylabel(u'计数',fontproperties = myfont)
ax[0][0].set_title(u'店铺的好评率分布',fontproperties = myfont)

sns.distplot(train['shop_score_service'][train['shop_score_service']>0.9],ax=ax[0][1])
ax[0][1].set_xlabel(u'服务态度评分',fontproperties = myfont)
ax[0][1].set_ylabel(u'计数',fontproperties = myfont)
ax[0][1].set_title(u'店铺的服务态度评分分布',fontproperties = myfont)

sns.distplot(train['shop_score_delivery'][train['shop_score_delivery']>0.9],ax=ax[1][0])
ax[1][0].set_xlabel(u'物流服务评分',fontproperties = myfont)
ax[1][0].set_ylabel(u'计数',fontproperties = myfont)
ax[1][0].set_title(u'店铺的物流服务评分分布',fontproperties = myfont)

sns.distplot(train['shop_score_description'][train['shop_score_description']>0.9],ax=ax[1][1])
ax[1][1].set_xlabel(u'描述相符评分',fontproperties = myfont)
ax[1][1].set_ylabel(u'计数',fontproperties = myfont)
ax[1][1].set_title(u'店铺的描述相符评分分布',fontproperties = myfont)
plt.show()

#CVR of day and hour
train = pd.read_csv('C:/Users/Amber/Desktop/ad_data/ad_train.txt',sep=' ')#最开始最真实的train
import time
train["time_sting"]=train["context_timestamp"].apply(lambda x:time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(x)))
train["time_sting"]=pd.to_datetime(train["time_sting"])
train["hour"]=train["time_sting"].dt.hour
train["day"]=train["time_sting"].dt.day
train["day"]=train["day"].apply(lambda x:0 if x==31 else x)

grouped_df = train.groupby(["day", "hour"])["is_trade"].aggregate("mean").reset_index()
grouped_df = grouped_df.pivot('day', 'hour', 'is_trade')

plt.figure(figsize=(12,6))
sns.heatmap(grouped_df)
plt.title("CVR of Day Vs Hour")
plt.show()
