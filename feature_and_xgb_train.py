import pandas as pd
import numpy as np
import gc
import time
import xgboost as xgb
import os
#导入相关包
print('ok')

pd.set_option('display.height',100)
pd.set_option('display.max_rows',50)
pd.set_option('display.max_columns',50)
pd.set_option('display.width',100)

gc.collect()
path = "/mnt/datasets/fusai/"
mypath = "/mnt/datasets/fusai/xuke/"
submit_path = "/home/kesci/work/"
# 设定默认路径
path = "/mnt/datasets/fusai/"
mypath = "/mnt/datasets/fusai/xuke/"
submit_path = "/home/kesci/work/"
#os.makedirs('xuke')
#os.chdir('xuke')
!ls .
gc.collect()

columns_app = ['user_id', 'day']
columns_activity = ['user_id', 'day', 'page', 'video_id', 'author_id', 'action_type']
columns_register = ['user_id', 'day', 'register_type', 'device_type']
columns_video = ['user_id', 'day']

app_launch_path = 'app_launch_log.txt'
user_activity_path = 'user_activity_log.txt'
user_register_path = 'user_register_log.txt'
video_create_path = 'video_create_log.txt'





launch_app = pd.read_table(path+app_launch_path, names=columns_app).sort_values(['user_id', 'day'], ascending=True)
activity = pd.read_table(path+user_activity_path, names=columns_activity).sort_values(['user_id', 'day'], ascending=True)
register = pd.read_table(path+user_register_path, names=columns_register).sort_values(['user_id', 'day'], ascending=True)
create_video = pd.read_table(path+video_create_path, names=columns_video).sort_values(['user_id', 'day'], ascending=True)
print("数据加载完毕！！")






def get_user_id(begin_day, end_day):

    temp_app = launch_app[(launch_app['day'] >= begin_day) & (launch_app['day'] <= end_day)]
    temp_activity = activity[(activity['day'] >= begin_day) & (activity['day'] <= end_day)]
    temp_video = create_video[(create_video['day'] >= begin_day) & (create_video['day'] <= end_day)]

    user_id_app = pd.DataFrame(temp_app['user_id'].drop_duplicates())
    user_id_activity = pd.DataFrame(temp_activity['user_id'].drop_duplicates())
    user_id_video = pd.DataFrame(temp_video['user_id'].drop_duplicates())
    # 合并，再去重----得到这个窗口的所有用户
    user = pd.concat([user_id_app, user_id_activity, user_id_video], ignore_index=True)
    user = pd.DataFrame(user['user_id'].drop_duplicates())
    user.index = range(len(user))
    # print(user.info())
    # print(user.head(100))
    del temp_app
    del temp_activity
    del temp_video
    gc.collect()
    del user_id_app
    del user_id_activity
    del user_id_video
    gc.collect()
    return user


def create_label(train_user, test_user):
    bool_ = train_user['user_id'].isin(test_user['user_id'])
    train_user['label'] = 0
    train_user['label'][bool_] = 1
    # print(train_user.info())
    # print(train_user.head(100))
    
    del bool_
    gc.collect()
    return train_user

#初始化数据
gc.collect()

!ls .
print('提取1-16的label：')
# 以1-16天的数据 预测17-23某用户是否活跃
train_user = get_user_id(1, 16)
test_user = get_user_id(17, 23)
label_1_16 = create_label(train_user, test_user)
label_1_16.to_csv('xk_label_1_16.csv', index=None)
print(len(label_1_16))
print(label_1_16.head(10))
del label_1_16
gc.collect()
print("提取完毕！！")


print('提取8-23的label：')
# 以8-23天的数据 预测24-30某用户是否活跃
train_user = get_user_id(8, 23)
test_user = get_user_id(24, 30)
label_8_23 = create_label(train_user, test_user)
label_8_23.to_csv('xk_label_8_23.csv', index=None)
print(len(label_8_23))
print(label_8_23.head(10))
del label_8_23
gc.collect()
print("提取完毕！！")


print("提取1-30天的用户：")
label_1_30 = get_user_id(1, 30)
label_1_30.to_csv('xk_label_1_30.csv', index=None)
print(len(label_1_30))
print(label_1_30.head(10))
del label_1_30
gc.collect()
print("提取完毕！！")

del train_user
del test_user
gc.collect()





def actions_numbers(activity,f_begin_day,f_end_day):
    #获取activity操作总数及其平均值
    # list_userid = activity['user_id'].unique()
    # print('共有',len(list_userid),'个用户')
    feature = activity[["user_id", "day"]].groupby("user_id").count().reset_index()
    feature.columns = ["user_id", "actions_numbers"]
    feature['actions_numbers_mean'] = (feature['actions_numbers']/(f_end_day-f_begin_day)).map(lambda x:round(x, 2))
    print(feature.head())
    return feature

def actions_last_date(activity,f_begin_day,f_end_day):
    #获取activity操作最后日期，以及最后一天距离现在的天数
    action_last = activity[["user_id", "day"]]
    action_last = action_last.groupby("user_id").apply(lambda x: x.iloc[-1])
    action_last.columns = ["user_id", "actions_last_date"]
    action_last['actions_last_date_from_now'] = f_end_day - action_last['actions_last_date']
    print(action_last.head())
    return action_last
    
def launch_app_numbers(launch_app,f_begin_day,f_end_day):
    #获取lauch_app操作总数以及平均值
    feature = launch_app[["user_id", "day"]].groupby("user_id").count().reset_index()
    feature.columns = ["user_id", "launch_app_numbers"]
    feature['launch_app_numbers_mean'] = (feature['launch_app_numbers']/(f_end_day-f_begin_day)).map(lambda x:round(x, 2))
    print(feature.head())
    return feature
    
def launch_app_last_date(lauch_app,f_begin_day,f_end_day):
    #获取lauch_app操作最后日期，以及最后一天距离现在的天数
    lauch_app = lauch_app[["user_id", "day"]]
    lauch_app = lauch_app.groupby("user_id").apply(lambda x: x.iloc[-1])
    lauch_app.columns = ["user_id", "lauch_app_last_date"]
    lauch_app['lauch_app_last_date_from_now'] = f_end_day - lauch_app['lauch_app_last_date']
    print(lauch_app.head())
    return lauch_app
    
def create_video_numbers(create_video,f_begin_day,f_end_day):
    #获取create_video操作总数以及平均值
    feature = create_video[["user_id", "day"]].groupby("user_id").count().reset_index()
    feature.columns = ["user_id", "create_video_numbers"]
    feature['create_video_numbers_mean'] = (feature['create_video_numbers']/(f_end_day-f_begin_day)).map(lambda x:round(x, 2))
    print(feature.head())
    return feature
    
def create_video_last_date(create_video,f_begin_day,f_end_day):
    #获取create_video操作最后日期，以及最后一天距离现在的天数
    create_video = create_video[["user_id", "day"]]
    create_video = create_video.groupby("user_id").apply(lambda x: x.iloc[-1])
    create_video.columns = ["user_id", "create_video_last_date"]
    create_video['create_video_last_date_from_now'] = f_end_day - create_video['create_video_last_date']
    print(create_video.head())
    return create_video
    
def register_last_date(register,f_begin_day,f_end_day):
    #获取register操作最后日期，以及最后一天距离现在的天数
    register = register[["user_id", "day"]]
    register = register.groupby("user_id").apply(lambda x: x.iloc[-1])
    register.columns = ["user_id", "register_last_date"]
    register['register_last_date_from_now'] = f_end_day - register['register_last_date']
    print(register.head())
    return register

def watched_by_others_num(activity,f_begin_day,f_end_day):
    #获取用户作为作者被别人看的次数
    feature = activity[["author_id", "day"]].groupby("author_id").count().reset_index()
    feature.columns = ["user_id", "watched_by_others_num"]
    print(feature)
    return feature

def last_continue_activity_from_now(activity,f_begin_day,f_end_day):
    #上一次连续操作的日子，以及距离现在的天数
    activity = activity[activity.duplicated(['user_id','day'])==True]
    activity = activity[['user_id', 'day']].groupby(['user_id']).max().reset_index()
    activity.columns = ['user_id','last_continue_activity_day']
    activity['last_continue_activity_from_now'] = f_end_day - activity['last_continue_activity_day']
    print(activity)
    return activity
    
def get_watch_self_feature(activity,f_begin_day,f_end_day):
    #用户自己看自己的次数
    activity = activity[activity['user_id'] == activity['author_id']]
    print(activity)
    activity = activity[['user_id', 'day']].groupby(['user_id']).count().reset_index()
    activity.columns = ['user_id','watch_self_nums']
    print(activity)
    return activity
    
def activity_days_nums(activity,f_begin_day,f_end_day):#线下没用
    #操作的总天数
    activity = activity[activity.duplicated(['user_id','day'])==False]
    activity = activity[['user_id', 'day']].groupby(['user_id']).count().reset_index()
    activity.columns = ['user_id','activity_days_nums']
    print(activity)
    return activity

def actions_first_date(activity,f_begin_day,f_end_day):
    #获取activity操作第一次日期，以及距离现在的天数
    actions_first = activity[["user_id", "day"]]
    actions_first = actions_first.groupby("user_id").apply(lambda x: x.iloc[0])
    actions_first.columns = ["user_id", "actions_first_date"]
    actions_first['actions_first_date_from_now'] = f_end_day - actions_first['actions_first_date']
    print(actions_first.head())
    return actions_first
    
def launch_app_first_date(launch_app,f_begin_day,f_end_day):
    #获取launch_app操作第一次日期，以及距离现在的天数
    launch_app_first = launch_app[["user_id", "day"]]
    launch_app_first = launch_app_first.groupby("user_id").apply(lambda x: x.iloc[0])
    launch_app_first.columns = ["user_id", "launch_app_first_date"]
    launch_app_first['launch_app_first_first_date_from_now'] = f_end_day - launch_app_first['launch_app_first_date']
    print(launch_app_first.head())
    return launch_app_first
    
def create_video_first_date(create_video,f_begin_day,f_end_day):
    #获取create_video操作第一次日期，以及距离现在的天数
    create_video = create_video[["user_id", "day"]]
    create_video = create_video.groupby("user_id").apply(lambda x: x.iloc[0])
    create_video.columns = ["user_id", "create_video_first_date"]
    create_video['create_video_first_date_from_now'] = f_end_day - create_video['create_video_first_date']
    print(create_video.head())
    return create_video
    
def actions_type_i_numbers(activity,f_begin_day,f_end_day,i):
    #获取type i 操作数目
    activity = activity[activity['action_type']==i]
    feature = activity[["user_id", "action_type"]].groupby("user_id").count().reset_index()
    names = 'actions_action_type_'+str(i)+'_numbers'
    name_mean = names+'_mean'
    feature.columns = ["user_id", names]
    print(feature)
    feature[name_mean] = (feature[names]/(f_end_day-f_begin_day)).map(lambda x:round(x, 2))
    print(feature)
    return feature
    
def actions_page_numbers(activity,f_begin_day,f_end_day):
    features = pd.DataFrame()
    features['user_id'] = activity['user_id'].unique()
    for i in range(5):
        #获取page操作数目
        activity = activity[activity['page']==i]
        feature = activity[["user_id", "page"]].groupby("user_id").count().reset_index()
        names = 'actions_page_'+str(i)+'_numbers'
        name_mean = names+'_mean'
        feature.columns = ["user_id", names]
        print(feature)
        feature[name_mean] = (feature[names]/(f_end_day-f_begin_day)).map(lambda x:round(x, 2))
        features = pd.merge(features,feature,on='user_id',how='left')
    print(features)
    return features

def actions_page_0_numbers(activity,f_begin_day,f_end_day):
    #获取page0操作数目
    i = 0
    activity = activity[activity['page']==0]
    feature = activity[["user_id", "page"]].groupby("user_id").count().reset_index()
    names = 'actions_page_'+str(i)+'_numbers'
    name_mean = names+'_mean'
    feature.columns = ["user_id", names]
    print(feature)
    feature[name_mean] = (feature[names]/(f_end_day-f_begin_day)).map(lambda x:round(x, 2))
    print(feature)
    return feature
    
def actions_page_1_numbers(activity,f_begin_day,f_end_day):
    #获取page1操作数目
    i = 1
    activity = activity[activity['page']==1]
    feature = activity[["user_id", "page"]].groupby("user_id").count().reset_index()
    names = 'actions_page_'+str(i)+'_numbers'
    name_mean = names+'_mean'
    feature.columns = ["user_id", names]
    print(feature)
    feature[name_mean] = (feature[names]/(f_end_day-f_begin_day)).map(lambda x:round(x, 2))
    print(feature)
    return feature
    
def actions_page_2_numbers(activity,f_begin_day,f_end_day):
    #获取page2操作数目
    i = 2
    activity = activity[activity['page']==2]
    feature = activity[["user_id", "page"]].groupby("user_id").count().reset_index()
    names = 'actions_page_'+str(i)+'_numbers'
    name_mean = names+'_mean'
    feature.columns = ["user_id", names]
    print(feature)
    feature[name_mean] = (feature[names]/(f_end_day-f_begin_day)).map(lambda x:round(x, 2))
    print(feature)
    return feature
    
def actions_page_3_numbers(activity,f_begin_day,f_end_day):
    #获取page3操作数目
    i = 3
    activity = activity[activity['page']==3]
    feature = activity[["user_id", "page"]].groupby("user_id").count().reset_index()
    names = 'actions_page_'+str(i)+'_numbers'
    name_mean = names+'_mean'
    feature.columns = ["user_id", names]
    print(feature)
    feature[name_mean] = (feature[names]/(f_end_day-f_begin_day)).map(lambda x:round(x, 2))
    print(feature)
    return feature

def actions_page_4_numbers(activity,f_begin_day,f_end_day):
    #获取page4操作数目
    i = 4
    activity = activity[activity['page']==4]
    feature = activity[["user_id", "page"]].groupby("user_id").count().reset_index()
    names = 'actions_page_'+str(i)+'_numbers'
    name_mean = names+'_mean'
    feature.columns = ["user_id", names]
    print(feature)
    feature[name_mean] = (feature[names]/(f_end_day-f_begin_day)).map(lambda x:round(x, 2))
    print(feature)
    return feature

def actions_window_numbers(activity,f_begin_day,f_end_day):
    #划窗获取create_video操作总数以及平均值
    features = pd.DataFrame()
    features['user_id'] = activity['user_id'].unique()
    window = [1,3,5,7]
    for i in window:
        feature = activity[activity['day']>f_end_day-i]
        feature = feature[["user_id", "day"]].groupby("user_id").count().reset_index()
        names = 'actions_pre_'+str(i)+'_numbers'
        feature.columns = ["user_id", names]
        names1 = names+'_mean'
        feature[names1] = (feature[names]/(i)).map(lambda x:round(x, 2))
        features = pd.merge(features,feature,on='user_id',how='left')
    print(features.head())
    return features
    
def lauch_app_window_numbers(lauch_app,f_begin_day,f_end_day):
    #划窗获取lauch_app操作总数以及平均值
    features = pd.DataFrame()
    features['user_id'] = lauch_app['user_id'].unique()
    window = [7]
    for i in window:
        feature = lauch_app[lauch_app['day']>f_end_day-i]
        feature = feature[["user_id", "day"]].groupby("user_id").count().reset_index()
        names = 'lauch_app_pre_'+str(i)+'_numbers'
        feature.columns = ["user_id", names]
        names1 = names+'_mean'
        feature[names1] = (feature[names]/(i)).map(lambda x:round(x, 2))
        features = pd.merge(features,feature,on='user_id',how='left')
    print(features.head())
    return features

def last_continue_launch_app_from_now(launch_app,f_begin_day,f_end_day):
    #上一次连续launch_app的日子，以及距离现在的天数
    launch_app = launch_app[launch_app.duplicated(['user_id','day'])==True]
    launch_app = launch_app[['user_id', 'day']].groupby(['user_id']).max().reset_index()
    launch_app.columns = ['user_id','last_continue_launch_app_day']
    launch_app['last_continue_launch_app_from_now'] = f_end_day - launch_app['last_continue_launch_app_day']
    print(launch_app)
    return launch_app
    
def launch_app_days_nums(launch_app,f_begin_day,f_end_day):#
    #launch_app的总天数
    launch_app = launch_app[launch_app.duplicated(['user_id','day'])==False]
    launch_app = launch_app[['user_id', 'day']].groupby(['user_id']).count().reset_index()
    launch_app.columns = ['user_id','launch_app_days_nums']
    print(launch_app)
    return launch_app
    
    
def launch_app_average_internal(launch_app,f_begin_day,f_end_day):#
    #launch_app的平均间隔
    # launch_app = launch_app[launch_app.duplicated(['user_id','day'])==False]
    launch_app = (launch_app[['user_id', 'day']].groupby(['user_id']).count()/
        (launch_app[['user_id', 'day']].groupby(['user_id']).max()
        -launch_app[['user_id', 'day']].groupby(['user_id']).min())).applymap(lambda x: 100 if x == float('inf') else x).reset_index()
    launch_app.columns = ['user_id','launch_app_average_internal_temp']
    launch_app['launch_app_average_internal'] = (launch_app['launch_app_average_internal_temp']/1.0).map(lambda x:round(x, 2))
    print(launch_app['launch_app_average_internal'])
    launch_app.pop('launch_app_average_internal_temp')
    print(launch_app)
    return launch_app
    
def activity_average_internal(activity,f_begin_day,f_end_day):#
    #activity的平均间隔
    activity = activity[activity.duplicated(['user_id','day'])==False]
    activity = (activity[['user_id', 'day']].groupby(['user_id']).count()/
        (activity[['user_id', 'day']].groupby(['user_id']).max()
        -activity[['user_id', 'day']].groupby(['user_id']).min())).applymap(lambda x: 15 if x == float('inf') else x).reset_index()
    activity.columns = ['user_id','activity_average_internal_temp']
    activity['activity_average_internal'] = (activity['activity_average_internal_temp']/1.0).map(lambda x:round(x, 2))
    print(activity['activity_average_internal'])
    activity.pop('activity_average_internal_temp')
    print(activity)
    return activity
    
def activity_authors_nums(activity,f_begin_day,f_end_day):#线下没用
    #操作的作者的个数
    activity = activity[activity.duplicated(['user_id','author_id'])==False]
    activity = activity[['user_id', 'author_id']].groupby(['user_id']).count().reset_index()
    activity.columns = ['user_id','activity_authors_nums']
    print(activity)
    return activity
    
def activity_authors_window_numbers(activity,f_begin_day,f_end_day):
    #划窗获取activity_authors总数以及平均值
    features = pd.DataFrame()
    features['user_id'] = activity['user_id'].unique()
    window = [1,3,5,7]
    for i in window:
        feature = activity[activity['day']>f_end_day-i]
        feature = feature[feature.duplicated(['user_id','author_id'])==False]
        feature = feature[['user_id', 'author_id']].groupby(['user_id']).count().reset_index()
        names = 'activity_authors_pre_'+str(i)+'_numbers'
        feature.columns = ["user_id", names]
        names1 = names+'_mean'
        feature[names1] = (feature[names]/(i)).map(lambda x:round(x, 2))
        features = pd.merge(features,feature,on='user_id',how='left')
    print(features.head())
    return features
    
def video_id_feature(activity,f_begin_day,f_end_day):#
    #观看video_id的特征
    video_id_count = activity.groupby(['user_id','video_id']).agg({'user_id':'mean','video_id':'count'})
    video_id_max = video_id_count.groupby(['user_id'])['video_id'].max().rename('video_id_max').reset_index()
    video_id_min = video_id_count.groupby(['user_id'])['video_id'].min().rename('video_id_min').reset_index()
    video_id_mean = video_id_count.groupby(['user_id'])['video_id'].mean().rename('video_id_mean').reset_index()
    video_id_std = video_id_count.groupby(['user_id'])['video_id'].std().rename('video_id_std').reset_index()
    video_id_kurt = video_id_count.groupby(['user_id'])['video_id'].agg(lambda x: pd.Series.kurt(x)).rename('video_id_kurt').reset_index()
    video_id_skew = video_id_count.groupby(['user_id'])['video_id'].skew().rename('video_id_skew').reset_index()
    video_id_last = video_id_count.groupby(['user_id'])['video_id'].last().rename('video_id_last').reset_index()
    video_id = pd.merge(video_id_max,video_id_min,how='left',on='user_id')
    video_id = pd.merge(video_id,video_id_mean,how='left',on='user_id')
    video_id = pd.merge(video_id,video_id_std,how='left',on='user_id')
    video_id = pd.merge(video_id,video_id_kurt,how='left',on='user_id')
    video_id = pd.merge(video_id,video_id_skew,how='left',on='user_id')
    video_id = pd.merge(video_id,video_id_last,how='left',on='user_id')
    video_id['video_id_mean'] = video_id['video_id_mean'].map(lambda x:round(x, 2))
    video_id['video_id_std'] = video_id['video_id_std'].map(lambda x:round(x, 2))
    video_id['video_id_kurt'] = video_id['video_id_kurt'].map(lambda x:round(x, 2))
    video_id['video_id_skew'] = video_id['video_id_skew'].map(lambda x:round(x, 2))
    print(video_id)
    return video_id
    
def author_id_feature(activity,f_begin_day,f_end_day):#
    #观看author_id的特征
    author_id_count = activity.groupby(['user_id','author_id']).agg({'user_id':'mean','author_id':'count'})
    author_id_max = author_id_count.groupby(['user_id'])['author_id'].max().rename('author_id_max').reset_index()
    author_id_min = author_id_count.groupby(['user_id'])['author_id'].min().rename('author_id_min').reset_index()
    author_id_mean = author_id_count.groupby(['user_id'])['author_id'].mean().rename('author_id_mean').reset_index()
    author_id_std = author_id_count.groupby(['user_id'])['author_id'].std().rename('author_id_std').reset_index()
    author_id_kurt = author_id_count.groupby(['user_id'])['author_id'].agg(lambda x: pd.Series.kurt(x)).rename('author_id_kurt').reset_index()
    author_id_skew = author_id_count.groupby(['user_id'])['author_id'].skew().rename('author_id_skew').reset_index()
    author_id_last = author_id_count.groupby(['user_id'])['author_id'].last().rename('author_id_last').reset_index()
    author_id = pd.merge(author_id_max,author_id_min,how='left',on='user_id')
    author_id = pd.merge(author_id,author_id_mean,how='left',on='user_id')
    author_id = pd.merge(author_id,author_id_std,how='left',on='user_id')
    author_id = pd.merge(author_id,author_id_kurt,how='left',on='user_id')
    author_id = pd.merge(author_id,author_id_skew,how='left',on='user_id')
    author_id = pd.merge(author_id,author_id_last,how='left',on='user_id')
    author_id['author_id_mean'] = author_id['author_id_mean'].map(lambda x:round(x, 2))
    author_id['author_id_std'] = author_id['author_id_std'].map(lambda x:round(x, 2))
    author_id['author_id_kurt'] = author_id['author_id_kurt'].map(lambda x:round(x, 2))
    author_id['author_id_skew'] = author_id['author_id_skew'].map(lambda x:round(x, 2))
    print(author_id)
    return author_id
    
def video_slide_window(video):
    video_total_count = video[['user_id']].groupby(['user_id']).size().rename('video_total_count').reset_index()
    return video_total_count  
    
def create_video_slide(video,f_begin_day,f_end_day):#
    #滑窗 窗口内video次数
    feature = pd.DataFrame()
    feature['user_id'] = video['user_id'].drop_duplicates()
    end_day = f_end_day
    #滑窗 窗口内video次数
    # video_total_count_all=video_slide_window(video)
    video_total_count_1=video_slide_window(video[(video['day']>=end_day) & (video['day']<=end_day)])
    video_total_count_2=video_slide_window(video[(video['day']>=end_day-1) & (video['day']<=end_day)])
    video_total_count_3=video_slide_window(video[(video['day']>=end_day-2) & (video['day']<=end_day)])
    video_total_count_5=video_slide_window(video[(video['day']>=end_day-4) & (video['day']<=end_day)])
    video_total_count_7=video_slide_window(video[(video['day']>=end_day-6) & (video['day']<=end_day)])
    video_total_count_10=video_slide_window(video[(video['day']>=end_day-9) & (video['day']<=end_day)])
    feature = pd.merge(feature,video_total_count_1,how='left',on='user_id')
    feature = feature.rename({'video_total_count':'video_total_count_1'},axis=1)
    feature = pd.merge(feature,video_total_count_2,how='left',on='user_id')
    feature = feature.rename({'video_total_count':'video_total_count_2'},axis=1)
    feature = pd.merge(feature,video_total_count_3,how='left',on='user_id')
    feature = feature.rename({'video_total_count':'video_total_count_3'},axis=1)
    feature = pd.merge(feature,video_total_count_5,how='left',on='user_id')
    feature = feature.rename({'video_total_count':'video_total_count_5'},axis=1)
    feature = pd.merge(feature,video_total_count_7,how='left',on='user_id')
    feature = feature.rename({'video_total_count':'video_total_count_7'},axis=1)
    feature = pd.merge(feature,video_total_count_10,how='left',on='user_id')
    feature = feature.rename({'video_total_count':'video_total_count_10'},axis=1)
    print(feature)
    return feature  
    
    
def create_video_date(video,f_begin_day,f_end_day):#
    #滑窗 窗口内video次数
    feature = pd.DataFrame()
    feature['user_id'] = video['user_id'].drop_duplicates()
    end_day = f_end_day
    video_count = video.groupby(['user_id','dat']).agg({'video_day':'count'}).rename({'day':'video_count'},axis=1).reset_index()
    video_count_max = video_count.groupby(['user_id'])['video_count'].max().rename('video_count_max').reset_index()
    video_count_min = video_count.groupby(['user_id'])['video_count'].min().rename('video_count_min').reset_index()
    video_count_mean = video_count.groupby(['user_id'])['video_count'].mean().rename('video_count_mean').reset_index()
    video_count_std = video_count.groupby(['user_id'])['video_count'].std().rename('video_count_std').reset_index()
    video_count_kurt = video_count.groupby(['user_id'])['video_count'].agg(lambda x: pd.Series.kurt(x)).rename('video_count_kurt').reset_index()
    video_count_skew = video_count.groupby(['user_id'])['video_count'].skew().rename('video_count_skew').reset_index()
    video_count_last = video_count.groupby(['user_id'])['video_count'].last().rename('video_count_last').reset_index()

    feature = pd.merge(feature,video_count_max,how='left',on='user_id')
    feature = pd.merge(feature,video_count_min,how='left',on='user_id')
    feature = pd.merge(feature,video_count_mean,how='left',on='user_id')
    feature = pd.merge(feature,video_count_std,how='left',on='user_id')
    feature = pd.merge(feature,video_count_kurt,how='left',on='user_id')
    feature = pd.merge(feature,video_count_skew,how='left',on='user_id')
    feature = pd.merge(feature,video_count_last,how='left',on='user_id')

    author_id['video_count_std'] = author_id['video_count_std'].map(lambda x:round(x, 2))
    author_id['video_count_kurt'] = author_id['video_count_kurt'].map(lambda x:round(x, 2))
    author_id['video_count_mean'] = author_id['video_count_mean'].map(lambda x:round(x, 2))
    author_id['author_id_skew'] = author_id['author_id_skew'].map(lambda x:round(x, 2))
    print(feature)
    return feature  

    
def gen_features(begin_day,end_day,f_begin_day,f_end_day):
    #读取label
    features = pd.read_csv('label_' + str(begin_day) + '_' + str(end_day) + '.csv')
    #合并注册类别、设备类别特征
    register = pd.read_table(path+user_register_path, names=columns_register).sort_values(['user_id', 'day'],
    ascending=True)
    features = pd.merge(features, register[['user_id','register_type','device_type']],on='user_id',how='left')
    ###############################处理activity特征#########################
    
    activity = pd.read_table(path+user_activity_path, names=columns_activity).sort_values(['user_id', 'day'],
    ascending=True)
    activity = activity[(activity['day'] >= f_begin_day) & (activity['day'] <= f_end_day)]
    #获取activity操作最后日期，以及最后一天距离现在的天数
    feature = actions_last_date(activity,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    #获取activity操作总数及其平均值
    feature = actions_numbers(activity,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    #获取用户作为作者被别人看的次数
    feature = watched_by_others_num(activity,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    del activity
    gc.collect()
    ###############################处理launch_app特征#########################
    
    launch_app = pd.read_table(path+app_launch_path, names=columns_app).sort_values(['user_id', 'day'], 
    ascending=True)
    launch_app = launch_app[(launch_app['day'] >= f_begin_day) & (launch_app['day'] <= f_end_day)]
    #获取lauch_app操作最后日期，以及最后一天距离现在的天数
    feature = launch_app_last_date(launch_app,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    #获取lauch_app操作总数及其平均值
    feature = launch_app_numbers(launch_app,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    
    del launch_app
    gc.collect()
    ###############################处理register特征#########################
    register = pd.read_table(path+user_register_path, names=columns_register).sort_values(['user_id', 'day'],
    ascending=True)
    register = register[(register['day'] >= f_begin_day) & (register['day'] <= f_end_day)]
    #获取register操作最后日期，以及最后一天距离现在的天数
    feature = register_last_date(register,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    del register
    gc.collect()
    ###############################处理create_video特征#########################
    create_video = pd.read_table(path+video_create_path, names=columns_video).sort_values(['user_id', 'day'],
    ascending=True)
    create_video = create_video[(create_video['day'] >= f_begin_day) & (create_video['day'] <= f_end_day)]
    #获取create_video操作最后日期，以及最后一天距离现在的天数
    feature = create_video_last_date(create_video,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    #获取create_video操作总数及其平均值
    feature = create_video_numbers(create_video,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    del create_video
    gc.collect()
    
    
    return features
    
def add_feature(begin_day,end_day,f_begin_day,f_end_day):
    features = pd.read_csv('xk_feature_'+str(begin_day)+'_'+str(end_day)+'.csv')
    # try:

    # features.pop('video_total_count_x')
    # features.pop('video_total_count_y')
    # features.pop('video_total_count_x.1')
    # features.pop('video_total_count_y.1')
    # features.pop('video_total_count_x.2')
    # features.pop('video_total_count_y.2')
    # features.pop('video_id_last')
    # features.pop('actions_page_3_numbers_mean')
    # features.pop('actions_page_4_numbers')
    # features.pop('actions_page_4_numbers_mean')
    # except:
    #     features.pop('actions_page_0_numbers')
    #     features.pop('actions_page_0_numbers_mean')
    # features.pop('actions_type_2_numbers')
    # features.pop('actions_type_2_numbers_mean')
    # features.pop('actions_type_3_numbers')
    # features.pop('actions_type_3_numbers_mean')
    # features.pop('actions_type_4_numbers')
    # features.pop('actions_type_4_numbers_mean')
    # features.pop('actions_type_5_numbers')
    # features.pop('actions_type_5_numbers_mean')
    # try:
    #     # features.pop('last_continue_launch_app_day')
    #     # features.pop('launch_app_average_internal_y')
    #     # features.pop('launch_app_average_internal')
    #     # features.pop('launch_app_average_internal_x')
    #     # features.pop('lauch_app_pre_5_numbers_mean')
    #     # features.pop('lauch_app_pre_5_numbers')
    #     # features.pop('lauch_app_pre_7_numbers_mean')
    #     # features.pop('lauch_app_pre_7_numbers')
    # except:
    #     pass
    ###############################处理activity特征#########################
    activity = pd.read_table(path+user_activity_path, names=columns_activity).sort_values(['user_id', 'day'],
    ascending=True)
    activity = activity[(activity['day'] >= f_begin_day) & (activity['day'] <= f_end_day)]
    #上一次连续操作的日子，以及距离现在的天数(已经添加，线下提升)
    feature = last_continue_activity_from_now(activity,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    # #自己看自己视频的次数
    # feature = get_watch_self_feature(activity,f_begin_day,f_end_day)
    # features = pd.merge(features,feature,on='user_id',how='left')
    # #操作的天数
    # feature = activity_days_nums(activity,f_begin_day,f_end_day)
    # features = pd.merge(features,feature,on='user_id',how='left')
    #操作第一天的日期，以及距离现在的时间
    feature = actions_first_date(activity,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    #操作类型012345次数、平均次数
    for i in range(6):
        feature = actions_type_i_numbers(activity,f_begin_day,f_end_day,i)
        features = pd.merge(features,feature,on='user_id',how='left')
    #操作界面01234次数、平均次数
    feature = actions_page_0_numbers(activity,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    feature = actions_page_1_numbers(activity,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    feature = actions_page_2_numbers(activity,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    feature = actions_page_3_numbers(activity,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    feature = actions_page_4_numbers(activity,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    # 划窗获取create_video操作总数以及平均值
    feature = actions_window_numbers(activity,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    #划窗获取create_video操作总数以及平均值
    feature = activity_average_internal(activity,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    #获取操作作者次数
    feature = activity_authors_nums(activity,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    #划窗获取activity_authors_window_numbers操作总数以及平均值
    feature = activity_authors_window_numbers(activity,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    #获取观看的 video_id_feature特征
    feature = video_id_feature(activity,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    # 获取观看的 author_id_feature特征
    feature = author_id_feature(activity,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    

    del activity
    gc.collect()
    # # ###############################处理launch_app特征#########################
    
    launch_app = pd.read_table(path+app_launch_path, names=columns_app).sort_values(['user_id', 'day'], 
    ascending=True)
    launch_app = launch_app[(launch_app['day'] >= f_begin_day) & (launch_app['day'] <= f_end_day)]
    #launch_app第一天的日期，以及距离现在的时间
    feature = launch_app_first_date(launch_app,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    # 划窗构建launch_app次数
    feature = lauch_app_window_numbers(launch_app,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    # # 上一次连续launch_app的日期，以及距离现在的天数(只有一个不为空，舍去)
    # feature = last_continue_launch_app_from_now(launch_app,f_begin_day,f_end_day)
    # features = pd.merge(features,feature,on='user_id',how='left')
    # launch_app的总天数
    feature = launch_app_days_nums(launch_app,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
    # launch_app的平均间隔
    feature = launch_app_average_internal(launch_app,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')

 


    del launch_app
    gc.collect()
    # # ###############################处理create_video特征#########################
    create_video = pd.read_table(path+video_create_path, names=columns_video).sort_values(['user_id', 'day'],
    ascending=True)
    create_video = create_video[(create_video['day'] >= f_begin_day) & (create_video['day'] <= f_end_day)]
    
    feature = create_video_first_date(create_video,f_begin_day,f_end_day)
    features = pd.merge(features,feature,on='user_id',how='left')
   
    # # 划窗构建create特征
    # feature = create_video_slide(create_video,f_begin_day,f_end_day)
    # features = pd.merge(features,feature,on='user_id',how='left')
    # create 日期特征
    # feature = create_video_date(create_video,f_begin_day,f_end_day)
    # features = pd.merge(features,feature,on='user_id',how='left')

    
    del create_video
    gc.collect()

    return features
    
gc.collect()
!ls .

# ##############第一次提取特征用
#提取第一个窗口的特征
print('一、开始提取第一个窗口（1-16）的特征....')
feature_1_16 = gen_features(1, 16, 1, 16)
print(feature_1_16.head())
feature_1_16.to_csv('xk_feature_1_16.csv', index=None)
del feature_1_16
gc.collect()
print('第一个窗口提取完毕！！')

# 提取第二个窗口的特征
print('二、开始提取第二个窗口（8-23）的特征....')
feature_8_23 = gen_features(8, 23, 8, 23)
feature_8_23.to_csv('xk_feature_8_23.csv', index=None)
del feature_8_23
gc.collect()
print('第二个窗口提取完毕！！')

#提取15-30天的特征(1-30的用户)---预测31-37天的活跃用户
print('三、开始提取15-30的特征(1-30的用户),用以预测31-37....')
feature_1_30 = gen_features(1, 30, 15, 30)
feature_1_30.to_csv('xk_feature_1_30.csv', index=None)
del feature_1_30
gc.collect()
print('提取完毕！！')
######################加特征用
# 提取第一个窗口的特征
# columns_app = ['user_id', 'day']
# columns_activity = ['user_id', 'day', 'page', 'video_id', 'author_id', 'action_type']
# columns_register = ['user_id', 'day', 'register_type', 'device_type']
# columns_video = ['user_id', 'day']

gc.collect()
print('一、开始添加第一个窗口（1-16）的特征....')
feature_1_16 = add_feature(1, 16, 1, 16)
print(feature_1_16.shape)
print(feature_1_16.columns)
feature_1_16.to_csv('xk_feature_1_16.csv', index=None)
del feature_1_16
gc.collect()
print('第一个窗口添加完毕！！')

# 提取第二个窗口的特征
print('二、开始添加第二个窗口（8-23）的特征....')
feature_8_23 = add_feature(8, 23, 8, 23)
print(feature_8_23.shape)
print(feature_8_23.columns)
feature_8_23.to_csv('xk_feature_8_23.csv', index=None)
del feature_8_23
gc.collect()
print('第二个窗口添加完毕！！')

#提取15-30天的特征(1-30的用户)---预测31-37天的活跃用户
print('三、开始添加15-30的特征(1-30的用户),用以预测31-37....')
feature_1_30 = add_feature(1, 30, 15, 30)
print(feature_1_30.shape)
print(feature_1_30.columns)
feature_1_30.to_csv('xk_feature_1_30.csv', index=None)
del feature_1_30
gc.collect()
print('添加完毕！！')






# 将两个窗口(1-16和8-23)的特征融合，用以训练模型
gc.collect()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
lec = LabelEncoder()
enc = OneHotEncoder()
print('开始提取训练数据...')
feature_1_16 = pd.read_csv('xk_feature_1_16.csv')
feature_8_23 = pd.read_csv('xk_feature_8_23.csv')

train_data = pd.concat([feature_1_16, feature_8_23], ignore_index=True)
train_data = train_data.drop(labels=['activity_authors_pre_1_numbers',
        'activity_authors_pre_3_numbers',
       'activity_authors_pre_5_numbers',
        'activity_authors_pre_7_numbers'],axis=1)
# sss = pd.DataFrame(enc.fit_transform(train_data['register_type'].reshape(-1,1)).toarray())
# print(sss)
# train_data = pd.concat([train_data,sss],axis=1)
# train_data.pop('register_type')
train_data.pop('activity_average_internal')

#将少于1000的device_type改为0000
# a = feature['device_type'].value_counts().reset_index()
# a.columns = ['device_type','device_type_counts']
# b = list(a[a['device_type_counts']>1000].device_type)
# a['device_type_gai'] = a['device_type'].apply(lambda x:x if x in b else 0000)
# a.pop('device_type_counts')
# train_data = pd.merge(train_data,a,on='device_type',how='left')
# train_data.pop('device_type')
# print(train_data[train_data['device_type_gai']==None].shape)
# train_data['device_type_gai'] = train_data['device_type_gai'].fillna(0).apply(lambda x:int(x))
# print(train_data['device_type_gai'])

train_data['action_rate1'] = (train_data['actions_pre_1_numbers']/
    train_data['actions_numbers']).map(lambda x:round(x, 2))
train_data['action_rate3'] = (train_data['actions_pre_3_numbers']/
    train_data['actions_numbers']).map(lambda x:round(x, 2))
train_data['action_rate5'] = (train_data['actions_pre_5_numbers']/
    train_data['actions_numbers']).map(lambda x:round(x, 2))
train_data['action_rate7'] = (train_data['actions_pre_7_numbers']/
    train_data['actions_numbers']).map(lambda x:round(x, 2))

train_data['action_rate13'] = (train_data['actions_pre_1_numbers']/
    train_data['actions_pre_3_numbers']).map(lambda x:round(x, 2))
train_data['action_rate15'] = (train_data['actions_pre_1_numbers']/
    train_data['actions_pre_5_numbers']).map(lambda x:round(x, 2))
train_data['action_rate17'] = (train_data['actions_pre_1_numbers']/
    train_data['actions_pre_7_numbers']).map(lambda x:round(x, 2))
    
train_data['action_rate35'] = (train_data['actions_pre_3_numbers']/
    train_data['actions_pre_5_numbers']).map(lambda x:round(x, 2))
train_data['action_rate37'] = (train_data['actions_pre_3_numbers']/
    train_data['actions_pre_7_numbers']).map(lambda x:round(x, 2))
train_data['action_rate57'] = (train_data['actions_pre_5_numbers']/
    train_data['actions_pre_7_numbers']).map(lambda x:round(x, 2))
    
train_data['action_type_rate01'] = (train_data['actions_action_type_0_numbers']/
    train_data['actions_action_type_1_numbers']).map(lambda x:round(x, 2))
train_data['action_type_rate02'] = (train_data['actions_action_type_0_numbers']/
    train_data['actions_action_type_2_numbers']).map(lambda x:round(x, 2))
train_data['action_type_rate03'] = (train_data['actions_action_type_0_numbers']/
    train_data['actions_action_type_3_numbers']).map(lambda x:round(x, 2))
train_data['action_type_rate04'] = (train_data['actions_action_type_4_numbers']/
    train_data['actions_action_type_4_numbers']).map(lambda x:round(x, 2))
train_data['action_type_rate05'] = (train_data['actions_action_type_0_numbers']/
    train_data['actions_action_type_5_numbers']).map(lambda x:round(x, 2))
    
train_data['action_type_rate12'] = (train_data['actions_action_type_1_numbers']/
    train_data['actions_action_type_2_numbers']).map(lambda x:round(x, 2))
train_data['action_type_rate13'] = (train_data['actions_action_type_1_numbers']/
    train_data['actions_action_type_3_numbers']).map(lambda x:round(x, 2))
train_data['action_type_rate14'] = (train_data['actions_action_type_1_numbers']/
    train_data['actions_action_type_4_numbers']).map(lambda x:round(x, 2))
train_data['action_type_rate15'] = (train_data['actions_action_type_1_numbers']/
    train_data['actions_action_type_5_numbers']).map(lambda x:round(x, 2))

# train_data['actions_action_page_0_numbers_per_author'] = (train_data['actions_page_0_numbers']/
#     train_data['activity_authors_nums']).map(lambda x:round(x,2))
# train_data['actions_action_page_1_numbers_per_author'] = (train_data['actions_page_1_numbers']/
#     train_data['activity_authors_nums']).map(lambda x:round(x,2))
# train_data['actions_action_page_2_numbers_per_author'] = (train_data['actions_page_2_numbers']/
#     train_data['activity_authors_nums']).map(lambda x:round(x,2))
# train_data['actions_action_page_3_numbers_per_author'] = (train_data['actions_page_3_numbers']/
#     train_data['activity_authors_nums']).map(lambda x:round(x,2))
# train_data['actions_action_page_4_numbers_per_author'] = (train_data['actions_page_4_numbers']/
#     train_data['activity_authors_nums']).map(lambda x:round(x,2))

# train_data['action_page_rate12'] = (train_data['actions_page_1_numbers']/
#     train_data['actions_page_2_numbers']).map(lambda x:round(x, 2))
# train_data['action_page_rate13'] = (train_data['actions_page_1_numbers']/
#     train_data['actions_page_3_numbers']).map(lambda x:round(x, 2))
# train_data['action_page_rate14'] = (train_data['actions_page_1_numbers']/
#     train_data['actions_page_4_numbers']).map(lambda x:round(x, 2))
    
# train_data['action_page_rate23'] = (train_data['actions_page_2_numbers']/
#     train_data['actions_page_3_numbers']).map(lambda x:round(x, 2))
# train_data['action_page_rate24'] = (train_data['actions_page_2_numbers']/
#     train_data['actions_page_4_numbers']).map(lambda x:round(x, 2))
    
# train_data['action_page_rate34'] = (train_data['actions_page_3_numbers']/
#     train_data['actions_page_4_numbers']).map(lambda x:round(x, 2))



# train_data['last_continue_launch_app_day222'] = train_data['last_continue_launch_app_day']
# last_continue_launch_app_day',
#       'last_continue_launch_app_from_now
# train_data.pop('last_continue_launch_app_from_now')
train_data.pop('actions_action_type_0_numbers_mean')
train_data.pop('actions_action_type_1_numbers_mean')
train_data.pop('actions_action_type_2_numbers_mean')
train_data.pop('actions_action_type_3_numbers_mean')
train_data.pop('actions_action_type_4_numbers_mean')
train_data.pop('actions_action_type_5_numbers_mean')

# 'video_id_max', 'video_id_min', 'video_id_mean',
#       'video_id_std', 'video_id_kurt', 'video_id_skew', 'video_id_last'
# train_data.pop('video_id_last')
# train_data.pop('video_id_std')
# train_data.pop('video_id_kurt')
# train_data.pop('video_id_skew')
# train_data.pop('video_id_max')
# train_data.pop('video_id_min')
# train_data.pop('video_id_mean')


# train_data.pop('author_id_last')
# train_data.pop('author_id_std')
# train_data.pop('author_id_kurt')
# train_data.pop('author_id_skew')
# train_data.pop('author_id_max')
# train_data.pop('author_id_min')
# train_data.pop('author_id_mean')
# train_data.pop('actions_page_2_numbers_mean')
# train_data.pop('actions_page_3_numbers_mean')
# train_data.pop('actions_page_4_numbers_mean')

# print(train_data['activity_average_internal'])
del feature_1_16
del feature_8_23
gc.collect()
print('训练数据提取完毕！！')
print(train_data.columns)
print(train_data.shape)
y_train = train_data.pop('label')
x_train = train_data.drop(columns=["user_id"])
del train_data
gc.collect()
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=2018)
print('开始训练模型...')

params = {'booster': 'gbtree',
          'objective': 'binary:logistic',
          'gamma': 0.2,
          'max_depth': 5,  # 构建树的深度，越大越容易过拟合
          'eta': 0.01,  # 如同学习率
          'seed': 2018,  
          'subsample': 0.9,
          'colsample_bytree': 0.9,
          'silent': 1,
          'eval_metric ': 'auc'
          }
plst = list(params.items())
train_data = xgb.DMatrix(x_train, label=y_train)
num_rounds = 800
model = xgb.train(plst, train_data, num_rounds)
model.save_model('xgb_xk.model')
print('训练完成！！！！')
x_test = xgb.DMatrix(x_test)
offline_result = model.predict(x_test)
print('auc:', roc_auc_score(y_test, offline_result))




from sklearn.metrics import roc_auc_score

# model = xgb.Booster(model_file='xgb_xk.model')
# x_test = xgb.DMatrix(x_test)
offline_result = model.predict(x_test)
print('auc:', roc_auc_score(y_test, offline_result))

# 使用以下命令可以进行tar.bz2格式文件的压缩和解压缩，效果拔群。
# !tar -jcvf <要生成的压缩文件名> <要压缩的文件> # 压缩
# !tar -jxvf <要解压缩的文件> # 解压缩



model = xgb.Booster(model_file='xgb_xk.model')
# 开始预测
feature_pre = pd.read_csv('xk_feature_1_30.csv')
# feature_pre.pop('activity_days_nums')
feature_pre['action_rate1'] = (feature_pre['actions_pre_1_numbers']/
    feature_pre['actions_numbers']).map(lambda x:round(x, 2))
feature_pre['action_rate3'] = (feature_pre['actions_pre_3_numbers']/
    feature_pre['actions_numbers']).map(lambda x:round(x, 2))
feature_pre['action_rate5'] = (feature_pre['actions_pre_5_numbers']/
    feature_pre['actions_numbers']).map(lambda x:round(x, 2))
feature_pre['action_rate7'] = (feature_pre['actions_pre_7_numbers']/
    feature_pre['actions_numbers']).map(lambda x:round(x, 2))

feature_pre['action_rate13'] = (feature_pre['actions_pre_1_numbers']/
    feature_pre['actions_pre_3_numbers']).map(lambda x:round(x, 2))
feature_pre['action_rate15'] = (feature_pre['actions_pre_1_numbers']/
    feature_pre['actions_pre_5_numbers']).map(lambda x:round(x, 2))
feature_pre['action_rate17'] = (feature_pre['actions_pre_1_numbers']/
    feature_pre['actions_pre_7_numbers']).map(lambda x:round(x, 2))
    
feature_pre['action_rate35'] = (feature_pre['actions_pre_3_numbers']/
    feature_pre['actions_pre_5_numbers']).map(lambda x:round(x, 2))
feature_pre['action_rate37'] = (feature_pre['actions_pre_3_numbers']/
    feature_pre['actions_pre_7_numbers']).map(lambda x:round(x, 2))
feature_pre['action_rate57'] = (feature_pre['actions_pre_5_numbers']/
    feature_pre['actions_pre_7_numbers']).map(lambda x:round(x, 2))
    
feature_pre['action_type_rate01'] = (feature_pre['actions_action_type_0_numbers']/
    feature_pre['actions_action_type_1_numbers']).map(lambda x:round(x, 2))
feature_pre['action_type_rate02'] = (feature_pre['actions_action_type_0_numbers']/
    feature_pre['actions_action_type_2_numbers']).map(lambda x:round(x, 2))
feature_pre['action_type_rate03'] = (feature_pre['actions_action_type_0_numbers']/
    feature_pre['actions_action_type_3_numbers']).map(lambda x:round(x, 2))
feature_pre['action_type_rate04'] = (feature_pre['actions_action_type_4_numbers']/
    feature_pre['actions_action_type_4_numbers']).map(lambda x:round(x, 2))
feature_pre['action_type_rate05'] = (feature_pre['actions_action_type_0_numbers']/
    feature_pre['actions_action_type_5_numbers']).map(lambda x:round(x, 2))
    
feature_pre['action_type_rate12'] = (feature_pre['actions_action_type_1_numbers']/
    feature_pre['actions_action_type_2_numbers']).map(lambda x:round(x, 2))
feature_pre['action_type_rate13'] = (feature_pre['actions_action_type_1_numbers']/
    feature_pre['actions_action_type_3_numbers']).map(lambda x:round(x, 2))
feature_pre['action_type_rate14'] = (feature_pre['actions_action_type_1_numbers']/
    feature_pre['actions_action_type_4_numbers']).map(lambda x:round(x, 2))
feature_pre['action_type_rate15'] = (feature_pre['actions_action_type_1_numbers']/
    feature_pre['actions_action_type_5_numbers']).map(lambda x:round(x, 2))

feature_pre.pop('activity_average_internal')
feature_pre.pop('actions_action_type_0_numbers_mean')
feature_pre.pop('actions_action_type_1_numbers_mean')
feature_pre.pop('actions_action_type_2_numbers_mean')
feature_pre.pop('actions_action_type_3_numbers_mean')
feature_pre.pop('actions_action_type_4_numbers_mean')
feature_pre.pop('actions_action_type_5_numbers_mean')
# feature_pre.pop('video_id_last')
# feature_pre.pop('video_id_std')
# feature_pre.pop('video_id_kurt')
# feature_pre.pop('video_id_skew')
# feature_pre.pop('video_id_max')
# feature_pre.pop('video_id_min')
# feature_pre.pop('video_id_mean')


# feature_pre.pop('author_id_last')
# feature_pre.pop('author_id_std')
# feature_pre.pop('author_id_kurt')
# feature_pre.pop('author_id_skew')
# feature_pre.pop('author_id_max')
# feature_pre.pop('author_id_min')
# feature_pre.pop('author_id_mean')

# 'video_id_max', 'video_id_min', 'video_id_mean',
#       'video_id_std', 'video_id_kurt', 'video_id_skew', 'video_id_last'

# feature_pre.pop('watch_self_nums')
test_data = xgb.DMatrix(feature_pre.drop(columns=["user_id"]))

mysubmission = pd.DataFrame()
feature_pre['probability'] = model.predict(test_data)

feature_pre[["user_id", "probability"]].to_csv("submission_xk.txt", index=None, header=None)






!./kesci_submit -token 920634de98616435 -file submission_xk_xgb_lgb.txt
