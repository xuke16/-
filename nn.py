import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import gc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import time
gc.collect()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
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
train_data = train_data.fillna(0)
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

config = tf.ConfigProto( # limit to num_cpu_core CPU usage
                device_count={"CPU": 6},
                inter_op_parallelism_threads = 1,
                intra_op_parallelism_threads = 4,
                log_device_placement=False,allow_soft_placement=True,)
config_proto = tf.ConfigProto(log_device_placement=0, allow_soft_placement=0)
config_proto.gpu_options.allow_growth = True

trY = np.array(y_train).reshape(-1,1)
trX = np.array(x_train)
# print(trY)
print(trX.shape)
# print("finish loading train set ", train)

teY = np.array(y_test).reshape(-1,1)
teX = np.array(x_test)

y_train = tf.concat([1 - trY, trY], 1)
y_test = tf.concat([1 - teY, teY], 1)

learning_rate = 0.001
training_epochs = 1000
batch_size = 82279*7
display_step = 10

n_samples = trX.shape[0]  # sample_num
n_features = trX.shape[1]  # feature_num
n_class = 2

x = tf.placeholder(tf.float32, [None, n_features])
y = tf.placeholder(tf.float32, [None, n_class])

W = tf.Variable(tf.zeros([n_features, n_class]),name="weight")
b = tf.Variable(tf.zeros([n_class]),name="bias")

# predict label
pred = tf.matmul(x, W) + b

saver = tf.train.Saver(max_to_keep=1)

# accuracy
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

aucc = tf.contrib.metrics.streaming_auc(pred, y, weights=None, num_thresholds=200, metrics_collections=None, updates_collections=None, curve='ROC', name=None)
predict_op = tf.nn.sigmoid(pred)

# cross entropy
cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

# train
with tf.Session(config = config) as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            # print('epoch:',i)
            _, c = sess.run([optimizer, cost],
                            feed_dict={x: trX[i * batch_size: (i + 1) * batch_size],
                                       y: y_train[i * batch_size: (i + 1) * batch_size, :].eval()})
            avg_cost = c / total_batch
            # predictionsss = (tf.nn.softmax(sess.run(pred, feed_dict={x: trX[i * batch_size: (i + 1) * batch_size]})).eval())[:,1]
            # print('epoch',i,'/',total_batch,' train auc:', roc_auc_score(trY[i * batch_size: (i + 1) * batch_size, :],
            # predictionsss),'time',time.asctime())

        if (epoch + 1) % display_step == 0:
            predictionsss = (tf.nn.softmax(sess.run(pred, feed_dict={x: trX[i * batch_size: (i + 1) * batch_size]})).eval())[:,1]
            print('epoch',epoch,'/',training_epochs,' train auc:', roc_auc_score(trY[i * batch_size: (i + 1) * batch_size, :],
            predictionsss),"cost=", avg_cost, 'time',time.asctime())
            # predictionsss = (sess.run(pred, feed_dict={x: trX}), 1)
            predictionsss = tf.nn.softmax(sess.run(pred, feed_dict={x: teX})).eval()[:,1].astype('float32')
            print('test auc:', roc_auc_score(teY, predictionsss))
    # saver.save(sess, "./mymodel")
    print("Optimization Finished!")
    print("Testing Accuracy:", accuracy.eval({x: teX, y: y_test.eval()}))
    # predictionsss = tf.nn.softmax(sess.run(pred, feed_dict={x: trX})).eval()[:,1].astype('float32')

    # print('auc:', roc_auc_score(trY, predictionsss))

    # print(predictionsss)
    
    ###########开始预测
    feature_pre = pd.read_csv('xk_feature_1_30.csv')
    # print(feature_pre['label'])
    
    feature_pre = feature_pre.drop(labels=['activity_authors_pre_1_numbers',
        'activity_authors_pre_3_numbers',
       'activity_authors_pre_5_numbers',
        'activity_authors_pre_7_numbers'],axis=1)
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
    test_data = (feature_pre.drop(columns=["user_id",'label'])).fillna(0)
    # print(test_data.columns)
    testX = np.array(test_data)
    predictionsss = tf.nn.softmax(sess.run(pred, feed_dict={x: testX})).eval()[:,1].astype('float32')
    print(predictionsss)
    mysubmission = pd.DataFrame()
    feature_pre['probability'] = predictionsss
    feature_pre[["user_id", "probability"]].to_csv("submission_nn.txt", index=None, header=None)
    print(feature_pre[["user_id", "probability"]])
