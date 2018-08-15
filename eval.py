# coding: utf-8

import json
import numpy as np
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Add, MaxPooling1D, Concatenate, Dot, Flatten
from keras.models import Model
from keras.preprocessing import sequence
from keras import backend as K
import os
import tensorflow as tf
from configparser import ConfigParser
from keras.optimizers import Adam
from keras import losses
from sklearn.utils.class_weight import compute_class_weight
from preprocess import readData
from preprocess import readRelation

def ranking_loss(y_true, y_pred):
     return K.maximum(0.0, 0.1 + K.sum(y_pred*y_true,axis=-1))

def model_construct():
    # CONFIG
    config = ConfigParser()
    config.read('./config.ini')

    question_input = Input(shape=(config.getint('pre', 'question_maximum_length'), ), dtype='int32',name="question_input")
    relation_all_input = Input(shape=(config.getint('pre', 'relation_word_maximum_length'), ), dtype='int32',name="relation_all_input")
    relation_input = Input(shape=(config.getint('pre', 'relation_maximum_length'), ), dtype='int32',name="relation_input")

    question_emd = np.load('/home/stevenwd/HR-BiLSTM/question_emd_matrix.npy')
    relation_emd = np.load('/home/stevenwd/HR-BiLSTM/relation_emd_matrix.npy')
    relation_all_emd = np.load('/home/stevenwd/HR-BiLSTM/relation_all_emd_matrix.npy')

    question_emd = Embedding(question_emd.shape[0],
            config.getint('pre', 'word_emd_length'),
            weights=[question_emd],
            input_length=config.getint('pre', 'question_maximum_length'),
            trainable=False,name="question_emd")(question_input)

    sharedEmbd_r_w = Embedding(relation_all_emd.shape[0],
            config.getint('pre', 'word_emd_length'),
            weights=[relation_all_emd],
            input_length=config.getint('pre', 'relation_word_maximum_length'),
            trainable=False,name="sharedEmbd_r_w")
    relation_word_emd = sharedEmbd_r_w(relation_all_input)
    sharedEmbd_r = Embedding(relation_emd.shape[0],
        config.getint('pre', 'word_emd_length'),
        weights=[relation_emd],
        input_length=config.getint('pre', 'relation_maximum_length'),
        trainable=True,name="sharedEmbd_r")
    relation_emd = sharedEmbd_r(relation_input)
    bilstem_layer = Bidirectional(LSTM(units=40, return_sequences=True, implementation=2),name="bilstem_layer")
    question_bilstm_1 = bilstem_layer(question_emd)
    question_bilstm_2 = Bidirectional(LSTM(units=40, return_sequences=True, implementation=2),name="question_bilstm_2")(question_bilstm_1)
    relation_word_bilstm = bilstem_layer(relation_word_emd)
    relation_bilstm = bilstem_layer(relation_emd)
    question_res = Add()([question_bilstm_1, question_bilstm_2])
    question_maxpool = MaxPooling1D(80, padding='same')(question_res)
    question_flatten = Flatten()(question_maxpool)
    relation_maxpool = MaxPooling1D(80, padding='same')(relation_bilstm)
    relation_word_maxpool = MaxPooling1D(80, padding='same')(relation_word_bilstm)
    relation_res = Add()([relation_maxpool, relation_word_maxpool])
    relation_flatten = Flatten()(relation_res)
    result = Dot(axes=-1, normalize=True)([question_flatten, relation_flatten])
    model = Model(inputs=[question_input, relation_input, relation_all_input,], outputs=result)
    model.compile(optimizer=Adam(), loss=ranking_loss)
    return model

if __name__ == '__main__':
    # GPU settings
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    neg_num = json.load(open('./neg_number.json', 'r'))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    K.set_session(sess)
    model = model_construct()
    model.load_weights('./my_model_weights_06_2048.h5')
    print(model.summary())

    question_feature = np.load('./test_question_feature.npy')

    relation_feature = np.load('./test_relation_feature.npy')
    relation_all_feature = np.load('./test_relation_all_feature.npy')

    print('positive data loaded...')
    simi_pos = model.predict([question_feature, relation_feature, relation_all_feature], batch_size=1024)

    print('positive similarity computed...')
    np.save('test_pre_pos.npy', simi_pos)

    relation_feature_neg = np.load('./test_relation_feature_neg.npy')
    relation_all_feature_neg = np.load('./test_relation_all_feature_neg.npy')

    print('negtive data loaded...')
    simi_neg = model.predict([question_feature, relation_feature_neg, relation_all_feature_neg], batch_size=1024)

    print('negtive similarity computed...')
    np.save('test_pre_neg.npy', simi_neg)

    acc = np.sum(simi_pos>simi_neg) / simi_pos.shape[0]
    print("relation accurcy: " + str(acc))

    index = 0
    false_list = list()
    true_list = list()
    all_set = set()

    config = ConfigParser()
    config.read('./config.ini')
    data = readData(config.get('pre', 'test_filepath'))
    relation = readRelation(config.get('pre', 'relation_filepath'))
    for num,neg_index in neg_num:
        if np.sum(simi_pos[index: index+num]-simi_neg[index: index+num]<0) > 0:
            false_list.append(neg_index)
            print (simi_pos[index])
            print (np.max(simi_neg[index: index+num]))
            print (len(simi_neg[index: index+num]))
            print (np.argmax(simi_neg[index: index+num]))
            print (simi_neg[index: index+num][np.argmax(simi_neg[index: index+num])])
            print (neg_index)
            print ("")
            pass
        else:
            true_list.append(neg_index)
        index += num
        all_set.add(neg_index)
    print (max(true_list))
    true_list = set([i for i in true_list if i in all_set and i not in false_list])
    print (len(all_set))
    print (len(true_list))
    for i in all_set:
        if i not in true_list:
            print (i)
    print (data[0])
    print (relation[0][1])
    print (relation[1][1])
    print('sentence accurcy: '+str(len(true_list)/len(all_set)))
