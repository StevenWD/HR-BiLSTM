# coding: utf-8

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

def ranking_loss(y_true, y_pred):
    return K.maximum(0.0, 0.1 + K.sum(y_pred*y_true,axis=-1))

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

K.set_session(sess)

# CONFIG
config = ConfigParser()
config.read('./config.ini')

# INPUT
question_input = Input(shape=(config.getint('pre', 'question_maximum_length'), ), dtype='int32',name="question_input")
relation_all_input = Input(shape=(config.getint('pre', 'relation_word_maximum_length'), ), dtype='int32',name="relation_all_input")
relation_input = Input(shape=(config.getint('pre', 'relation_maximum_length'), ), dtype='int32',name="relation_input")
relation_all_input_neg = Input(shape=(config.getint('pre', 'relation_word_maximum_length'), ), dtype='int32',name="relation_all_input_neg")
relation_input_neg = Input(shape=(config.getint('pre', 'relation_maximum_length'), ), dtype='int32',name="relation_input_neg")

# EMBEDDING
question_emd = np.load('./question_emd_matrix.npy')
relation_emd = np.load('./relation_emd_matrix.npy')
relation_all_emd = np.load('./relation_all_emd_matrix.npy')

question_emd = Embedding(question_emd.shape[0],
        config.getint('pre', 'word_emd_length'),
        weights=[question_emd],
        input_length=config.getint('pre', 'question_maximum_length'),
        trainable=False,name="question_emd")(question_input)

sharedEmbd_r_w = Embedding(relation_all_emd.shape[0],
        config.getint('pre', 'word_emd_length'),
        weights=[relation_all_emd],
        input_length=config.getint('pre', 'relation_word_maximum_length'),
        trainable=True,name="sharedEmbd_r_w")

relation_word_emd = sharedEmbd_r_w(relation_all_input)

sharedEmbd_r = Embedding(relation_emd.shape[0],
        config.getint('pre', 'word_emd_length'),
        weights=[relation_emd],
        input_length=config.getint('pre', 'relation_maximum_length'),
        trainable=True,name="sharedEmbd_r")

relation_emd = sharedEmbd_r(relation_input)

relation_word_emd_neg = sharedEmbd_r_w(relation_all_input_neg)

relation_emd_neg = sharedEmbd_r(relation_input_neg)

# Bi-LSTM
bilstem_layer = Bidirectional(LSTM(units=40, return_sequences=True, implementation=2),name="bilstem_layer")
question_bilstm_1 = bilstem_layer(question_emd)
question_bilstm_2 = Bidirectional(LSTM(units=40, return_sequences=True, implementation=2),name="question_bilstm_2")(question_bilstm_1)

relation_word_bilstm = bilstem_layer(relation_word_emd)
relation_bilstm = bilstem_layer(relation_emd)

relation_word_bilstm_neg = bilstem_layer(relation_word_emd_neg)
relation_bilstm_neg = bilstem_layer(relation_emd_neg)

# Max-Pooling
question_res = Add()([question_bilstm_1, question_bilstm_2])
question_maxpool = MaxPooling1D(80, padding='same')(question_res)
question_flatten = Flatten()(question_maxpool)

# relation_con = Concatenate(axis=-2)([relation_word_bilstm, relation_bilstm])
relation_maxpool = MaxPooling1D(80, padding='same')(relation_bilstm)
relation_word_maxpool = MaxPooling1D(80, padding='same')(relation_word_bilstm)
relation_res = Add()([relation_maxpool, relation_word_maxpool])
relation_flatten = Flatten()(relation_res)

relation_maxpool_neg = MaxPooling1D(80, padding='same')(relation_bilstm_neg)
relation_word_maxpool_neg = MaxPooling1D(80, padding='same')(relation_word_bilstm_neg)
relation_res_neg = Add()([relation_maxpool_neg, relation_word_maxpool_neg])
relation_flatten_neg = Flatten()(relation_res_neg)

# COSINE SIMILARITY
result = Dot(axes=-1, normalize=True)([question_flatten, relation_flatten])
result_neg = Dot(axes=-1, normalize=True)([question_flatten, relation_flatten_neg])

out = Concatenate(axis=-1)([result, result_neg])

model = Model(inputs=[question_input, relation_input, relation_all_input,relation_input_neg, relation_all_input_neg ], outputs=out)
model.compile(optimizer=Adam(), loss=ranking_loss)

print(model.summary())
# quit()
train_question_features = np.load('./train_question_feature.npy')
train_relation_features = np.load('./train_relation_feature.npy')
train_relation_all_features = np.load('./train_relation_all_feature.npy')
train_relation_features_neg = np.load('./train_relation_feature_neg.npy')
train_relation_all_features_neg = np.load('./train_relation_all_feature_neg.npy')
train_labels = np.load('./train_label.npy')

# test_question_features = np.load('./test_question_feature.npy')
# test_relation_features = np.load('./test_relation_feature.npy')
# test_relation_all_features = np.load('./test_relation_all_feature.npy')

# class_weight = compute_class_weight('balanced', np.unique(train_labels), train_labels)

model.fit([train_question_features, train_relation_features, train_relation_all_features, train_relation_features_neg, train_relation_all_features_neg], train_labels, epochs=10, batch_size=2048, shuffle=True)
model.save('model.h5')
model.save_weights('my_model_weights.h5')
# model.load_weights('my_model_weights.h5')

# test_predict = model.predict([test_question_features, test_relation_features, test_relation_all_features], batch_size=512)
# np.save('test_predict.npy', test_predict)
