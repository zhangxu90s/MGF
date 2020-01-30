# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   
sess = tf.Session(config=config)

KTF.set_session(sess)
import keras
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard,EarlyStopping, ReduceLROnPlateau
from keras.layers import Embedding, Input, Bidirectional, Lambda,LSTM,SimpleRNN,Dense,Activation,subtract,add,multiply,concatenate,dot,Dropout,BatchNormalization
from keras.models import Model,Sequential
from keras.optimizers import Adam,Adadelta,RMSprop
from keras.preprocessing.sequence import pad_sequences
import data_helper
from attention import AttentionWithContext, AttentionLayer

dim = 300
input_dim = data_helper.MAX_SEQUENCE_LENGTH
emb_dim = data_helper.EMB_DIM
model_path = './model/siameselstm.hdf5'
tensorboard_path = './model/ensembling'

embedding_matrix = data_helper.load_pickle('embedding_matrix.pkl')

embedding_layer = Embedding(embedding_matrix.shape[0],
                            emb_dim,
                            weights=[embedding_matrix],
                            input_length=input_dim,
                            trainable=False)

def base_network1(input_shape):
    input = Input(shape=input_shape)

    p = embedding_layer(input)
    p = LSTM(dim, return_sequences=True, dropout=0.5,name='f_input')(p)
    p = AttentionWithContext()(p)

    q = embedding_layer(input)     
    q = LSTM(dim, return_sequences=True, dropout=0.5,name='a_input')(q)
    q = LSTM(dim, return_sequences=False, name='b_input')(q)
    multi_memory_lstm = add([p,q])
    return Model(input, multi_memory_lstm, name='DFF')

def base_network2(input_shape):
    input = Input(shape=input_shape)
    p = embedding_layer(input)
    p = LSTM(dim, return_sequences=True, dropout=0.5,name='f_input')(p)
    p = LSTM(dim, return_sequences=True, name='t_input')(p)
    multi_memory = AttentionWithContext()(p)
    
    return Model(input, multi_memory, name='review_base_nn')


def f1_score(y_true, y_pred):

    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    if c3 == 0:
        return 0

    precision = c1 / c2

    recall = c1 / c3

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def precision(y_true, y_pred):

    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    if c3 == 0:
        return 0

    precision = c1 / c2

    return precision


def recall(y_true, y_pred):

    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    if c3 == 0:
        return 0

    recall = c1 / c3

    return recall

def loss(y_true, y_pred):
    return - (K.mean(K.square(y_true - y_pred)) * y_true * K.log(y_pred + 1e-8) + (1-K.mean(K.square(y_true - y_pred))) * (1 - y_true) * K.log(1 - y_pred + 1e-8))

def siamese_model():
    input_shape = (input_dim,)
    
    base_net1 = base_network1(input_shape)
    base_net2 = base_network2(input_shape)
    
    input_q1 = Input(shape=input_shape, dtype='int32', name='sequence1')
    processed_q1 = base_net2([input_q1])
    
    input_q2 = Input(shape=input_shape, dtype='int32', name='sequence2')
    processed_q2 = base_net2([input_q2])
    
    input_q3 = Input(shape=input_shape, dtype='int32', name='sequence12')
    processed_q3 = base_net1([input_q3])

    input_q4 = Input(shape=input_shape, dtype='int32', name='sequence22')
    processed_q4 = base_net1([input_q4])
            
    q1 = add([processed_q1,processed_q3])

    q2 = add([processed_q2,processed_q4])

    abs_diff = Lambda(lambda x: K.abs(x[0] - x[1]))([q1,q2])
    multi_diff = multiply([q1,q2])    
    doot = dot([q1,q2],axes=1, normalize=True)
    all_diff = concatenate([multi_diff,abs_diff])
     
    p1 = Dense(300)(q1)   
    p1 = BatchNormalization()(p1)
    p1 = Activation('relu')(p1)

    p2 = Dense(300)(q2)    
    p2 = BatchNormalization()(p2)
    p2 = Activation('relu')(p2)
                
    abs_p = Lambda(lambda x: K.abs(x[0] - x[1]))([p1,p2])
    multi_p = multiply([p1,p2])
    dootp = dot([p1,p2],axes=1, normalize=True)
    all_p = concatenate([multi_p,abs_p])

    
    similarity = Dropout(0.5)(all_diff)
    similarity = Dense(300)(similarity)    
    similarity = BatchNormalization()(similarity)
    similarity = Activation('relu')(similarity)
    similarity = Dense(600)(similarity)
    similarity = add([all_p,similarity])
    similarity = Dropout(0.5)(similarity)
    similarity = Activation('relu')(similarity)
    similarity = Dense(1)(similarity)
    similarity = add([doot,similarity])
    similarity = BatchNormalization()(similarity)
    similarity = Activation('sigmoid')(similarity)
       
    model = Model([input_q1, input_q2, input_q3, input_q4], [similarity])
    #loss:binary_crossentropy
    op = RMSprop(lr=0.001)
    model.compile(loss=loss, optimizer=op, metrics=['accuracy', precision, recall, f1_score])
    return model


def train():
    
    data = data_helper.load_pickle('model_data.pkl')

    train_q1 = data['train_q1']
    train_q2 = data['train_q2']
    train_q3 = data['train_q3']
    train_q4 = data['train_q4']
    train_y = data['train_label']

    dev_q1 = data['dev_q1']
    dev_q2 = data['dev_q2']
    dev_q3 = data['dev_q3']
    dev_q4 = data['dev_q4']
    dev_y = data['dev_label']
    
    test_q1 = data['test_q1']
    test_q2 = data['test_q2']
    test_q3 = data['test_q3']
    test_q4 = data['test_q4']
    test_y = data['test_label']
    
    model = siamese_model()
    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
    tensorboard = TensorBoard(log_dir=tensorboard_path)    
    earlystopping = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=5, mode='max')
    callbackslist = [checkpoint, tensorboard,earlystopping,reduce_lr]

    model.fit([train_q1, train_q2, train_q3, train_q4], train_y,
              batch_size=512,
              epochs=200,
              validation_data=([dev_q1, dev_q2, dev_q3, dev_q4], dev_y),
              callbacks=callbackslist)

    loss, accuracy, precision, recall, f1_score = model.evaluate([test_q1, test_q2, test_q3, test_q4],test_y,verbose=1,batch_size=256)
    print("Test best model =loss: %.4f, accuracy:%.4f, precision:%.4f,recall: %.4f, f1_score:%.4f" % (loss, accuracy, precision, recall, f1_score))

if __name__ == '__main__':
    train()
    
 