#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  14 22:17:45 2020

@author: qhu
"""

import numpy as np
import pandas as pd
import math
import argparse

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import tensorflow as tf
from datetime import datetime
from tensorflow import keras
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default='')
    parser.add_argument('--train_file', default='')
    parser.add_argument('--val_file', default='')
    parser.add_argument('--test_file', default='')
    parser.add_argument('--pretrained_weights', default='pretrained_model.h5')
    parser.add_argument('--save_file', default='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--augment', default=True)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr_init', type=float, default=1e-4)
    parser.add_argument('--lr_patience', type=int, default=5)
    parser.add_argument('--lr_cooldown', type=int, default=0)
    parser.add_argument('--lr_reduce', type=float, default=0.5)
    parser.add_argument('--estop_patience', type=int, default=25)
    return parser.parse_args()


def create_model(input_shape):
    '''
    Define network
    '''    
    inputs = Input(shape=input_shape)
    chexNet = DenseNet121(include_top=False, input_tensor=inputs, weights="imagenet")
    x = GlobalAveragePooling2D()(chexNet.output)
    predictions = Dense(1, activation="sigmoid", name="out", kernel_initializer='he_normal')(x)
    model = Model(inputs=inputs, outputs=predictions)

    return model

def roc_auc(y_true, y_pred):
    value, update_op = tf.metrics.auc(y_true, y_pred)

    metric_vars = [i for i in tf.local_variables() if 'roc_auc' in i.name.split('/')[1]]

    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def main(args):
    # params
    IMG_SIZE = args.image_size
    IMG_DIM = [IMG_SIZE, IMG_SIZE, 3]
    BATCH_SIZE = args.batch_size
    IMG_DIR = args.image_dir
    LR_INIT = args.lr_init
    LR_REDUCE = args.lr_reduce
    LR_PATIENCE = args.lr_patience
    LR_COOLDOWN = args.lr_cooldown
    EPOCH = args.epoch
    ES_PATIENCE = args.estop_patience
    SAVE_FILE = args.save_file.format(LR_INIT, LR_REDUCE * 10, LR_PATIENCE, 
                                      LR_COOLDOWN, ES_PATIENCE, BATCH_SIZE)
    
    # read in previously saved train/val/test dataframes
    train = pd.read_csv(args.train_file)
    val = pd.read_csv(args.val_file)
    test = pd.read_csv(args.test_file)
    
    # define image generators
    if args.augment:
        train_gen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                       horizontal_flip=True,
                                       vertical_flip=False,
                                       rotation_range=5,
                                       width_shift_range=0,
                                       height_shift_range=0)
        
    else:
        train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    # get images using generators
    train_generator = train_gen.flow_from_dataframe(train, directory=IMG_DIR,
                                                    x_col="", y_col="",
                                                    target_size=(IMG_DIM[0],IMG_DIM[1]), class_mode="binary",
                                                    batch_size=BATCH_SIZE, seed=8)
    val_generator = test_gen.flow_from_dataframe(val, directory=IMG_DIR,
                                                 x_col="", y_col="",
                                                 target_size=(IMG_DIM[0],IMG_DIM[1]), class_mode="binary",
                                                 batch_size=BATCH_SIZE, seed=8)
    test_generator = test_gen.flow_from_dataframe(test, directory=IMG_DIR,
                                                  x_col="", y_col="",
                                                  target_size=(IMG_DIM[0],IMG_DIM[1]), class_mode="binary",
                                                  batch_size=1, shuffle=False, seed=8)
    
    
    # create the pre-trained model and load weights trained on the RSNA pneumonia dataset
    model = create_model(IMG_DIM)
    model.load_weights(args.pretrained_weights)
    
    # compute class weight
    class_weight = dict(enumerate(compute_class_weight('balanced', 
                                                       np.unique(np.asarray(train_generator.classes)), 
                                                       train_generator.classes)))
    
    # Watch validation loss and stop once it starts decreasing
    early_stopping = EarlyStopping(monitor='val_loss', patience=ES_PATIENCE, 
                                   verbose=1, restore_best_weights=True)
    scheduler = ReduceLROnPlateau(monitor='val_loss', patience=LR_PATIENCE, 
                                  cooldown=LR_COOLDOWN, verbose=1, mode='min', 
                                  factor=LR_REDUCE, min_lr=1e-7)

    #tensorboard log
    logdir = "./logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    
    # We use a lower learning rate
    model.compile(optimizer=Adam(lr=LR_INIT, decay=1e-4, clipnorm=1.0), #clipvalue=0.5
                  loss='binary_crossentropy',
                  weighted_metrics=['accuracy'],
                  metrics=[tf.keras.metrics.AUC()])
    
      
    # We fit our model again for fine-tuning
    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.samples // BATCH_SIZE,
                        epochs=EPOCH,
                        validation_data=val_generator,
                        callbacks=[early_stopping, scheduler, tensorboard_callback],
                        validation_steps=val_generator.samples // BATCH_SIZE,
                        class_weight=class_weight,
                        workers=8)
    model.save_weights('./weights/' + SAVE_FILE + '.h5')
    
    # evaluate
    pred = model.predict_generator(test_generator, test_generator.samples)
    # print(pred.shape)
    test['predictions']=pred
    y_test = test_generator.classes
    test.to_excel('./results/' + SAVE_FILE + '.xlsx', index=False)
    print('accuracy: ', accuracy_score(y_test, np.round(pred).astype('uint8')))
    print('auc: ', roc_auc_score(y_test, pred))
    print('F1 score: ', f1_score(y_test, np.round(pred).astype('uint8')))
    print("precision: ", precision_score(y_test, np.round(pred).astype('uint8')))
    print("recall: ", recall_score(y_test, np.round(pred).astype('uint8')))
    
    array = confusion_matrix(y_test, np.round(pred).astype('uint8'))
    cm = pd.DataFrame(array, index=range(2), columns=range(2))
    print(cm)
    


if __name__ == '__main__':
    args = parse_args()
    main(args)
