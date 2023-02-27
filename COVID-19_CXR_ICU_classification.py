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
from sklearn.utils import resample


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default='')
    parser.add_argument('--train_file', default='')
    parser.add_argument('--val_file', default='')
    parser.add_argument('--test_file', default='')
    parser.add_argument('--pretrained_weights', default='pretrained_weights.h5')
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

    return model, chexNet


def roc_auc(y_true, y_pred):
    # tf = keras.backend.tf
    value, update_op = tf.metrics.auc(y_true, y_pred)

    metric_vars = [i for i in tf.local_variables() if 'roc_auc' in i.name.split('/')[1]]

    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)
    
    Returns:
      weighted_loss (function): weighted loss function
    """
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Tensor): overall scalar loss summed across all classes
        """
        loss = 0.0
        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class #complete this line
            loss += -1 * (K.mean(pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i]+epsilon)) \
                                    + K.mean(neg_weights[i] * (1-y_true[:, i]) * K.log(1-y_pred[:, i]+epsilon)))
        return loss
    return weighted_loss


def compute_class_weights(labels):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """
    # total number of patients (rows)
    N = labels.shape[0]
    pos_weights = N/(len(np.unique(labels))*np.sum(labels == 1, axis=0))
    neg_weights = N/(len(np.unique(labels))*np.sum(labels == 0, axis=0))
    return pos_weights, neg_weights


def bootstrapCI(y, pred, alpha = 0.95, n_iterations = 5000):
    n_size = int(len(pred) * 1)
    stats = list()
    for i in range(n_iterations):
        truth_resampled, predictions_resampled = resample(y, pred, n_samples=n_size)
        score = roc_auc_score(truth_resampled, predictions_resampled)
        stats.append(score)
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    return lower, upper


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
                                                    x_col='', y_col=["ICU24","ICU48","ICU72","ICU96"],
                                                    target_size=(IMG_DIM[0],IMG_DIM[1]), class_mode="raw",
                                                    batch_size=BATCH_SIZE, seed=8)
    val_generator = test_gen.flow_from_dataframe(val, directory=IMG_DIR,
                                                 x_col='', y_col=["ICU24","ICU48","ICU72","ICU96"],
                                                 target_size=(IMG_DIM[0],IMG_DIM[1]), class_mode="raw",
                                                 batch_size=BATCH_SIZE, seed=8)
    test_generator = test_gen.flow_from_dataframe(test, directory=IMG_DIR,
                                                  x_col='', y_col=["ICU24","ICU48","ICU72","ICU96"],
                                                  target_size=(IMG_DIM[0],IMG_DIM[1]), class_mode="raw",
                                                  batch_size=1, shuffle=False, seed=8)
    
    
    # create the pre-trained model and load weights trained on the RSNA pneumonia dataset
    model0, chexNet = create_model(IMG_DIM)
    model0.load_weights(args.pretrained_weights)
    
    x = model0.layers[-2].output
    predictions = Dense(4, activation="sigmoid", kernel_initializer='he_normal')(x)
    model = Model(inputs=model0.input, outputs=predictions)
    
    # compute class weight
    pos_weights, neg_weights = compute_class_weights(train_generator.labels)
    
    # Watch validation loss and stop once it starts decreasing
    early_stopping = EarlyStopping(monitor='val_loss', patience=ES_PATIENCE, 
                                   verbose=1, restore_best_weights=True)
    scheduler = ReduceLROnPlateau(monitor='val_loss', patience=LR_PATIENCE, 
                                  cooldown=LR_COOLDOWN, verbose=1, mode='min', 
                                  factor=LR_REDUCE, min_lr=1e-7)

    #tensorboard log
    logdir = "./logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    
    # train only the newly added top layer (randomly initialized), freeze all the rest
    for layer in chexNet.layers:
        layer.trainable = False
        
    # compile the model (after setting layers to non-trainable)
    model.compile(Adam(lr=LR_INIT*10),
                  loss=get_weighted_loss(pos_weights, neg_weights),
                  weighted_metrics=['accuracy'],
                  metrics=[tf.keras.metrics.AUC()])
    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.samples // BATCH_SIZE,
                        epochs=2,
                        validation_data=val_generator,
                        callbacks=[early_stopping, scheduler, tensorboard_callback],
                        validation_steps=val_generator.samples // BATCH_SIZE,
                        workers=4)
        
    # now set some/all layers to be trainable
    for layer in model.layers:
        layer.trainable = True
    
    # We use a lower learning rate
    model.compile(optimizer=Adam(lr=LR_INIT, decay=1e-4, clipnorm=1.0), #clipvalue=0.5
                  loss=get_weighted_loss(pos_weights, neg_weights),
                  weighted_metrics=['accuracy'],
                  metrics=[tf.keras.metrics.AUC()])
      
    # We fit our model again for fine-tuning
    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.samples // BATCH_SIZE,
                        epochs=EPOCH,
                        validation_data=val_generator,
                        callbacks=[early_stopping, scheduler, tensorboard_callback],
                        validation_steps=val_generator.samples // BATCH_SIZE,
                        workers=8)
    model.save_weights('./weights/' + SAVE_FILE + '.h5')

    
    # evaluate
    pred = model.predict_generator(test_generator, test_generator.samples)

    y_test = test_generator.labels
    for i in range(4):
        hr = 24*(i+1)
        test['predictions_'+str(hr)]=pred[:,i]
        print('accuracy: ', accuracy_score(y_test[:,i], np.round(pred[:,i]).astype('uint8')))
        print('auc [95%% CI]: %.4f [%.3f, %.3f]' % (roc_auc_score(y_test[:,i], pred[:,i]), *bootstrapCI(y_test[:,i], pred[:,i])))
        print('F1 score: ', f1_score(y_test[:,i], np.round(pred[:,i]).astype('uint8')))
        print("precision: ", precision_score(y_test[:,i], np.round(pred[:,i]).astype('uint8')))
        print("recall: ", recall_score(y_test[:,i], np.round(pred[:,i]).astype('uint8')))
        array = confusion_matrix(y_test[:,i], np.round(pred[:,i]).astype('uint8'))
        cm = pd.DataFrame(array, index=range(2), columns=range(2))
        print(cm)
    test.to_excel('./results/' + SAVE_FILE + '.xlsx', index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)
