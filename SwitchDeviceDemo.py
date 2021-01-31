# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 14:27:59 2021

@author: Aaron Chung
"""
import tensorflow as tf
from tensorflow import keras
import os

def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

  return model

def SwitchDevice(model:tf.Module,create_model_method,epoch:int,device='/GPU:0'):
  if device[0]!='/' or 'CPU GPU TPU'.find(device[1:4])==-1:
    print("ERROR:the device should be like '/GPU:0',and only support CPU GPU TPU")
    return model
  model.stop_training=True
  print('Training of '+model.name+' has been stopped')
  str = device[1:6]
  gpus = tf.config.experimental.list_logical_devices(device[1:4])
  for gpu in gpus:
    if(gpu.name.find(str)!=-1):
      checkpoint_path = f"training/cache/ckpt/epoch_{epoch}"
      if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
      checkpoint_prefix = os.path.join(checkpoint_path,"ckpt")
      model.save_weights(filepath=checkpoint_prefix)
      print('Weights of '+model.name+' has been saved in '+checkpoint_prefix)
      with tf.device(device):
        new_model=create_model_method()
        new_model.load_weights(filepath=checkpoint_prefix)
        new_epoch=epoch-1
        print(f"New model's device has been swtiched to {device},training is ready to run")
        print(f"WARNING:if you are using epoch-releated lrate,the start epoch should be {new_epoch}")
        return new_model
  print("ERROR:All devices missmatched,anything wrong with the 'device' str?")

