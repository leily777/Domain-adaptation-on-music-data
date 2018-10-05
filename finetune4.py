# -*- coding: utf-8 -*-
"""
With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. 
Specify the configuration settings at the beginning according to your 
problem.
This script was written for TensorFlow 1.0 and come with a blog post 
you can find here:
  
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert 
contact: f.kratzert(at)gmail.com
"""
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from alexnet4 import AlexNet
from datagenerator import ImageDataGenerator
#from operator import add


"""
Configuration settings
"""

# Path to the textfiles for the trainings and validation set
train_file = 'trainer.txt'
val_file = 'tgt_tester.txt'
tgt_file = 'tgt_tester.txt'

# Learning params
learning_rate = 0.001
num_epochs = 1
batch_size = 2

# Network params
dropout_rate = 0.5
num_classes = 4
train_layers = ['fc8', 'fc7','fc9','fc10','fc10_1','fc6']

# How often we want to write the tf.summary data to disk
display_step = 1

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "./tboard/run5/"
checkpoint_path = "./modelchkpt2/"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)


# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [1, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size//2, num_classes])
#y = tf.slice(y, [0, 0], [batch_size//2, -1])
y_dom = tf.placeholder(tf.float32, [1, 2])   # 2 is no. of domain classes
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers, batch_size)

# Link variable to model output
#score1 = model.fc8
score2 = model.fc10
score2_1 = model.fc10_1


# List of trainable variables of the layers we want to train
train_layers1 = ['fc9','fc10','fc6']
var_list1_full = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers1]
train_layers2 = ['fc9_1','fc10_1','fc6']
var_list2_full = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers2]


# Op for calculating the loss
#with tf.name_scope("cross_ent"):
#  y_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score1, labels = y))  

# Op for calculating the loss
with tf.name_scope("cross_ent_dom"):
  dom_loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score2, labels = y_dom))
  dom_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score2_1, labels = y_dom))
  
#loss = tf.add(y_loss,dom_loss,name='total_loss')

# Train op
with tf.name_scope("train"):
  # Get gradients of all trainable variables
  
  var_list1 = [var_list1_full[1],var_list1_full[3],var_list1_full[5] ]
  var_list2 = [var_list2_full[1],var_list2_full[3],var_list2_full[5] ]
  print(var_list1,var_list2,'\n')
  gradients1 = tf.gradients([dom_loss1], var_list1)
  gradients2 = tf.gradients([dom_loss2], var_list2)

  gradients = list(zip(gradients1, var_list1))
  
  # Create optimizer and apply gradient descent to the trainable variables
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary  
#for gradient, var in gradients:
#  tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary  
#for var in var_list:
#  tf.summary.histogram(var.name, var)
  
# Add the loss to summary
#tf.summary.scalar('y_loss', y_loss)
#tf.summary.scalar('dom_loss', dom_loss)
#tf.summary.scalar('total_loss', loss)
  

# Evaluation op: Accuracy of the model
#with tf.name_scope("accuracy1"):
#  correct_pred = tf.equal(tf.argmax(score1, 1), tf.argmax(y, 1))
#  accuracy1 = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#Dom accuracy
with tf.name_scope("accuracy2"):
  correct_pred = tf.equal(tf.argmax(score2, 1), tf.argmax(y_dom, 1))
  accuracy2 = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  
#total accuracy
#accuracy = tf.reduce_mean([accuracy1,accuracy2],name='total_acc')

# Add the accuracy to the summary
#tf.summary.scalar('accuracy1', accuracy1)
tf.summary.scalar('accuracy2', accuracy2)
#tf.summary.scalar('total_accuracy',accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(train_file, 
                                     horizontal_flip = False, shuffle = True,nb_classes = num_classes)
val_generator = ImageDataGenerator(val_file, shuffle = False,nb_classes = num_classes)
tgt_generator = ImageDataGenerator(tgt_file, shuffle = True,nb_classes = num_classes) 

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / (batch_size//2) ).astype(np.int16)
train_batches_per_epoch = 2
val_batches_per_epoch = np.floor(val_generator.data_size / (batch_size//2) ).astype(np.int16)
val_batches_per_epoch = 1

#Creating permanent domain labels for trainig
Dbatch = np.hstack( [np.zeros(batch_size//2, dtype=np.int32)])#, np.ones(batch_size//2, dtype=np.int32)])
batch_ydom = np.eye(2,dtype=np.float32)[Dbatch]

# Start Tensorflow session
with tf.Session() as sess:
 
  # Initialize all variables
  sess.run(tf.global_variables_initializer())
  
  # Add the model graph to TensorBoard
  writer.add_graph(sess.graph)
  
  # Load the pretrained weights into the non-trainable layer
  model.load_initial_weights(sess)
  #saver.restore(sess, r"C:\Users\divy\Desktop\internship\finalcode\finetuneAlexnet\modelchkpt2_music_AlexNet1_full\model_epoch25.ckpt")
  
  print("{} Start training...".format(datetime.now()))
  print("{} Open Tensorboard at --logdir {}".format(datetime.now(), 
                                                    filewriter_path))
  
  # Loop over number of epochs
  for epoch in range(num_epochs):
    
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        
        step = 1
        
        while step < train_batches_per_epoch:
            
            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(batch_size//2)
            tgt_x,tgt_y =  tgt_generator.next_batch(batch_size//2)
            #batch_xs = np.vstack( [batch_xs, tgt_x] )
            #batch_ys = batch_ys[:batch_size//2]
            
            # And run the training op
            for i in range(len(var_list1_full)):
                sess.run( var_list2_full[i].assign(var_list1_full[i]) )
            
            sc1,sc2, res1, res2 = sess.run([score2,score2_1,gradients1,gradients2], feed_dict={x: batch_xs,
                                          #y: batch_ys,
                                          y_dom: batch_ydom,    #instead of y: batch_ys
                                          keep_prob: 1.0})
            print('\n final_outputs\n')
            print(sc1,sc2)
            print('flipped gradients \n')
            print(res1[2].shape,res1[1].shape,res1[0].shape)
            print(res1[2],res1[1][:3],res1[0][:3] )
            print('not flipped gradients \n')
            print(res2[2].shape,res2[1].shape,res2[0].shape)
            print(res2[2],res2[1][:3],res2[0][:3] )
            print('\n')

            step += 1
            
        
        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()
        train_generator.reset_pointer()
        
        print("{} Saving checkpoint of model...".format(datetime.now()))  
        
        #save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)  
        
        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
        
