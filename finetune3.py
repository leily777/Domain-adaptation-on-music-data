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


The problem maybe diff in abss value of both cross-ent losses

"""

import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from alexnet1 import AlexNet
from datagenerator import ImageDataGenerator
from operator import add


"""
Configuration settings
"""

# Path to the textfiles for the trainings and validation set
train_file = 'trainer.txt'
val_file = 'tgt_tester.txt'
tgt_file = 'tgt_tester.txt'

# Learning params
learning_rate = 0.00001
num_epochs = 40
batch_size = 128

# Network params
dropout_rate = 0.5
num_classes = 4
train_layers = ['fc8', 'fc7','fc9','fc10','fc6']

# How often we want to write the tf.summary data to disk
display_step = 1

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "./tboard/run4/"
checkpoint_path = "./modelchkpt2/"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)


# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size//2, num_classes])
#y = tf.slice(y, [0, 0], [batch_size//2, -1])
y_dom = tf.placeholder(tf.float32, [batch_size, 2])   # 2 is no. of domain classes
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers, batch_size)

# Link variable to model output
score1 = model.fc8
score2 = model.fc10

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
  y_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score1, labels = y))  

# Op for calculating the loss
with tf.name_scope("cross_ent_dom"):
  dom_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score2, labels = y_dom))
  dom_loss = -1.0*dom_loss

#TYPES OF LOSSES
  
#TYPE-I
#loss = tf.add(y_loss,dom_loss)

#TYPE-II
diff_xy = tf.subtract(y_loss,dom_loss)
diff_xy = tf.abs(diff_xy)
loss1 = tf.multiply(y_loss,dom_loss)
loss = tf.add(loss1,diff_xy)

#TYPE-III
#diff_xy = tf.subtract(y_loss,dom_loss)
#diff_xy = tf.abs(diff_xy)
#loss1 = tf.multiply(y_loss,dom_loss)
#loss1 = tf.abs(loss1)
#loss = tf.add(loss1,diff_xy)


# Train op
with tf.name_scope("train"):
  # Get gradients of all trainable variables
  gradients = tf.gradients(loss, var_list)
  gradients = list(zip(gradients, var_list))
  
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
tf.summary.scalar('y_loss', y_loss)
tf.summary.scalar('dom_loss', dom_loss)
tf.summary.scalar('total_loss', loss)
  

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy1"):
  correct_pred = tf.equal(tf.argmax(score1, 1), tf.argmax(y, 1))
  accuracy1 = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#Dom accuracy
with tf.name_scope("accuracy2"):
  correct_pred = tf.equal(tf.argmax(score2, 1), tf.argmax(y_dom, 1))
  accuracy2 = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  
#total accuracy
accuracy = tf.reduce_mean([accuracy1,accuracy2],name='total_acc')

# Add the accuracy to the summary
tf.summary.scalar('accuracy1', accuracy1)
tf.summary.scalar('accuracy2', accuracy2)
tf.summary.scalar('total_accuracy',accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(train_file, 
                                     horizontal_flip = False, shuffle = True,nb_classes = num_classes)
val_generator = ImageDataGenerator(val_file, shuffle = True,nb_classes = num_classes)
tgt_generator = ImageDataGenerator(tgt_file, shuffle = True,nb_classes = num_classes) 

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / (batch_size//2) ).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / (batch_size//2) ).astype(np.int16)


#Creating permanent domain labels for trainig
Dbatch = np.hstack( [np.zeros(batch_size//2, dtype=np.int32), np.ones(batch_size//2, dtype=np.int32)])
batch_ydom = np.eye(2,dtype=np.float32)[Dbatch]

# Start Tensorflow session
with tf.Session() as sess:
 
  # Initialize all variables
  sess.run(tf.global_variables_initializer())
  
  # Add the model graph to TensorBoard
  writer.add_graph(sess.graph)
  
  # Load the pretrained weights into the non-trainable layer
  #model.load_initial_weights(sess)
  saver.restore(sess, r"C:\Users\divy\Desktop\internship\finalcode\finetuneAlexnet\modelchkpt1_music_AlexNet1\model_epoch40.ckpt")
  
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
            batch_xs = np.vstack( [batch_xs, tgt_x] )
            #batch_ys = batch_ys[:batch_size//2]
            
            # And run the training op
            sess.run(train_op, feed_dict={x: batch_xs,
                                          y: batch_ys,
                                          y_dom: batch_ydom,    #instead of y: batch_ys
                                          keep_prob: dropout_rate})
            
            # Generate summary with the current batch of data and write to file
            if step%display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: batch_xs,
                                                        y: batch_ys,
                                                        y_dom: batch_ydom, 
                                                        keep_prob: 1.})
                writer.add_summary(s, epoch*train_batches_per_epoch + step)
                
            step += 1
            
        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        test_acc = [0.0,0.0,0.0,0.0]
        test_count = 0
        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch(batch_size//2)
            tgt_x,tgt_y =  tgt_generator.next_batch(batch_size//2)
            batch_tx = np.vstack( [batch_tx, tgt_x] )
            #batch_ty = batch_ty[:batch_size//2]

            acc = sess.run([y_loss,dom_loss,accuracy1,accuracy2], feed_dict={x: batch_tx,
                                                y: batch_ty,
                                                y_dom: batch_ydom, 
                                                keep_prob: 1.})

            test_acc = list( map(add, test_acc, acc) )
            test_count += 1
        test_acc = [x / test_count for x in test_acc]
        print("{} Validation Accuracy = {}".format(datetime.now(), test_acc))
        
        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()
        train_generator.reset_pointer()
        
        print("{} Saving checkpoint of model...".format(datetime.now()))  
        
        #save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)  
        
        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
        
