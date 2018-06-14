import tensorflow as tf
import numpy as np
import pdb


class Siamese_L:
    def __init__(self,state,target):


        self.siamese_s=construct_Siamese(self.state_flat)
        self.siamese_t=construct_Siamese(self.target_flat)
        self.obs = tf.concat(values=[self.siamese_s, self.siamese_t], axis=1,name='obs')
        #return self.obs
    




    def construct_Siamese(self, input): # Discriminator network  This can be a convolutional neural network 

        with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
        layer_1 = tf.layers.dense(inputs=input, units=512, activation=tf.nn.leaky_relu, name='layer1')
        layer_2 = tf.layers.dense(inputs=layer_1, units=512, activation=tf.nn.leaky_relu, name='layer2')
        #layer_3 = tf.layers.dense(inputs=layer_2, units=256, activation=tf.nn.leaky_relu, name='layer3')
        return layer2    


        #with tf.variable_scope('network') as network_scope:






        
