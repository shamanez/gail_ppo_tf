import tensorflow as tf
import numpy as np
import pdb

#from siamese import S_Layer


class Policy_net:
    def __init__(self, name: str,s_class):


        self.state = tf.placeholder("float", [None, 2048, 4],name='state')
        self.target_s = tf.placeholder("float", [None, 2048, 4],name='target_state')
        self.state_flat=tf.reshape(self.state, [-1, 8192])
        self.target_flat=tf.reshape(self.target_s,[-1,8192])
        #S_Output =S_Layer(self.state_flat,self.target_flat)
        S_Output =s_class.siamese_f(self.state_flat,self.target_flat) #Using the siamese function from the siamese class
        #self.obs=S_Output.obs
        self.obs=S_Output
        
     
        with tf.variable_scope(name):

            with tf.variable_scope('policy_net'):  #a small neural network to build policy network 
                layer_1 = tf.layers.dense(inputs=self.obs, units=512, activation=tf.nn.relu) #take observations as inputs
                layer_2 = tf.layers.dense(inputs=layer_1, units=512, activation=tf.nn.relu)
                layer_3 = tf.layers.dense(inputs=layer_2, units=512, activation=tf.nn.relu)
                layer_4 = tf.layers.dense(inputs=layer_3, units=4, activation=tf.nn.relu)
                self.act_probs = tf.layers.dense(inputs=layer_4, units=4, activation=tf.nn.softmax)

            with tf.variable_scope('value_net'): #this is can be more like an actor critic but not that much sure -- 
                layer_1 = tf.layers.dense(inputs=self.obs, units=512, activation=tf.nn.relu) #take observations as inputs
                layer_2 = tf.layers.dense(inputs=layer_1, units=512, activation=tf.nn.relu)
                layer_3 = tf.layers.dense(inputs=layer_2, units=512, activation=tf.nn.relu)
                self.v_preds = tf.layers.dense(inputs=layer_3, units=1, activation=None)  #value prediction for this state 
                
            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1) #this is just to act randomly
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])
                      
            self.act_deterministic = tf.argmax(self.act_probs, axis=1) #action with maximum pr
            self.scope = tf.get_variable_scope().name

       
            
    def act(self, state,target, stochastic=True):
        if stochastic:
            return tf.get_default_session().run([self.act_stochastic,self.v_preds,self.act_probs],feed_dict={self.state: state,self.target_s:target})
        else:
            return tf.get_default_session().run([self.act_deterministic,self.v_preds,self.act_probs],feed_dict={self.state: state,self.target_s:target})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def _fc_weight_variable(self, shape, name='W_fc'):

        with tf.variable_scope(name):
            input_channels = shape[0]
            d = 1.0 / np.sqrt(input_channels)
            initial = tf.random_uniform(shape, minval=-d, maxval=d)
            return tf.Variable(initial, name=name)

    def _fc_bias_variable(self, shape, input_channels, name='b_fc'):
        with tf.variable_scope(name):
            d = 1.0 / np.sqrt(input_channels)
            initial = tf.random_uniform(shape, minval=-d, maxval=d)
            return tf.Variable(initial, name=name)


