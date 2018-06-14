import tensorflow as tf
import numpy as np
import pdb
#from siamese import S_Layer


class Discriminator:
    def __init__(self,s_class):
        self.scope = tf.get_variable_scope().name
        self.E_state_D = tf.placeholder("float", [None, 2048, 4],name='E_state')
        self.E_target_s_D = tf.placeholder("float", [None, 2048, 4],name='E_target_state')
        self.E_state_flat_D=tf.reshape(self.E_state_D, [-1, 8192])
        self.E_target_flat_D=tf.reshape(self.E_target_s_D,[-1,8192])

        self.A_state_D = tf.placeholder("float", [None, 2048, 4],name='A_state')
        self.A_target_s_D = tf.placeholder("float", [None, 2048, 4],name='A_target_state')
        self.A_state_flat_D=tf.reshape(self.A_state_D, [-1, 8192])
        self.A_target_flat_D=tf.reshape(self.A_target_s_D,[-1,8192])

        self.expert_a = tf.placeholder(dtype=tf.int32, shape=[None])
        expert_a_one_hot = tf.one_hot(self.expert_a, depth=4)  #expert action as one hot action
        self.agent_a = tf.placeholder(dtype=tf.int32, shape=[None])  
        agent_a_one_hot = tf.one_hot(self.agent_a, depth=4)



        #E_S_Output =S_Layer(self.E_state_flat_D,self.E_target_flat_D)  #State target concatenated outputs 
        #A_S_Output =S_Layer(self.A_state_flat_D,self.A_target_flat_D)
        E_S_Output =s_class.siamese_f(self.E_state_flat_D,self.E_target_flat_D)  #Use the function from the siamese class
        A_S_Output =s_class.siamese_f(self.A_state_flat_D,self.A_target_flat_D)

        self.expert_s=E_S_Output
        self.agent_s=A_S_Output


    
        

        with tf.variable_scope('discriminator'):
 
            # add noise for stabilise training            
            expert_a_one_hot += tf.random_normal(tf.shape(expert_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2 #This can cause problems in my environment so should be careful
            expert_s_a = tf.concat([self.expert_s, expert_a_one_hot], axis=1) #making state action pairs 
            

            # add noise for stabilise training
            agent_a_one_hot += tf.random_normal(tf.shape(agent_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2 #Add some noise to the algorithm
            agent_s_a = tf.concat([self.agent_s, agent_a_one_hot], axis=1)  #concat agents state action pairs 


            with tf.variable_scope('network') as network_scope:
                prob_1 = self.construct_network(input=expert_s_a)  #construct discriminator - You need to put geenrated data well as expert data in to the discriminator 
                network_scope.reuse_variables()  # share parameter  #Use same parameters very useful
                prob_2 = self.construct_network(input=agent_s_a) #output probability for the agent

            with tf.variable_scope('loss'):
                loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(prob_1, 0.01, 1))) #this is like the batch loss
                loss_agent = tf.reduce_mean(tf.log(tf.clip_by_value(1 - prob_2, 0.01, 1))) #SAme
                loss = loss_expert + loss_agent
                loss = -loss  #loss need to be minimize 
                tf.summary.scalar('discriminator', loss)

            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(loss) #train op
            self.rewards = tf.log(tf.clip_by_value(prob_2, 1e-10, 1))  # log(P(expert|s,a)) larger is better for agent #reward is directly calculating for 

   

    def construct_network(self, input): # Discriminator network  This can be a convolutional neural network 
        layer_1 = tf.layers.dense(inputs=input, units=512, activation=tf.nn.leaky_relu, name='layer1')
        layer_2 = tf.layers.dense(inputs=layer_1, units=512, activation=tf.nn.leaky_relu, name='layer2')
        layer_3 = tf.layers.dense(inputs=layer_2, units=256, activation=tf.nn.leaky_relu, name='layer3')
        prob = tf.layers.dense(inputs=layer_3, units=1, activation=tf.sigmoid, name='prob')
        return prob

    def train(self, expert_s,expert_t, expert_a, agent_s,agent_t, agent_a):
        return tf.get_default_session().run(self.train_op, feed_dict={self.E_state_D: expert_s,
                                                                      self.E_target_s_D: expert_t,  
                                                                      self.expert_a: expert_a,
                                                                      self.A_state_D: agent_s,
                                                                      self.A_target_s_D: agent_t,
                                                                      self.agent_a: agent_a})


    def get_rewards(self, agent_s,agent_t, agent_a):
        return tf.get_default_session().run(self.rewards, feed_dict={ self.A_state_D: agent_s,
                                                                      self.A_target_s_D:agent_t, 
                                                                      self.agent_a: agent_a})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


    def _fc_weight_variable_D(self, shape, name='W_fc'):
        input_channels = shape[0]
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial, name=name)

    def _fc_bias_variable_D(self, shape, input_channels, name='b_fc'):
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial, name=name)

