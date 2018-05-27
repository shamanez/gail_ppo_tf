import tensorflow as tf
import pdb


class Policy_net:
    def __init__(self, name: str, env, temp=0.1):
        """
        :param name: string
        :param env: gym env
        :param temp: temperature of boltzmann distribution
        """

        ob_space = env.observation_space #this is what explain the observation  - [position of cart, velocity of cart, angle of pole, rotation rate of pole] 
        act_space = env.action_space  #This is number of actions which is two

        ''' #My OB  and Act spaces
        ob_space = [vector with 10 elements]
        act_space=[10 discrete actions] 

        '''



        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + list(ob_space.shape), name='obs')
            ''' #for the recurrent neural network
            self.obs = tf.placeholder(tf.float32, [None, max_length, frame_size])   #we do not need to define the maximum time steps since we use dynamic RNN
            '''
           
            def length(sequence):
                used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
                length = tf.reduce_sum(used, reduction_indices=1)
                length = tf.cast(length, tf.int32)
                return length

            with tf.variable_scope('policy_net'):  #a small neural network to build policy network 
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh) #take observations as inputs
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                layer_3 = tf.layers.dense(inputs=layer_2, units=act_space.n, activation=tf.tanh)
                self.act_probs = tf.layers.dense(inputs=tf.divide(layer_3, temp), units=act_space.n, activation=tf.nn.softmax)
            
            '''#this is for the dynamic rnn network for my model 
            with tf.variable_scope('policy_net'):  #a small neural network to build policy network 
                gru_cell1 = tf.contrib.rnn.GRUCell(128)
                if self.train:
                    gru_cell1 = tf.contrib.rnn.DropoutWrapper(gru_cell1, output_keep_prob=0.5)
                init_state = cell.zero_state(BATCH_SIZE, tf.float32)
                cell_output, final_stateTuple = tf.nn.dynamic_rnn(gru_cell1,  self.obs, sequence_length=length(self.obs), initial_state=init_state, time_major=False)
                self.act_probs = tf.layers.dense(inputs=tf.divide(final_stateTuple, temp), units=10, activation=tf.nn.softmax)
            '''

            with tf.variable_scope('value_net'): #this is can be more like an actor critic but not that much sure -- 
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                self.v_preds = tf.layers.dense(inputs=layer_2, units=1, activation=None)  #value prediction for this state 

            '''
            with tf.variable_scope('value_net'): 
                gru_cell1 = tf.contrib.rnn.GRUCell(128)
                if self.train:
                    gru_cell1 = tf.contrib.rnn.DropoutWrapper(gru_cell1, output_keep_prob=0.5)
                init_state = cell.zero_state(BATCH_SIZE, tf.float32)
                cell_output, final_stateTuple = tf.nn.dynamic_rnn(gru_cell1, self.obs, sequence_length=length(self.obs), initial_state=init_state, time_major=False)
                self.v_preds = tf.layers.dense(inputs=final_stateTuple, units=1, activation=None)  #value prediction for this state 

            '''

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1) #this is just to act randomly
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])
                      

            self.act_deterministic = tf.argmax(self.act_probs, axis=1) #action with maximum pr
            self.scope = tf.get_variable_scope().name
            
    def act(self, obs, stochastic=True):
        if stochastic:
            return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

