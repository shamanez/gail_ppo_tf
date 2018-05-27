import tensorflow as tf


class Discriminator:
    def __init__(self, env):
        """
        :param env:
        Output of this Discriminator is reward for learning agent. Not the cost.  #prety important to understand 
        Because discriminator predicts  P(expert|s,a) = 1 - P(agent|s,a).
        """

        with tf.variable_scope('discriminator'):
            self.scope = tf.get_variable_scope().name
            self.expert_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.observation_space.shape)) #to feed the exper's state
            self.expert_a = tf.placeholder(dtype=tf.int32, shape=[None])  #to feed the expert actions inside the session
            expert_a_one_hot = tf.one_hot(self.expert_a, depth=env.action_space.n)  #expert action as one hot action

            '''#If we want to use a LSTM to 
            self.expert_s = tf.placeholder(tf.float32, [None, max_length, state_dim])
            self.expert_a = tf.placeholder(tf.float32, [None, max_length, action])
            expert_a_one_hot = tf.one_hot(self.expert_a, depth=number of actions)
            '''

            # add noise for stabilise training
            expert_a_one_hot += tf.random_normal(tf.shape(expert_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2 #This can cause problems in my environment so should be careful
            expert_s_a = tf.concat([self.expert_s, expert_a_one_hot], axis=1) #making state action pairs 

            self.agent_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.observation_space.shape))  #agent state 
            self.agent_a = tf.placeholder(dtype=tf.int32, shape=[None])  
            agent_a_one_hot = tf.one_hot(self.agent_a, depth=env.action_space.n)


            '''#If we want to use a LSTM to 
            self.agent_s = tf.placeholder(tf.float32, [None, max_length, state_dim])
            self.agent_a = tf.placeholder(tf.float32, [None, max_length, action])
            agent_a_one_hot = tf.one_hot(self.expert_a, depth=number of actions)
            '''
            # add noise for stabilise training
            agent_a_one_hot += tf.random_normal(tf.shape(agent_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2
            agent_s_a = tf.concat([self.agent_s, agent_a_one_hot], axis=1)  #concat agents state action pairs 

            with tf.variable_scope('network') as network_scope:
                prob_1 = self.construct_network(input=expert_s_a)  #construct discriminator - You need to put geenrated data well as expert data in to the discriminator 
                network_scope.reuse_variables()  # share parameter  #Use same parameters very useful
                prob_2 = self.construct_network(input=agent_s_a) #output probability for the agent


            ''' #This is a  lstm discrmonator    
            with tf.variable_scope('network') as network_scope:
                prob_1 = self.construct_network(input=expert_s_a)  #construct discriminator - You need to put geenrated data well as expert data in to the discriminator 
                network_scope.reuse_variables()  # share parameter  #Use same parameters very useful
                prob_2 = self.construct_network(input=agent_s_a) #output probability for the agent
            '''

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
        layer_1 = tf.layers.dense(inputs=input, units=20, activation=tf.nn.leaky_relu, name='layer1')
        layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.nn.leaky_relu, name='layer2')
        layer_3 = tf.layers.dense(inputs=layer_2, units=20, activation=tf.nn.leaky_relu, name='layer3')
        prob = tf.layers.dense(inputs=layer_3, units=1, activation=tf.sigmoid, name='prob')
        return prob

    def train(self, expert_s, expert_a, agent_s, agent_a):

        return tf.get_default_session().run(self.train_op, feed_dict={self.expert_s: expert_s,
                                                                      self.expert_a: expert_a,
                                                                      self.agent_s: agent_s,
                                                                      self.agent_a: agent_a})


    def get_rewards(self, agent_s, agent_a):
        return tf.get_default_session().run(self.rewards, feed_dict={self.agent_s: agent_s,
                                                                     self.agent_a: agent_a})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

