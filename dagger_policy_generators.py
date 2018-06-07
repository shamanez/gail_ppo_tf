# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import networkx as nx

import pdb
import matplotlib.pyplot as plt

class PolicyGenerator(object):

    def __init__(self):
        pass

    def run_policy(self, state):
        return NotImplementedError()

class ShortestPathOracle(PolicyGenerator):  #This is for find the exerts policies  this is a subclass ofpolict generatip

    def __init__(self, env, action_size): #Here you need to input the  action
        self.shortest_path_actions = self._calculate_shortest_paths(env, action_size) #get the list of shortest path actions a

    def _calculate_shortest_paths(self, env, action_size):

        s_next_s_action = {}  #This is a tuple collecting current sttate and next state and the action 
        G = nx.DiGraph()  #This is a graph use to keep the combining paths of states

        for s in range(env.n_locations): #Will traverse through all the locations 
          for a in range(action_size): #at each state the agent can get 4 actions
            next_s = env.transition_graph[s, a] #Check the next state given the currrent state and the action
            if next_s >= 0: #This means next state is a possible one not -1 so can traverse 
              s_next_s_action[(s, next_s)] = a  #we note the by taking action a we can go from state s to s_next
              G.add_edge(s, next_s) #need to connect the graph  (if the transition from one state to another state is possible we draw an edge)

#Here the G is made up of all the connections between nodes 
        best_action = np.zeros((env.n_locations, action_size), dtype=np.float) #Now need to find the best action (one hot best for each state)
        #nx.draw_networkx(G) 
        #plt.show() 

        #print("Terminal State ID from the Calculate Shortes Pth",env.terminal_state_id)
        for i in range(env.n_locations):  
          if i == env.terminal_state_id: #check if this location is the goal . Won't do anything below for loop will cotinue
            continue
          if env.shortest_path_distances[i, env.terminal_state_id] == -1: #this means from current state u can't go to terminal state
            continue
          #The following function will generate list of shortes paths from node one to the  
          for path in nx.all_shortest_paths(G, source=i, target=env.terminal_state_id): #check from current location to the terminal state #Heere the source in the start from ith node and go to terminal state

            action = s_next_s_action[(i, path[1])] #for one shorted path 
            best_action[i, action] += 1  #Add one action to the position #Best action to take in the node i to go to the terminal state

        action_sum = best_action.sum(axis=1, keepdims=True)
        action_sum[action_sum == 0] = 1  # prevent divide-by-zero
        shortest_path_actions = best_action / action_sum
        '''  
        import csv
        csvfile='/home/dl/Documents/gail_ppo_tf/x.csv'
        with open(csvfile, "w") as output:
          writer = csv.writer(output, lineterminator='\n')
          writer.writerows(shortest_path_actions)

        '''
        #for path in nx.all_shortest_paths(G, source=179, target=env.terminal_state_id): #source what ever to check
          #print(path)

        #print(shortest_path_actions)
        #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        return shortest_path_actions #get the shortest path actions to go to the erminal point starting from current location and moving to different positions. For each positions we get the probbilities of taking each of 4 actions

    def run_policy(self, s_t_id):
        return self.shortest_path_actions[s_t_id]

# Hulk smash net to defeat THOR challenge
class SmashNet(PolicyGenerator):##### we can use this to train our neural network
  """
    Implementation of the target-driven deep siamese actor-critic network from [Zhu et al., ICRA 2017]
    We use tf.variable_scope() to define domains for parameter sharing
  """
  def __init__(self,
               action_size,
               device="/cpu:0",
               network_scope="network",
               scene_scopes=["scene"]):

    self.action_size = action_size
    self.device = device

    self.pi = dict()

    self.W_fc1 = dict() #I think these a scene specific layers
    self.b_fc1 = dict()

    self.W_fc2 = dict()
    self.b_fc2 = dict()

    self.W_fc3 = dict()
    self.b_fc3 = dict()

    self.W_policy = dict()
    self.b_policy = dict()

    with tf.device(self.device):


#Need to understad this state why 4 elements ??????????/

      # state (input)
      self.s = tf.placeholder("float", [None, 2048, 4])  #Input vectors of 4 frames came out from the resnet 

      # target (input)
      self.t = tf.placeholder("float", [None, 2048, 4]) #Why four inputs for the target ?

      # "navigation" for global net, "thread-n" for local thread nets
      with tf.variable_scope(network_scope):
        # network key
        key = network_scope
        
        # flatten input
        self.s_flat = tf.reshape(self.s, [-1, 8192])
        self.t_flat = tf.reshape(self.t, [-1, 8192])

        # shared siamese layer
        self.W_fc1[key] = self._fc_weight_variable([8192, 512])
        self.b_fc1[key] = self._fc_bias_variable([512], 8192)

        h_s_flat = tf.nn.relu(tf.matmul(self.s_flat, self.W_fc1[key]) + self.b_fc1[key])
        h_t_flat = tf.nn.relu(tf.matmul(self.t_flat, self.W_fc1[key]) + self.b_fc1[key])
        h_fc1 = tf.concat(values=[h_s_flat, h_t_flat], axis=1)

        # shared fusion layer
        self.W_fc2[key] = self._fc_weight_variable([1024, 512])
        self.b_fc2[key] = self._fc_bias_variable([512], 1024)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2[key]) + self.b_fc2[key])

        for scene_scope in scene_scopes: #currently only one scene
          # scene-specific key

          key = self._get_key([network_scope, scene_scope])

          # "thread-n/scene"
          with tf.variable_scope(scene_scope):

            # scene-specific adaptation layer
            self.W_fc3[key] = self._fc_weight_variable([512, 512])
            self.b_fc3[key] = self._fc_bias_variable([512], 512)
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2, self.W_fc3[key]) + self.b_fc3[key])

            # weight for policy output layer
            self.W_policy[key] = self._fc_weight_variable([512, action_size]) #output layer weight 
            self.b_policy[key] = self._fc_bias_variable([action_size], 512)

            # policy (output)
            pi_ = tf.matmul(h_fc3, self.W_policy[key]) + self.b_policy[key] #policy selecting the optimum
            self.pi[key] = tf.nn.softmax(pi_)  #policy is done

  def run_policy(self, sess, state, target, scopes):
    key = self._get_key(scopes[:2])
    pi_out = sess.run( self.pi[key], feed_dict = {self.s : [state], self.t: [target]} )[0]
    return pi_out #get the probabulties of doing each action in the current state

  def prepare_loss(self, scopes): # only called by local thread nets

    # drop task id (last element) as all tasks in
    # the same scene share the same output branch
    scope_key = self._get_key(scopes[:-1]) # "thread-n/scene"
   

    with tf.device(self.device):

      # oracle policy
      self.opi = tf.placeholder("float", [None, self.action_size]) #policy of the expert

      # avoid NaN with clipping when value in pi becomes zero
      log_spi = tf.log(tf.clip_by_value(self.pi[scope_key], 1e-20, 1.0))  #log probabilities

      # cross entropy policy loss (output)
      policy_loss = - tf.reduce_sum(log_spi * self.opi, axis=1)

      self.loss = policy_loss

  # TODO: returns all parameters for current net
  def get_vars(self):
    var_list = [
      self.W_fc1, self.b_fc1,
      self.W_fc2, self.b_fc2,
      self.W_fc3, self.b_fc3,
      self.W_policy, self.b_policy,
    ]
    vs = []
    for v in var_list:
      vs.extend(v.values())
    return vs

  def sync_from(self, src_network, name=None):  #you need to copy global net variable to thelocal variables 
    src_vars = src_network.get_vars()
    dst_vars = self.get_vars()

    local_src_var_names = [self._local_var_name(x) for x in src_vars]
    local_dst_var_names = [self._local_var_name(x) for x in dst_vars]

    # keep only variables from both src and dst
    src_vars = [x for x in src_vars
      if self._local_var_name(x) in local_dst_var_names]
    dst_vars = [x for x in dst_vars
      if self._local_var_name(x) in local_src_var_names]

    sync_ops = []

    with tf.device(self.device):
      with tf.name_scope(name, "SmashNet", []) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)

  # variable (global/scene/task1/W_fc:0) --> scene/task1/W_fc:0
  # removes network scope from name it seems
  def _local_var_name(self, var):
    return '/'.join(var.name.split('/')[1:])

  def _get_key(self, scopes):
    return '/'.join(scopes)

  # weight initialization based on muupan's code
  # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
  def _fc_weight_variable(self, shape, name='W_fc'):
    input_channels = shape[0]
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _fc_bias_variable(self, shape, input_channels, name='b_fc'):
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv_weight_variable(self, shape, name='W_conv'):
    w = shape[0]
    h = shape[1]
    input_channels = shape[2]
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv_bias_variable(self, shape, w, h, input_channels, name='b_conv'):
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")
