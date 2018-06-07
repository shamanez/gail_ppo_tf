# -*- coding: utf-8 -*-
import sys
import h5py
import json
import numpy as np
import random
import skimage.io
from skimage.transform import resize
from constants import ACTION_SIZE
from constants import SCREEN_WIDTH
from constants import SCREEN_HEIGHT
from constants import HISTORY_LENGTH
import pdb

class THORDiscreteEnvironment(object):  #This is the main environemtns

  def __init__(self, config=dict()):

    # configurations
    self.scene_name          = config.get('scene_name', 'bedroom_04') #This is the scene name we are feeding
    self.random_start        = config.get('random_start', True)
    self.n_feat_per_locaiton = config.get('n_feat_per_locaiton', 1) # 1 for no sampling
    self.terminal_state_id   = config.get('terminal_state_id', 1) #This is the target
    self.initial_state       = config.get('initial_state', None) #Initial state

    
    self.h5_file_path = config.get('h5_file_path', 'data/%s.h5'%self.scene_name)
    self.h5_file      = h5py.File(self.h5_file_path, 'r')  #Acirding to this code this will open the data/bathroom_o2.h5 file
   

    self.locations   = self.h5_file['location'][()]
    self.rotations   = self.h5_file['rotation'][()] #This has angles f 0,90,270,360 It's like agle of rotation for each location
    self.n_locations = self.locations.shape[0]  #NUMBER OF POSSIBLE LOCATIONS IN THE bathroom file is 408


    self.terminals = np.zeros(self.n_locations)
    self.terminals[self.terminal_state_id] = 1
    #I Think we can have more than  one termona state
    self.terminal_states, = np.where(self.terminals) #Here single terminal state that is 26
    
    
    self.transition_graph = self.h5_file['graph'][()] #For each pint in the 408 grid it has 4 surroundin grid poiints   (-1) means obstacles where you cannot move
    self.shortest_path_distances = self.h5_file['shortest_path_distance'][()]  #This Graph Gives you  408 *408 amd points you need to travel to get to another location

    self.history_length = HISTORY_LENGTH   #Number of previous frames we stacked in
    self.screen_height  = SCREEN_HEIGHT
    self.screen_width   = SCREEN_WIDTH#Screen resoution 84*84

      

    # we use pre-computed fc7 features from ResNet-50
    # self.s_t = np.zeros([self.screen_height, self.screen_width, self.history_length])
    self.s_t      = np.zeros([2048, self.history_length]) #State representation take the 84*84 and send it through the resnet 50 and get the output from the 
    self.s_t1     = np.zeros_like(self.s_t) #is this for the next state?
    self.s_target = self._tiled_state(self.terminal_state_id) #This target state also have four taget status of CNN output


    self.reset() #resting the environment

  # public methods

  def reset(self):
    # randomize initial state
    if self.initial_state:  #This is non
      k = self.initial_state
    else:
      while True:      #randomly initailiaze the starting state
        k = random.randrange(self.n_locations)  #select a location randomly there are 408 for bathroom
        min_d = np.inf
        # check if target is reachable
        for t_state in self.terminal_states: #here we have single terminal state       
          dist = self.shortest_path_distances[k][t_state] #Check if this is realistic 
          min_d = min(min_d, dist) #get the distance if we use the shortest path
        # min_d = 0  if k is a terminal state
        # min_d = -1 if no terminal state is reachable from k
        if min_d > 0: break  #This is realistic once te shorteset path distance is acheivable from the current location to target we take that position as start state 


    # reset parameters
    self.current_state_id = k   #The current state ID 
    self.s_t = self._tiled_state(self.current_state_id) #get the currrent state  actually four of same state 




    self.reward   = 0
    self.collided = False
    self.terminal = False

  def step(self, action):

    #if self.terminal, 'step() called in terminal state' #only run if the this is not the  current state is terminal state
    k = self.current_state_id
    if self.transition_graph[k][action] != -1:  #check if the next state is an obsticle or -1
      self.current_state_id = self.transition_graph[k][action] #we go to a new state
      

      if self.terminals[self.current_state_id]:
        self.terminal = True
        self.collided = False
      else:
        self.terminal = False
        self.collided = False
    else:
      self.terminal = False
      self.collided = True
    #print("pringint thhe new state ID",self.current_state_id)
    #print("Prinintg expert's collided or not:--",self.collided)
    
    #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
     
    self.s_t1 = np.append(self.s_t[:,1:], self.state, axis=1)  #add the new state to the four states (history ) #here the self.state is a property

  def update(self):
    self.s_t = self.s_t1

  # private methods

  def _tiled_state(self, state_id):
#Here the state ID is the currrent state
    k = random.randrange(self.n_feat_per_locaiton)
    f = self.h5_file['resnet_feature'][state_id][k][:,np.newaxis]   #get the resnet feature of the state 
    return np.tile(f, (1, self.history_length)) #coping the target in to last for frames :) 

  def _reward(self, terminal, collided):
    # positive reward upon task completion
    if terminal: return 10.0
    # time penalty or collision penalty
    return -0.1 if collided else -0.01

  # properties

  @property
  def action_size(self):
    # move forward/backward, turn left/right for navigation
    return ACTION_SIZE

  @property
  def action_definitions(self):
    action_vocab = ["MoveForward", "RotateRight", "RotateLeft", "MoveBackward"]
    return action_vocab[:ACTION_SIZE]

  @property
  def observation(self):
    return self.h5_file['observation'][self.current_state_id]

  @property
  def state(self): #a class property
    # read from hdf5 cache
    k = random.randrange(self.n_feat_per_locaiton)
    return self.h5_file['resnet_feature'][self.current_state_id][k][:,np.newaxis]

  @property
  def target(self):
    return self.s_target

  @property
  def x(self):
    return self.locations[self.current_state_id][0]

  @property
  def z(self):
    return self.locations[self.current_state_id][1]

  @property
  def r(self):
    return self.rotations[self.current_state_id]

if __name__ == "__main__":
  scene_name = 'bedroom_04'

  env = THORDiscreteEnvironment({
    'random_start': True,
    'scene_name': scene_name,
    'h5_file_path': 'data/%s.h5'%scene_name
  })
