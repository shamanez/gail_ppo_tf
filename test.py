#!/usr/bin/python3
import argparse
import gym
import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from network_models.discriminator import Discriminator
from algo.ppo import PPOTrain

import tensorflow as tf
import numpy as np
import random
import time
import sys
from scipy.spatial.distance import cosine
import pdb

from utils.accum_trainer import AccumTrainer
from utils.ops import sample_action
from scene_loader import THORDiscreteEnvironment as Environment
from dagger_policy_generators import SmashNet, ShortestPathOracle

from dagger_constants import ACTION_SIZE, GAMMA, LOCAL_T_MAX, ENTROPY_BETA, VERBOSE, VALID_TASK_LIST, NUM_VAL_EPISODES, VALIDATE, VALIDATE_FREQUENCY, SUCCESS_CUTOFF, MAX_VALID_STEPS
import argparse
import gym
import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from network_models.discriminator import Discriminator
from algo.ppo import PPOTrain

scene_scope='bathroom_02'
task_scope=26  #26 43 53 32 41
env = Environment({'scene_name': scene_scope,'terminal_state_id': int(task_scope)})

act=3
next_obs,is_terminal,is_collided=env.step(act)


'''
0=mover forward
1=turn left
2=turn right
3=Move down
'''