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
import pdb


class GailTraining(object):
	def argparser():
    	parser = argparse.ArgumentParser()
    	parser.add_argument('--logdir', help='log directory', default='log/train/gail')
    	parser.add_argument('--savedir', help='save directory', default='trained_models/gail')
    	parser.add_argument('--gamma', default=0.95)
    	parser.add_argument('--iteration', default=int(1e4))
    	return parser.parse_args()

    def main(args):
    	self.scene_scope=bathroom_02
    	self.task_scope=37  #26 43 53 32 41
    	self.env = Environment({'scene_name': self.scene_scope,'terminal_state_id': int(self.task_scope)})
    	self.env.reset()
        Policy = Policy_net('policy', env) #buiding the actor critic graph / object

        PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma) #gradiet updatror object or the graph
        pdb.set_trace()
        D = Discriminator(env) #discriminator of the Gan Kind of thing



