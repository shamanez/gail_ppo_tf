import tensorflow as tf
import threading
import numpy as np
import pdb as b

import signal
import random
import os
import time

from Expert_Traj import Expert

from scene_loader import THORDiscreteEnvironment as Environment
from dagger_constants import ACTION_SIZE, PARALLEL_SIZE, INITIAL_ALPHA_LOW, INITIAL_ALPHA_HIGH, INITIAL_ALPHA_LOG_RATE, INITIAL_DIFFIDENCE_RATE, MAX_TIME_STEP, CHECKPOINT_DIR, LOG_FILE, RMSP_EPSILON, RMSP_ALPHA, GRAD_NORM_CLIP, USE_GPU, NUM_CPU, TASK_TYPE, TRAIN_TASK_LIST, VALID_TASK_LIST, DYNAMIC_VALIDATE, ENCOURAGE_SYMMETRY

if __name__ == '__main__':
	network_scope = TASK_TYPE #Task type is navigation
	list_of_tasks = TRAIN_TASK_LIST #{'bathroom_02': ['26', '37', '43', '53', '16', '28', '32', '41']}}
	scene_scopes = list_of_tasks.keys() #dict_keys(['bathroom_02'])
	global_t = 0
	stop_requested = False

	branches = []
	for scene in scene_scopes:
		for task in list_of_tasks[scene]:
			branches.append((scene, task)) 

	print("Total navigation tasks: %d" % len(branches))
	NUM_TASKS = len(branches)



	for i in range(1):#NUM_TASKS):
		scene, task = branches[i] #Here task number is actually the target 
		print("Printing the scene and the  task type:",scene,task)


		training_thread = Expert(i,  
								MAX_TIME_STEP,
								network_scope = "thread-%d"%(i+1), #theread1 
								scene_scope = scene, #single scene
								task_scope = task)

		key = scene + "-" + task

		#while global_t < MAX_TIME_STEP and not stop_requested:
		training_thread.save_expert()

   
