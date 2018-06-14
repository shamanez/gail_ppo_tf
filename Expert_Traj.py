from scene_loader import THORDiscreteEnvironment as Environment
from dagger_policy_generators import SmashNet, ShortestPathOracle
import numpy as np
import random
import pdb
import time
import sys
from dagger_constants import ACTION_SIZE, GAMMA, LOCAL_T_MAX, ENTROPY_BETA, VERBOSE, VALID_TASK_LIST, NUM_VAL_EPISODES, VALIDATE, VALIDATE_FREQUENCY, SUCCESS_CUTOFF, MAX_VALID_STEPS


task_list=['26', '37', '43', '53', '16', '28', '32', '41']
scene_scope='bathroom_02'
task_scope=26


'''

env = Environment({            #This is where you access in to the environment  #scene_loader import THORDiscreteEnvironment 
        	'scene_name': scene_scope,
        	'terminal_state_id': int(task_scope)})
     

env.reset()
oracle = ShortestPathOracle(env, ACTION_SIZE) #Get the probabilities of the shortest paths to go to exat position

oracle_pi = oracle.run_policy(env.current_state_id)

pi_values=oracle_pi

s_t = env.s_t

print(env.current_state_id)


def ch_ac(pi_values):

	#print(pi_values)
	r = random.random() * np.sum(pi_values)

	#print(r)
	values = np.cumsum(pi_values)
	#print(values)

	for i in range(len(values)):
        	if values[i] >= r:
        		return i

action=ch_ac(pi_values)


env.step(action)

'''


class Expert(object):



	def __init__(self,
			   loop_index,
               max_global_time_step,
               network_scope="network",
               scene_scope="scene",
               task_scope="task"):

    	
	    self.max_global_time_step = max_global_time_step
	    self.network_scope = network_scope #assiciated with the thread number
	    self.scene_scope = scene_scope #Whether this is kitchen or not
	    self.task_scope = task_scope #This the targe
	    self.scopes = [network_scope, scene_scope, task_scope] # ["thread-n", "scene", "target"]
	    self.env = None
	    self.local_t = 0
	    self.oracle = None



	def choose_action(self,oracle_pi_values):

		pi_values=oracle_pi_values
		

		r = random.random() * np.sum(pi_values)
		values = np.cumsum(pi_values)
		for i in range(len(values)):
			if values[i] >= r: return i	



	def open_file_and_save(self,file_path, data):
		try:
			with open(file_path, 'ab') as f_handle:
				np.savetxt(f_handle, data, fmt='%s')
		except FileNotFoundError:
			with open(file_path, 'wb') as f_handle:
				np.savetxt(f_handle, data, fmt='%s')

	



	def save_expert(self):


		states = []
		action_list = []  
		target_list=[]

		
		for i in range(2):#180):
			self.env = Environment({'scene_name': self.scene_scope,'terminal_state_id': int(self.task_scope)})
			print("Starting -","*******************************************************************-----------",i)
			print("From the environment Current State ID--",self.env.current_state_id)
			print("From the environment Traget State ID--",self.env.terminal_state_id)
			print("Frm the environment Number of possible states",self.env.n_locations)
			print("________________________________________________________________________________")


			self.oracle = ShortestPathOracle(self.env, ACTION_SIZE)

			while(not(self.env.terminal)):

				s_t = self.env.s_t; 
				target=self.env.s_target
				#self.oracle = ShortestPathOracle(self.env, ACTION_SIZE)
				oracle_pi = self.oracle.run_policy(self.env.current_state_id) #get the policy of the oracle which means the shotest path kind of action in the given state
				action = self.choose_action(oracle_pi)

				states.append(s_t) #stack action
				action_list.append(action)
				target_list.append(target)
			


				self.env.step(action)  #here we change the next step

				is_terminal = self.env.terminal
				is_collided =self.env.collided
				self.local_t += 1

				if is_collided:
					print("Wrong action-------- Error Error Error Error Error Error Error Error Error Error")
					break

				# s_t1 -> s_t
				self.env.update() #update the new state	
				#self.env.reset() #With this 

			print("Done with one epoach one start state to end goal ")
			self.env.reset()

		states = np.reshape(states, newshape=[-1] + list([8192]))
		target_list=np.reshape(target_list, newshape=[-1] + list([8192]))
		self.open_file_and_save('trajectory/observations.csv', states)
		self.open_file_and_save('trajectory/actions.csv', action_list)
		self.open_file_and_save('trajectory/targets.csv', target_list)




    