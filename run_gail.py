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
from siamese import S_Layer as SIAMESE




def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/gail')
    parser.add_argument('--savedir', help='save directory', default='trained_models/gail')
    parser.add_argument('--savedir2', help='save directory', default='trained_models')
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--iteration', default=int(1e4))
    parser.add_argument('--restore',type=bool, default=False)

    parser.add_argument('--model', help='number of model to test. model.ckpt-number', default='') #to restore variables
    parser.add_argument('--alg', help='chose algorithm one of gail, ppo, bc', default='gail')
    return parser.parse_args()


def main(args):
    scene_scope='bathroom_02'
    task_scope=26  #26 43 53 32 41
    env = Environment({'scene_name': scene_scope,'terminal_state_id': int(task_scope)})

    S_Class=SIAMESE()  #Creating  a siamese class -object
    

    
    Policy = Policy_net('policy',S_Class) #buiding the actor critic graph / object  , Passing     
    Old_Policy = Policy_net('old_policy',S_Class) #same thing as the other PPO

   
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma) #gradiet updatror object or the graph
    D = Discriminator(S_Class) #discriminator of the Gan Kind of thing


    '''
    batch_n=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Siamese')
    '''

    #Loading Expert Data State/Tragets etc 
    expert_observations = np.genfromtxt('trajectory/observations.csv')  #load expert demnetrations 
    expert_targets = np.genfromtxt('trajectory/targets.csv')
    expert_actions = np.genfromtxt('trajectory/actions.csv', dtype=np.int32)
    expert_observations= np.reshape(expert_observations, newshape=[-1,2048,4])
    expert_targets= np.reshape(expert_targets, newshape=[-1,2048,4])

   
    saver = tf.train.Saver() #Assign another save if you want to use BC weights 
    if args.restore: #We need a seperate saver only for assigning paramters from BC trained thing
        saver2=tf.tran.Saver([tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy'),tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Siamese')])


    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir, sess.graph)
        sess.run(tf.global_variables_initializer()) #here already variables get intialized both old policy and new policy net

        if  args.restore:
            if args.model == '':
                saver2.restore(sess, args.modeldir+'/'+args.alg+'/'+'shamane.ckpt')
                print("Model Reastored")
            else:
                saver.restore(sess, args.modeldir+'/'+args.alg+'/'+'model.ckpt-'+args.model)

        

        
        success_num = 0 #This is use to check whether my agent went to the terminal point

        #var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


        for iteration in range(100000):#args.iteration):#Here start the adversial training
            print("Starting ........ The Iteration---------------------------------------------------- :",iteration)
            observations = []
            actions = []
            #rewards = []
            targets = [] #for the gail
            v_preds = []
            run_policy_steps = 0

            
          
            while(True):  #Here what is happenning is , this again samples  trajectories from untrain agent
                run_policy_steps += 1
                obs = np.stack([env.s_t]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs #Initial observation
                target = np.stack([env.s_target ]).astype(dtype=np.float32) #This is to make sure that input is [batch_size,2048,4]


                act, v_pred,prob = Policy.act(state=obs,target=target,stochastic=True) # Agents action and values 

                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)


                
                observations.append(obs)  #save the set of observations 
                targets.append(target)
                actions.append(act) #save the set of actions 
                v_preds.append(v_pred) 


                #next_obs, reward, done, info = env.step(act)  #get the next observation and reward acording to the observation
                next_obs,is_terminal,is_collided=env.step(act)

                if is_terminal:
                    success_num=success_num+1
                    print("Congratz yoy just reach the terminal state which is:",env.terminal_state_id)

                if is_collided:
                    print("Bad Luck your agent just collided couldn't made it  to the terminal state which is :",env.terminal_state_id)


                if (is_terminal or is_collided or (run_policy_steps==100)):  #run one episode till the termination
                    print("Number Of Exploration by the AGENT:",run_policy_steps)
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value #this list use to update the parameters of the calue net
                    print("Environment is resetting after the collition/Terminal")
                    obs = env.reset()
                    #reward = -1
                    break  #with tihs vreak all obsercation ,action and other lists get empty 

             
            
            #print(sum(rewards))


            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                               , iteration)
            #writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               #, iteration)

               

            if success_num >= 5000:
                saver.save(sess, args.savedir + '/model.ckpt')
                print('Clear!! Model saved.')
                break
            #else:
                #success_num = 0

            
            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, newshape=[-1,2048,4]) #collect observations 
            targets=np.reshape(targets,newshape=[-1,2048,4])
            actions = np.array(actions).astype(dtype=np.int32) #collect the actions 


            
            
            # train discriminator  #Here comes the Discriminator !!        
            Dis_input=[expert_observations,expert_targets,expert_actions,observations,targets,actions]
            observations.shape[0]
            expert_observations.shape[0]

            if observations.shape[0]<expert_observations.shape[0]:
                High=observations.shape[0]
            else:
                High=expert_observations.shape[0]
            for i in range(100):
                sample_indices = np.random.randint(low=0, high=High,
                                                   size=32)
                sampled_inp_D = [np.take(a=a, indices=sample_indices, axis=0) for a in Dis_input] 
                
                D.train(expert_s=sampled_inp_D[0],        
                        expert_t=sampled_inp_D[1],
                        expert_a=sampled_inp_D[2],
                        agent_s=sampled_inp_D[3],
                        agent_t=sampled_inp_D[4],
                        agent_a=sampled_inp_D[5])

                '''
               
                D.train(expert_s=expert_observations,        
                        expert_t=expert_targets,
                        expert_a=expert_actions,
                        agent_s=observations,
                        agent_t=targets,
                        agent_a=actions)
                '''
                


            
#To get rewards we can use a RNN , then we can get the each time unit output to collect the reward function 
            d_rewards = D.get_rewards(agent_s=observations,agent_t=targets, agent_a=actions) #how well our agent performed with respect to the expert 
            d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32) #rewards for each action pair

            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)  #this to calcuate the advantage function in PPO
            gaes = np.array(gaes).astype(dtype=np.float32)
            # gaes = (gaes - gaes.mean()) / gaes.std()
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32) #This is the next value function




            #train policy
            inp = [observations,targets ,actions, gaes, d_rewards, v_preds_next]
            PPO.assign_policy_parameters()  #Assigning policy params means assigning the weights to the default policy nets 
            for epoch in range(100):  #This is to train the Agent (Actor Critic ) from the obtaiend agent performances and already trained discriminator 
                sample_indices = np.random.randint(low=0, high=observations.shape[0],
                                                   size=32)  # indices are in [low, high)

                
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # Here trainign the policy network
                
                PPO.train(state=sampled_inp[0],
                          targets=sampled_inp[1],
                          actions=sampled_inp[2],
                          gaes=sampled_inp[3],
                          rewards=sampled_inp[4],
                          v_preds_next=sampled_inp[5])

            
                
            summary = PPO.get_summary(obs=inp[0],
                                      target=inp[1],  
                                      actions=inp[2],
                                      gaes=inp[3],
                                      rewards=inp[4],
                                      v_preds_next=inp[5])
            

            writer.add_summary(summary, iteration)
        writer.close()


if __name__ == '__main__':
    args = argparser()
    main(args)
