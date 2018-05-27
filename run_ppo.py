#!/usr/bin/python3          #This is basically the training ...
import argparse
import gym
import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from algo.ppo import PPOTrain
import pdb

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/ppo')
    parser.add_argument('--savedir', help='save directory', default='trained_models/ppo')
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--iteration', default=int(1e4), type=int)
    return parser.parse_args()


def main(args):
    env = gym.make('CartPole-v0')
    env.seed(0)
    ob_space = env.observation_space  #This is the environment for the gym to observe 

    Policy = Policy_net('policy', env) #take the environments  #this is normal policy class
    Old_Policy = Policy_net('old_policy', env)  #this is for the old policy  
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma)  #this is for training 
    

    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir, sess.graph)
        sess.run(tf.global_variables_initializer()) #Here all the variabls get initialized 
        obs = env.reset()  # [position of cart, velocity of cart, angle of pole, rotation rate of pole] Initial observation   
     
        reward = 0
        success_num = 0

        for iteration in range(args.iteration):
            observations = [] #to store observations 
            actions = []
            v_preds = []
            rewards = []
            episode_length = 0
         
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length  #Starting to run the 
                episode_length += 1 #episode length is something dynamic 
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                act, v_pred = Policy.act(obs=obs, stochastic=True) #get the action and value prediction (actor and critic network output)

                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)

                observations.append(obs)
                actions.append(act)
                v_preds.append(v_pred)
                rewards.append(reward)

                next_obs, reward, done, info = env.step(act) #get the observation from the environments 

                #The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center. That is done
               

                if done: #This is a termination stage 
            #this has all the next state eliements of the episode inputs 
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value  #after the terminal stage  there shouldn;t be a value function 
                    obs = env.reset()
                    reward = -1
                    break
                else:       #here your break the episode 
                    obs = next_obs   #if the system do not get terminated it will run for ever      
            #After a one episode get  terminated
               
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=episode_length)])    #From this we can learn how long the episode went
                               , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))]) #
                               , iteration)

            if sum(rewards) >= 195:
                success_num += 1
                if success_num >= 100:
                    saver.save(sess, args.savedir+'/model.ckpt')
                    print('Clear!! Model saved.')           #this is like after this much sucessfull attempts we are confident about the model 
                    break
            else:
                success_num = 0

            gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next) #this is the advantage function 
            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape)) #observations from the current policy
            actions = np.array(actions).astype(dtype=np.int32) #actions taken from current policy
            gaes = np.array(gaes).astype(dtype=np.float32) #generalized advantage enstimation 
            gaes = (gaes - gaes.mean()) / gaes.std() #Normalize it 
            rewards = np.array(rewards).astype(dtype=np.float32) #Extracted rewrds 
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            PPO.assign_policy_parameters() #before updating the new policy we assign current policy parameters to old policy

            inp = [observations, actions, gaes, rewards, v_preds_next]

            # train
            for epoch in range(6): #starting the optimization
                # sample indices from [low, high)
                sample_indices = np.random.randint(low=0, high=observations.shape[0], size=32)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data  Randomly take one sample from the training data
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          gaes=sampled_inp[2],
                          rewards=sampled_inp[3],
                          v_preds_next=sampled_inp[4])

            summary = PPO.get_summary(obs=inp[0],
                                      actions=inp[1],
                                      gaes=inp[2],
                                      rewards=inp[3],
                                      v_preds_next=inp[4])

            writer.add_summary(summary, iteration)
        writer.close()


if __name__ == '__main__':
    args = argparser()

    main(args)
