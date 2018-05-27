#!/usr/bin/python3
import argparse
import gym
import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from network_models.discriminator import Discriminator
from algo.ppo import PPOTrain

import pdb


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/gail')
    parser.add_argument('--savedir', help='save directory', default='trained_models/gail')
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--iteration', default=int(1e4))
    return parser.parse_args()


def main(args):
    env = gym.make('CartPole-v0')  #Get the environment to work with
    env.seed(0)
    ob_space = env.observation_space
 
    Policy = Policy_net('policy', env) #buiding the actor critic graph / object
    Old_Policy = Policy_net('old_policy', env) #same thing as the other PPO

    

    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma) #gradiet updatror object or the graph
    D = Discriminator(env) #discriminator of the Gan Kind of thing

    expert_observations = np.genfromtxt('trajectory/observations.csv')  #load expert demnetrations 
    expert_actions = np.genfromtxt('trajectory/actions.csv', dtype=np.int32) 


    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir, sess.graph)
        sess.run(tf.global_variables_initializer()) #here already variables get intialized both old policy and new policy net

        obs = env.reset()
        reward = 0  # do NOT use rewards to update policy
        success_num = 0
        

        for iteration in range(args.iteration):#Here start the adversial training
            observations = []
            actions = []
            rewards = []
            v_preds = []
            run_policy_steps = 0

            while True:  #Here what is happenning is , this again samples  trajectories from untrain agent
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs #Initial observation

                ''' #for my LSTM generate
                prev_ob=observations.append(obs)
                if len(prev_ob)<=3:
                    obs=prev_ob
                else:  #get last three states 
                    obs=prev_ob[len(prev_ob)-3:len(prev_ob)]

                '''
                act, v_pred = Policy.act(obs=obs, stochastic=True) # Agents action and values 
                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)

                observations.append(obs)  #save the set of observations 
                actions.append(act) #save the set of actions 
                rewards.append(reward) #save the set of rewards
                v_preds.append(v_pred) 

                next_obs, reward, done, info = env.step(act)  #get the next observation and reward acording to the observation
                
                
                
                if done:  #run one episode till the termination
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value #this list use to update the parameters of the calue net
                    obs = env.reset()
                    reward = -1
                    break  #with tihs vreak all obsercation ,action and other lists get empty 
                else:
                    obs = next_obs

                

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                               , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)


            if sum(rewards) >= 195: #check weather the agent has learned everything 
                success_num += 1
                if success_num >= 100:
                    saver.save(sess, args.savedir + '/model.ckpt')
                    print('Clear!! Model saved.')
                    break
            else:
                success_num = 0



            # convert list to numpy array for feeding tf.placeholder

            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape)) #collect observations 
            actions = np.array(actions).astype(dtype=np.int32) #collect the actions 

            

            # train discriminator  #Here comes the Discriminator !!        
            for i in range(2):
                D.train(expert_s=expert_observations,        #It's ok to use a dynamic RNN since my case is not fully observable
                        expert_a=expert_actions,
                        agent_s=observations,
                        agent_a=actions)
                                
            # output of this discriminator is reward
#To get rewards we can use a RNN , then we can get the each time unit output to collect the reward function 
            d_rewards = D.get_rewards(agent_s=observations, agent_a=actions) #how well our agent performed with respect to the expert 
            d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32) #rewards for each action pair
           
            
           

            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)  #this to calcuate the advantage function in PPO
            gaes = np.array(gaes).astype(dtype=np.float32)
            # gaes = (gaes - gaes.mean()) / gaes.std()
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32) #This is the next value function
             

            # train policy
            inp = [observations, actions, gaes, d_rewards, v_preds_next]
            PPO.assign_policy_parameters()  #Assigning policy params means assigning the weights to the default policy nets 
            for epoch in range(6):  #This is to train the Agent (Actor Critic ) from the obtaiend agent performances and already trained discriminator 
                sample_indices = np.random.randint(low=0, high=observations.shape[0],
                                                   size=32)  # indices are in [low, high)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # Here trainign the policy network
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
