import argparse
import gym
import numpy as np
from network_models.policy_net import Policy_net
import tensorflow as tf
import pdb


# noinspection PyTypeChecker
def open_file_and_save(file_path, data):
    """
    :param file_path: type==string
    :param data:
    """
    try:
        with open(file_path, 'ab') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')
    except FileNotFoundError:
        with open(file_path, 'wb') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='filename of model to test', default='trained_models/ppo/model.ckpt')
    parser.add_argument('--iteration', default=10, type=int)

    return parser.parse_args()


def main(args):
    env = gym.make('CartPole-v0')
    env.seed(0)
    ob_space = env.observation_space

    
    Policy = Policy_net('policy', env) #initialize the policy network
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.model) #Reastore parameters with the previously trained model
        obs = env.reset() #initial observation

        for iteration in range(args.iteration):  # episode
            observations = []
            actions = []
            run_steps = 0
            while True:
                run_steps += 1
                # prepare to feed placeholder Policy.obs
                obs = np.stack([obs]).astype(dtype=np.float32)   

                act, _ = Policy.act(obs=obs, stochastic=True)
                act = np.asscalar(act) #collect the action

                observations.append(obs)  #collect observations
                actions.append(act) #collect actions

                next_obs, reward, done, info = env.step(act) #get the next returns from the environment afture inputting the act
               

                if done:
                    print(run_steps)
                    obs = env.reset()
                    break
                else:
                    obs = next_obs

            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)
          

            open_file_and_save('trajectory/observations.csv', observations) #saving the expert trajectories  observations
            open_file_and_save('trajectory/actions.csv', actions) #ssaving expert actions in each tractories 


if __name__ == '__main__':
    args = argparser()
    main(args)
