import gym
import gym_bandits
import math
import numpy as np


def thompson_sampling(action_space):
    global steps
    EPS_START = 0.9
    EPS_END = 0.1
    EPS_DECAY = 100
    # global steps_done
    epsilon = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps / EPS_DECAY)
    # steps_done += 1
    # global steps
    steps += 1
    probs = np.zeros(action_space) + epsilon/(action_space - 1)
    best_action = np.argmax(qtable)
    probs[best_action] = 1-epsilon
    np.random.seed()
    return np.random.choice(range(action_space), p=probs)


def updateq(reward, action):
    qtable[action] = qtable[action] + 1/(steps+1)*(reward-qtable[action])


if __name__ == "__main__":
    # rerun this part of the code if you would like to "reset" or reinitialize your bandit environement
    np.random.seed(42)
    # gaussian distribution is just another name for "normal distribution" or bell curve (so many different names for the same thing!)
    env = gym.make('BanditTenArmedGaussian-v0')
    # n_state = env.observation_space.shape
    n_actions = env.action_space.n
    all_rewards = []
    steps = 0
    qtable = np.zeros(n_actions)
    for i_episode in range(10):
        # state = env.reset()
        total_reward = 0
        for i in range(30):
            # sampling the "action" array which in this case only contains 10 "options" because there is 10 bandits
            # action = get_action(state)
            # action = env.action_space.sample()
            action = epsilon_greedy(n_actions)

            # here we taking the next "step" in our environment by taking in our action variable randomly selected above
            state, reward, done, info = env.step(action)
            updateq(reward, action)
            total_reward += reward
        all_rewards.append(total_reward)
    print(all_rewards)
    # print(qtable)
    env.close()
