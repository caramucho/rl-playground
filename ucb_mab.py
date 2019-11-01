import gym
import gym_bandits
import math
import numpy as np


def ucb(action_space):
    global steps
    if steps < action_space:
        return steps
    ucb = [v + np.sqrt(2*np.log(steps) / count) for (v, count) in avg]
    best_action = np.argmax(ucb)
    return best_action


def update_avg(reward, action):
    n = avg[action][1]
    avg[action][0] = avg[action][0] + 1/(n+1)*(reward-avg[action][0])
    avg[action][1] += 1


if __name__ == "__main__":
    # rerun this part of the code if you would like to "reset" or reinitialize your bandit environement
    np.random.seed(42)
    # gaussian distribution is just another name for "normal distribution" or bell curve (so many different names for the same thing!)
    env = gym.make('BanditTenArmedGaussian-v0')
    # n_state = env.observation_space.shape
    n_actions = env.action_space.n
    all_rewards = []
    steps = 0
    avg = np.zeros((n_actions, 2))
    for i_episode in range(20):
        # state = env.reset()
        total_reward = 0
        for i in range(200):
            # sampling the "action" array which in this case only contains 10 "options" because there is 10 bandits
            # action = get_action(state)
            # action = env.action_space.sample()
            action = ucb(n_actions)
            steps += 1

            # here we taking the next "step" in our environment by taking in our action variable randomly selected above
            state, reward, done, info = env.step(action)
            update_avg(reward, action)
            total_reward += reward
        all_rewards.append(total_reward)
    print(all_rewards)
    # print(qtable)
    env.close()
