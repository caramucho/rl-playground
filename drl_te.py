from itertools import count
import gym
import math
import random
import numpy as np
from collections import deque
from collections import namedtuple
from PERMemory import PERMemory

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from multiprocessing_env import SubprocVecEnv

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value


def test_env(vis=False):
    state = env.reset()
    if vis:
        env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis:
            env.render()
        total_reward += reward
    return total_reward


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk


def drl_update(batch_size):
    if len(replay_buffer) < batch_size:
        return
    b_idx, transitions, ISWeights = replay_buffer.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    dists, state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    next_dists, next_state_values[non_final_mask] = target_model(
        non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Update the transition priority
    abs_errors = torch.abs(state_action_values -
                           expected_state_action_values.unsqueeze(1)).cpu().detach().numpy()
    replay_buffer.batch_update(b_idx, abs_errors)

    log_prob = dist.log_prob(action)
    entropy += dist.entropy().mean()

    log_probs.append(log_prob)
    values.append(value)
    rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
    masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

    next_state = torch.FloatTensor(next_state).to(device)
    _, next_value = model(next_state)
    returns = compute_returns(next_value, rewards, masks)
    log_probs = torch.cat(log_probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)
    # print(log_probs.shape, returns.shape, values.shape)

    advantage = returns - values

    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
    # print(actor_loss.item(), 0.5*critic_loss.item(), 0.001*entropy.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def optimize_model(batch_size):

    if len(replay_buffer) < batch_size:
        return
    b_idx, transitions, ISWeights = replay_buffer.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Update the transition priority
    abs_errors = torch.abs(state_action_values -
                           expected_state_action_values.unsqueeze(1)).cpu().detach().numpy()
    replay_buffer.batch_update(b_idx, abs_errors)

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")

    """<h2>Create Environments</h2>"""

    num_envs = 16
    # env_name = "CartPole-v0"
    env_name = "MountainCar-v0"

    # envs = [make_env() for i in range(num_envs)]
    # envs = SubprocVecEnv(envs)

    # env = gym.make(env_name)
    envs = gym.make(env_name)

    num_inputs = envs.observation_space.shape[0]
    num_outputs = envs.action_space.n

    # Hyper params:
    hidden_size = 256
    lr = 3e-4
    num_steps = 10

    model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
    target_model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
    update_target(model, target_model)

    optimizer = optim.Adam(model.parameters())

    # create PER buffer
    replay_buffer = PERMemory(5000)

    max_frames = 20000
    frame_idx = 0
    test_rewards = []

    state = envs.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)

    while frame_idx < max_frames:
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0

        for _ in range(num_steps):
            # state = torch.FloatTensor(state).to(device)
            dist, value = model(state)

            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy()[0])
            if done:
                next_state = None
            else:
                next_state = torch.FloatTensor(
                    next_state).unsqueeze(0).to(device)

            # Store the transition in memory
            replay_buffer.push(state, action, next_state, reward)

            state = next_state
            frame_idx += 1

            drl_update(3)

            if frame_idx % 1000 == 0:
                test_rewards.append(np.mean([test_env() for _ in range(10)]))
                # plot(frame_idx, test_rewards)
                print(test_rewards[-100:])
