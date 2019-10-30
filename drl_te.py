from itertools import count
import gym
import math
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=256):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(num_inputs,  hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = self.linear2(x)
        return x


class Critic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=256):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(num_inputs,  hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = self.linear2(x)
        return x


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


def get_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    n_state = env.observation_space.shape[0]
    n_actions = env.action_space.n
    actor_net = Actor(n_state, n_actions, hidden_size=128).to(device)
    critic_net = Critic(n_state, n_actions, hidden_size=128).to(device)
    target_actor_net = Actor(n_state, n_actions, hidden_size=128).to(device)
    target_critic_net = Critic(n_state, n_actions, hidden_size=128).to(device)
    update_target(actor_net, target_actor_net)
    update_target(critic_net, target_critic_net)
    replay_buffer = ReplayBuffer(5000)
    # optimizer = optim.Adam(policy_net.parameters())
    optimizer = optim.RMSprop(policy_net.parameters())
    all_rewards = []
    losses = []
    steps_done = 0

    num_episodes = 500
    for i_episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        total_reward = 0
        for t in count():
            # Select and perform an action
            action = get_action(state)
            next_state, reward, done, _ = env.step(action.item())
            if done:
                next_state = None
            else:
                next_state = torch.FloatTensor(
                    next_state).unsqueeze(0).to(device)

            total_reward += reward
            reward = torch.tensor([reward], device=device)

            # Store the transition in memory
            replay_buffer.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                # episode_durations.append(t + 1)
                # plot_durations()
                all_rewards.append(total_reward)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            print(all_rewards[-10:])
            update_target(policy_net, target_net)
