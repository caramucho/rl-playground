import math
from collections import deque
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple


class Bit_Flipping(object):
    def __init__(self, num_bits):
        self.num_bits = num_bits

    def reset(self):
        self.done = False
        self.num_steps = 0
        self.state = np.random.randint(2, size=self.num_bits)
        self.target = np.random.randint(2, size=self.num_bits)
        return np.copy(self.state), self.target

    def step(self, action):
        if self.done:
            raise ValueError

        self.state[action] = 1 - self.state[action]

        if self.num_steps > self.num_bits + 1:
            self.done = True
        self.num_steps += 1

        if np.sum(self.state == self.target) == self.num_bits:
            self.done = True
            return np.copy(self.state), 0, self.done, {}
        else:
            return np.copy(self.state), -1, self.done, {}


class Model(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=256):
        super(Model, self).__init__()

        self.linear1 = nn.Linear(num_inputs,  hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, state, goal):
        x = torch.cat([state, goal], 1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# class ReplayBuffer(object):
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)

#     def push(self, state, action, reward, next_state, done, goal):
#         self.buffer.append((state, action, reward, next_state, done, goal))

#     def sample(self, batch_size):
#         state, action, reward, next_state, done, goal = zip(
#             *random.sample(self.buffer, batch_size))
#         return np.stack(state), action, reward, np.stack(next_state), done, np.stack(goal)

#     def __len__(self):
#         return len(self.buffer)


Transition = namedtuple('Transition',
                        ('state', 'action',  'reward', 'next_state', 'done', 'goal'))


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity
        self.counter = 0

    def add(self, p, data):
        self.counter += 1
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root

    def __len__(self):
        return self.counter if self.counter < self.capacity else self.capacity


class PERMemory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    xi = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def push(self, *args):
        transition = Transition(*args)
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx = []
        ISWeights = []
        b_memory = []
        pri_seg = self.tree.total_p / n       # priority segment

        # for later calculate ISweight
        # min_prob = np.min(
        #     self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
        buffer_size = len(self.tree)
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights.append(np.power(prob * buffer_size, -self.beta))
            b_idx.append(idx)
            b_memory.append(data)
        # normalize important sampling weights
        ISWeights /= max(ISWeights)
        # update beta1
        self.beta = np.min(
            [1., self.beta + self.beta_increment_per_sampling])  # max = 1
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.xi  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def __len__(self):
        return len(self.tree)


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def compute_td_error(batch_size):
    if replay_buffer.__len__() < batch_size:
        return None
    b_idx, transitions, ISWeights = replay_buffer.sample(
        batch_size)
    batch = Transition(*zip(*transitions))
    state = torch.FloatTensor(batch.state).to(device)
    reward = torch.FloatTensor(batch.reward).unsqueeze(1).to(device)
    action = torch.LongTensor(batch.action).unsqueeze(1).to(device)
    next_state = torch.FloatTensor(batch.next_state).to(device)
    goal = torch.FloatTensor(batch.goal).to(device)
    mask = torch.FloatTensor(1 - np.float32(batch.done)
                             ).unsqueeze(1).to(device)

    q_values = model(state, goal)
    q_value = q_values.gather(1, action)

    next_q_values = target_model(next_state, goal)
    target_action = next_q_values.max(1)[1].unsqueeze(1)
    next_q_value = target_model(next_state, goal).gather(1, target_action)

    expected_q_value = reward + 0.99 * next_q_value * mask

    # loss = (q_value - expected_q_value.detach()).pow(2).mean()
    loss = F.smooth_l1_loss(q_value,
                            expected_q_value.detach())
    # Update the transition priority
    abs_errors = torch.abs(q_value -
                           expected_q_value).cpu().detach().numpy()
    replay_buffer.batch_update(b_idx, abs_errors)

    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss


def get_action(model, state, goal, epsilon=0.3):
    # if random.random() < epsilon:
    #     return random.randrange(env.num_bits)

    # q_value = model(state, goal)
    # best_action = torch.argmax(q_value, dim=1).item()
    # return best_action
    EPS_START = 0.9
    EPS_END = 0.2
    EPS_DECAY = 50000
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
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            goal = torch.FloatTensor(goal).unsqueeze(0).to(device)
            return torch.argmax(model(state, goal), dim=1).item()
    else:
        return torch.tensor([[random.randrange(num_bits)]], device=device, dtype=torch.long)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


if __name__ == "__main__":
    num_bits = 11
    env = Bit_Flipping(num_bits)

    model = Model(2 * num_bits, num_bits).to(device)
    target_model = Model(2 * num_bits, num_bits).to(device)
    update_target(model, target_model)

    replay_buffer = PERMemory(100000)
    # hyperparams:
    batch_size = 64
    new_goals = 5
    max_frames = 200000
    steps_done = 0
    optimizer = optim.Adam(model.parameters())
    frame_idx = 0
    all_rewards = []
    losses = []

    while frame_idx < max_frames:
        state, goal = env.reset()
        done = False
        episode = []
        total_reward = 0
        while not done:
            action = get_action(model, state, goal)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done, goal)
            state = next_state
            total_reward += reward
            frame_idx += 1

        loss = compute_td_error(batch_size)
        if loss is not None:
            losses.append(loss.item())
        all_rewards.append(total_reward)

        if frame_idx % 100 == 0:
            # plot(frame_idx, [np.mean(all_rewards[i:i+100])
            # for i in range(0, len(all_rewards), 100)], losses)
            print(all_rewards[-20:])
            update_target(model, target_model)
