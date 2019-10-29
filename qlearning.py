import numpy as np
import pandas as pd
import random


class Game:
    rewards = None
    positionCol = None
    positionRow = None

    def __init__(self, startCol=1, startRow=1):
        self.rewards = pd.DataFrame({1: [0, 1, 2, 3, 4], 2: [1, 2, 3, 4, 5], 3: [
                                    2, 3, 4, 5, 6], 4: [3, 4, 5, 6, 7], 5: [4, 5, 6, 7, 8]}, index={1, 2, 3, 4, 5})
        self.positionCol = startCol
        self.positionRow = startRow

    def move(self, direction):
        reward = 0
        end = False
        if direction == 'Up':
            self.positionRow -= 1
        elif direction == 'Down':
            self.positionRow += 1
        elif direction == 'Left':
            self.positionCol -= 1
        else:
            self.positionCol += 1

        # check if we lost
        if self.positionRow < 1 or self.positionRow > 5 or self.positionCol < 1 or self.positionCol > 5:
            end = True
            reward = -1000
        # check if we have reached the end
        elif self.positionCol == 5 and self.positionRow == 5:
            end = True
            reward = self.rewards[self.positionCol][self.positionRow]
        else:
            end = False
            reward = self.rewards[self.positionCol][self.positionRow]

        # return reward and end of game indicator
        return (reward, end)


def updateq(state, next_state, action, reward):
    alpha = 0.2
    discount_factor = 0.9
    qvalues = qtable[next_state]
    best_action_q = max(qvalues)
    qtable[state] = (1-alpha)*qtable[state] + alpha * \
        (reward + discount_factor * best_action_q)


def epsilon_greedy(state):
    epsilon = 0.3
    probs = np.zeros(4) + epsilon/3
    qvalues = qtable[state]
    best_action = np.argmax(qvalues)
    probs[best_action] = 1-epsilon
    print(probs)
    return np.random.choice([0, 1, 2, 3], p=probs)


if __name__ == "__main__":
    passqtable = defaultdict(lambda: np.zeros(4))
    for i in range(1000):
    end = False
    env = Game()
    state = (env.positionCol, env.positionRow)
    while not end:
        actionid = epsilon_greedy(state)
        actionspace = ["Up", "Down", "Left", "Right"]
        action = actionspace[actionid]
        reward, end = env.move(action)
        next_state = (env.positionCol, env.positionRow)
        updateq(state, next_state, actionid, reward)
        state = next_state
