import random
import os
from torch import Tensor
from datetime import datetime

class Maurice:

    def __init__(self, state_shape, action_count, lr=0.05, gamma=0.95, epsilon=1.0, min_epsilon=0.015, decay=0.995, load_model=None):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.action_dim= action_count
        self.qtable = None
        if load_model:
            self.load_weights(load_model)
        else:
            self.init_q_table(state_shape, action_count)

    def init_q_table(self, state_shape:tuple, action_dim:int):
        '''
        Will initialize the Q-table:
        State representation:
        [red-apple/green-apple/obstacle] for all 4 state at once
        '''
        dimmensions = [3, 4] # Cuz we down-size the state matrix we absolutely ignore state_shape *thug life*
        self.qtable = [[random.uniform(-1, 1) for _ in range(self.action_dim)] for _ in range(2**(dimmensions[0] * dimmensions[1]))]
        print(f"Initialized Q-table with width: {2**(dimmensions[0] * dimmensions[1])} and action count: {self.action_dim}")

    def save_weights(self, path:str=None):
        if not path:
            path = f"saves/maurice/maurice_qtable_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"
            if not os.path.exists("saves/maurice"):
                os.makedirs("saves/maurice")
        with open(path, "w") as f:
            for row in self.qtable:
                f.write(" ".join([str(x) for x in row]) + "\n")
        print(f"Saved Q-table to: {path}")

    def load_weights(self, path:str):
        with open(path, "r") as f:
            self.qtable = [[float(x) for x in line.strip().split()] for line in f.readlines()]
        print(f"Loaded Q-table from: {path}")

    def index_state(self, state:Tensor) -> int:
        '''
        Will return the index of the state in the Q-table
        '''
        index = 0
        for i in range(len(state)):
            red = state[i][0] if state[i][0] != 0 else 99999
            green = state[i][1] if state[i][1] != 0 else 99999
            obstacle = state[i][2] if state[i][3] == 0 else state[i][3] if state[i][2] == 0 else min(state[i][2], state[i][3])
            if obstacle == 0: obstacle = 99999
            # make it 2*i cuz a field is 2 bits
            if red < green and red < obstacle:
                index |= 0b01 << 3*i
            elif green < red and green < obstacle:
                index |= 0b10 << 3*i
            elif obstacle == 1:
                index |= 0b11 << 3*i
            else:
                index |= 0b100 << 3*i
        return index

    def update_q_table(self, state:Tensor, action:int, reward:float, next_state:Tensor, done):
        '''
        Will update the Q-table with the given experience
        '''
        index = self.index_state(state)
        q = self.qtable[index][action]

        if done: # Terminal state, indeed terribly slow way to check, but i ain't got that faith in me to edit 15 more lines today
            q_next = 0
        else:
            next_index = self.index_state(next_state)
            q_next = max(self.qtable[next_index])

        self.qtable[index][action] = q + self.lr * (reward + self.gamma * q_next - q)

    def update(self, experiences, batch_size=None):
        ''' batch_size is unused, there ain't no batch in Q-tables '''
        for s, a, r, sp, done in experiences:
            self.update_q_table(s, a, r, sp, done)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

    def qna(self, state:Tensor, learning_on=True) -> int:
        '''
        Will return the best action for the given state
        '''
        if learning_on and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        index = self.index_state(state)
        return self.qtable[index].index(max(self.qtable[index]))
