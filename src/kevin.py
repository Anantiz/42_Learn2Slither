import torch
from torch import Tensor, nn
import datetime
from numpy.random import randint

import random
import os

def sampler_stochastic(experiences: list, batch_size) -> list[list]:
    '''
    :returns: a stochastic list of at-most batch_size lists of experiences (if not enough samples, the last list will be smaller)
    '''
    if not isinstance(experiences, list):
        raise ValueError("Experiences must be a list")
    if not isinstance(batch_size, int):
        raise ValueError("Batch size must be an integer")
    size = len(experiences)
    if not (0 < batch_size <= size):
        raise ValueError("Batch size must be greater than 0 and less than or equal to the length of experiences")

    random.shuffle(experiences)  # Shuffle the experiences to ensure randomness
    return [experiences[i:i + batch_size] for i in range(0, size, batch_size)]


class Dqn(nn.Module):

    def __init__(self, input_shape: tuple, action_dim: int, skip_init=False):
        '''
        :param input_shape: the shape of the input tensor
        :param action_dim: the number of actions the agent can take
        :param skip_init: if True, the model will not be initialized, useful for loading a model from a file
        '''
        super().__init__()
        if not isinstance(input_shape, tuple):
            raise ValueError("Input shape must be a tuple")
        if not (0 < len(input_shape) < 3):
            raise ValueError("Input shape must be a first or second order tensor")

        if not skip_init:
            print(f"Creating a neural network with input shape: {input_shape} and output count: {action_dim}")
        self.input_shape = input_shape
        input_size = input_shape[0] * input_shape[1]
        scale = 32
        self.fc1 = nn.Linear(input_size, 1 * scale)
        self.fc2 = nn.Linear(1 * scale, 2 * scale)
        self.fc3 = nn.Linear(2 * scale, 4 * scale)
        self.fc4 = nn.Linear(4 * scale, action_dim)

    def forward(self, x:Tensor) -> Tensor:
        ''' x is a tensor, yo '''
        x = x.view(-1, self.input_shape[0] * self.input_shape[1])
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Kevin:
    '''
    Naming things is hard, Kevin does the following:
    - Manages a neural network
    - Manages updating the neural network
    - Can compute the Q-values for you and return the best action
    '''
    def __init__(self, input_shape: tuple, action_dim: int, load_model:str=None, lr=0.003, gamma=0.95, epsilon=1.0, min_epsilon=0.01, decay=0.995, target_update_freq=50):
        ''' Hi I'm Kevin ! '''
        if load_model is not None:
            self.dqn = torch.load(load_model)
            print(f"Loaded neural network from: {load_model}")
        else:
            self.dqn = Dqn(input_shape, action_dim)

        self.input_shape = input_shape
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.lr)

        self.dqn_target = Dqn(input_shape, action_dim, skip_init=True)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.target_update_freq = target_update_freq
        self.target_update_counter = 0

    def save_weights(self, path:str=None):
        if not path:
            path = f"saves/kevin/kevin_nn_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"
            if not os.path.exists("saves/kevin"):
                os.makedirs("saves/kevin")
        torch.save(self.dqn, path)
        print(f"Saved Kevin brain as: {path}")

    def share_memory(self):
        self.dqn.share_memory()
        self.dqn_target.share_memory()

    def update(self, experiences:list[tuple[Tensor, int, float, Tensor]], batch_size=32):
        """
        Update the policy using the given experiences.

        Args:
            experiences (list[tuple[Tensor, int, float, Tensor]]): A list of experiences, where each experience is a tuple containing
                a state (Tensor), an action (int), and a reward (float), state_prime (Tensor).
            batch_size (int, optional): The size of each batch for stochastic sampling. Default is 32.

        Returns:
            None
        """
        def target_update():
            ''' Soft update the target network '''
            self.target_update_counter += 1
            if self.target_update_counter % self.target_update_freq == 0:
                self.dqn_target.load_state_dict(self.dqn.state_dict())

        batches = sampler_stochastic(experiences, batch_size)
        for batch in batches:
            if len(batch) == 0: # What ?
                continue
            states, actions, rewards, next_states, done_mask = [], [], [], [], []
            for item in batch:
                states.append(item[0])
                actions.append(item[1])
                rewards.append(item[2])
                next_states.append(item[3])
                done_mask.append(item[4])

            states = torch.stack(states)
            actions = torch.tensor(actions)
            rewards = torch.tensor(rewards)
            next_states = torch.stack(next_states)

            q_values = self.dqn(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_values = self.dqn_target(next_states)
                next_q_values = next_q_values.max(1)[0].detach()

            target = rewards + self.gamma * next_q_values * Tensor(done_mask)
            loss = torch.nn.functional.mse_loss(q_values, target)

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.dqn.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay) # Decay epsilon as the nn learns
        target_update()

    def qna(self, state:Tensor, learning_on=True) -> int:
        ''' Q-values and action, I'm so funny bruh '''
        if learning_on:
            if torch.rand(1).item() < self.epsilon:
                return randint(0, self.action_dim)
        q_values = self.dqn(state)
        return torch.argmax(q_values).item()
