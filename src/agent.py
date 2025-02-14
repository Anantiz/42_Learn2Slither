import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.nn import functional as F

'''
Jargon notes:

    'fc' = fully conected
    'bn' = batch normalization
    'drop' = dropout layer
    'emb' = embeding
    'logits' = Pre softmax scores
    'probs / preds' = probabilities / predictions
    'chkpt' = checkpoint (saving model etc)
'''

class Dqn(nn.Module):
    def __init__(self, input_shape: tuple, action_dim: int):
        super().__init__()
        if not isinstance(input_shape, tuple):
            raise ValueError("Input shape must be a tuple")
        if len(input_shape) != 2:
            raise ValueError("Input shape must be 2D")

        print(f"Creating a neural network with input shape: {input_shape} and output count: {action_dim}")
        self.input_shape = input_shape
        input_size = input_shape[0] * input_shape[1]
        hidden1 = int(input_size * 4)
        hidden2 = int(hidden1 * 0.2)

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            nn.Linear(hidden2, action_dim)
        )
        self.out_features = self.network[-1].out_features
        print(self.network)
        print("Neural network created")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        '''
        :param s: a tensor, [4,4] (4 features from 4 directions), no batch dimension
        :return: a tensor, [4] (4 actions)
        '''
        if s.shape != self.input_shape:
            raise ValueError(f"Expected input shape {self.input_shape}, but got {s.shape}")

        # Flatten input tensor
        s = s.view(-1)  # Equivalent to s.reshape(input_size), ensuring a 1D tensor
        # print(f"S: {s}\n S.shape: {s.shape}")
        # exit()
        # Pass through the network
        return self.network(s)

class Agent:
    def __init__(self, gamma=0.95, lr=0.003, epsilon=1.0, min_epsilon=0.01, decay=0.995):
        """ Initialize the hyper parameters """
        ### Hyperparameters
        self.lr = lr            # Learning rate
        self.gamma = gamma      # Discount factor
        self.decay = decay      # Decay rate for epsilon
        self.epsilon = epsilon  # Starting Exploration rate
        self.min_epsilon = min_epsilon
        self.dqn = None
        self.action_list_size = 0 # Check if the action list size somehow changes
        self.action_list = []
        self.learning_on = True

    def init_neural_network(self, input_shape:tuple, output_shape:tuple):
        """ Initialize the neural network from scratch """
        self.dqn = Dqn(input_shape, output_shape)
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.lr)
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 1.0)  # Clip gradients


    def load_neural_network(self, path):
        """Load the neural network from a file"""
        pass

    def save_neural_network(self, path="model_weights.pth"):
        """Save the neural network to a file"""
        torch.save(self.dqn.state_dict(), path)

    def init_episode(self, initial_state: torch.tensor, actions: list[int], learning_on=True):
        """ Initialize the agent for an episode """
        self.action_list_size = len(actions)
        if self.action_list_size != self.dqn.out_features:
            raise ValueError("Action list size does not match the output of the neural network; \
                The curent neural network is not compatible with the curent simulation")
        self.action_list = actions
        self.learning_on = learning_on
        # `initial_state` is not used in this agent
        self.last_state = None
        self.last_action = 0
        self.buffer = deque(maxlen=1000)

    def update_policy(self):
        """
        Update the policy with the buffer
        buffer of shape: [s, a, r, s']
        """
        # print(f"Updating policy with {len(self.buffer)} samples")
        # The terminal value will be [last_state_before_death, fatal_action, reward_of_death, None]
        for s, a, r, s_prime in self.buffer:
            q_value = self.dqn(s)[a].unsqueeze(0)
            with torch.no_grad():
                q_target = torch.tensor([r], dtype=torch.float32) if s_prime is None else torch.tensor([r + self.gamma * torch.max(self.dqn(s_prime))], dtype=torch.float32)
                # q_target = torch.tensor([r], dtype=torch.float32) if s_prime is None else r + self.gamma * torch.max(self.dqn(s_prime))
            loss:torch.tensor = F.mse_loss(q_value, q_target)
            # Update
            self.dqn.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay) # Decay exploration the more we learn

        self.buffer.clear()

    def pick_action(self, state:torch.tensor) -> int:
        """ Pick an action given the state and list of actions """
        if state is None:
            return None
        if torch.rand(1).item() < self.epsilon:
            return self.action_list[torch.randint(0, self.action_list_size, (1,)).item()]
        q_values = self.dqn(state)
        return torch.argmax(q_values)

    def step(self, state:torch.tensor, reward:float) -> int:
        """
        :param state: a tensor
        :param reward: a float
        :return: action with highest Q value
        Queries the argent to take an action given the current state and possible actions.
        The policy is independent of the query origin. (Model-free)
        """
        if self.last_state is not None: # Because at the first iteration there is no last state
            self.buffer.append((self.last_state, self.last_action, reward, state))

        if state is None: # U dead lol
            if self.learning_on and len(self.buffer) > 30:
                self.update_policy()
            return None
        self.last_state = state
        self.last_action = self.action_list[self.pick_action(state)]
        return self.last_action
