#!python3
import torch
import traceback
import sys
import os

from colors import *
from simulator import Simulator, SnakeSimulator
from visualizer import Visualizer
from gym import Gym
from kevin import Kevin
from maurice import Maurice


"""
This project is about implementing a Q-learning algorithm for a snake game.
Basic architecture:
- Create the model-free agent
- Create a simulator for the snake game
- Create an Environment class to run an agent in a simulator

Q-learning:
- The Agent class will be responsible for implementing the Q-learning algorithm
- We will use a Neural Network to approximate the Q-values (we ain't in medieval times anymore)
- A step() will go as this: *receives state&actions* -> *compute reward and update the policy* -> *queries the NN* -> *returns action*
- Hopefully, that's a good overview.
"""

def print_torch_info():
    print(f"{CYAN}PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(torch.__config__.parallel_info())  # Check if MKL is enabled
    print(f"{RESET}\n")

def get_brain(file=None, type="kevin"):
    state_shape, action_count = SnakeSimulator().get_io_shape()
    if type == "kevin":
        if file:
            return Kevin(state_shape, action_count, load_model=file, lr=0.002, decay=0.998, epsilon=0.35)
        else:
            return Kevin(state_shape, action_count, lr=0.005, decay=0.995)
    elif type == "maurice":
        if file:
            return Maurice(state_shape, action_count, load_model=file, lr=0.003, epsilon=0.25)
        else:
            return Maurice(state_shape, action_count, lr=0.1)
    else:
        raise ValueError(f"Unknown brain type: {type}")

def main(ac:int=0, av:str=None):
    file = None
    action = "train_test"
    # action = "visualize"
    # action = "test"
    # file = r"saves/maurice/maurice_qtable_2025-02-22_21-10-09.pt"
    # file = r"saves/kevin/kevin_nn_2025-02-22_22-03-46.pt"
    try:
        brain = get_brain(file=file, type="kevin")
        gym = Gym(brain, lambda: SnakeSimulator(), 1)
        if action == "train":
            gym.train()
        elif action == "test":
            gym.test()
            gym.test()
            gym.test()
            gym.test(cli_map=True)
        elif action == "train_test":
            gym.train()
            gym.test()
            gym.test()
            gym.test()
            gym.test()
            gym.test()
        elif action == "visualize":
            path = gym.test_record(min_acepted_snake_len=15)
            if path:
                visualizer = Visualizer()
                visualizer.load_game(path)
                visualizer.start()
        else:
            print(f"{RED}Unknown action: {action}")
    except Exception as e:
        print(f"{RED}Exception occured in main branch: {RESET}{e}")
        traceback.print_exc()
        return

if __name__ == '__main__':
    # print_torch_info()
    main(len(sys.argv), sys.argv)
