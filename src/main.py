#!python3
import sys
from colors import *
from simulator import Simulator, SnakeSimulator
from agent import Agent
from environment import Environment

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

def main(ac: int = 0, av: str = None):
    try:
        agent = Agent()
        simulator = SnakeSimulator()
        state_shape, action_shape = simulator.get_io_shape()
        agent.init_neural_network(state_shape, action_shape)
        environment = Environment(agent, simulator)
        environment.run()
    except Exception as e:
        print(f"{RED}Exception occured in main branch: {RESET}{e}")
        return

if __name__ == '__main__':
    main(len(sys.argv), sys.argv)
