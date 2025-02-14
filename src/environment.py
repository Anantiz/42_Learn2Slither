from simulator import Simulator, SnakeSimulator
from agent import Agent
from colors import *
import torch

class Environment:
    def __init__(self, agent: Agent = None, simulator: Simulator = None):
        self.agent = agent
        self.simulator = simulator

    def set_agent(self, agent: Agent):
        if not isinstance(agent, Agent):
            raise ValueError("Agent must inherit from class Agent")
        self.agent = agent

    def set_simulator(self, simulator: Simulator):
        if not isinstance(simulator, Simulator):
            raise ValueError("Simulator must inherit from class Simulator")
        self.simulator = simulator

    def run(self, epoch=10_000):
        '''
        Will execute one round of:
        - init_episode() from the simulator
        - init_episode() from the agent
        - Complete the episode
        '''
        try:
            if self.agent is None:
                raise ValueError("Agent not set")
            if self.simulator is None:
                raise ValueError("Simulator not set")

            # Initialize the environment
            state, action, actions = None, 0, []
            # Actions must be always a list of fixed size, however,
            # the values can be different in special cases, not all agent suport this
            initial_state, actions = self.simulator.init_episode()
            self.agent.init_episode(initial_state, actions)

            # Run the environment
            while epoch:
                state, reward = self.simulator.step(action)
                action = self.agent.step(state, reward)
                if state is None: # End game signal, u dead lol
                    epoch -= 1
                    # print(f"remaining_epoch={epoch}")
                    initial_state, actions = self.simulator.init_episode()
            self.agent.save_neural_network()
        except Exception as e:
            print(f"{RED}Environment couldn't run(): {RESET}{e}")
            return
