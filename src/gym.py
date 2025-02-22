from simulator import Simulator
import os

from kevin import Kevin
from maurice import Maurice

import threading
import colors
import torch
from torch import Tensor

class Gym:
    def __init__(self, brain, sim_generator:callable, cpu_count:int=1):
        '''
        :param brain: the class that handles the Agents and Neural Network
        :param sim_generator: a callable that returns a Simulator instance
        :param cpu_count: the number of threads to use (default: 1); max: os.cpu_count() for performance
        raises: True
        '''
        self.sim_generator = sim_generator
        if not callable(sim_generator):
            raise ValueError("Simulator generator must be a callable")
        if not isinstance(sim_generator(), Simulator):
            raise ValueError("Simulator generator must return a Simulator instance")

        assert 0 < cpu_count <= (os.cpu_count() if os.cpu_count() is not None else cpu_count),\
            f"CPU count must be greater than 0 and should be less than the available CPU count of the system ({os.cpu_count()} cpu(s) available)"
        self.cpu_count = cpu_count # Hopefully you won't rip a core out while running this
        self.brain = brain

    def run_episode(self, agent, sim, learning_on=True) -> list[tuple[Tensor, int, float, Tensor]]:
        '''
        Will use the given simulator instance and agent instance to run an episode.
        : return: a list of tuples (state:torch.tensor, action:int, reward:float)
        '''
        buff = []
        sim.init_episode()
        state, done = sim.get_state()
        action:int = agent.qna(state)
        reward:float = sim.step(action)
        prev_state = (state, action, reward) # s, a, r
        while True:
            state, done = sim.get_state()
            buff.append((prev_state[0], prev_state[1], prev_state[2], state, done)) # s, a, r, s', done
            if done: break
            action = agent.qna(state)
            reward = sim.step(action)
            prev_state = (state, action, reward)
        return buff

    def _thread_worker(self, agent, sim, results:list, learning_on=True):
        '''
        Wrapper for the run_episode method to be used in a thread
        '''
        buff = self.run_episode(agent, sim)
        results.extend(buff)

    def train(self, epoch=5000, batch_size=64):
        mod = 10
        if epoch >= 400:
            mod = 50
        elif epoch >= 1000:
            mod = 100

        experiences:list[tuple[Tensor, int, float, Tensor]] = list()

        if self.cpu_count > 1:
            sim_pool = [self.sim_generator() for _ in range(self.cpu_count)]
        else:
            sim = self.sim_generator()

        if isinstance(self.brain, Maurice): # Cuz maurice is a Q-table, he don't do certain things
            self.cpu_count = 1 # No need for threads
            self.batch_size = 1 # No need for batches, q-tables have independent updates

        for e in range(epoch):
            while len(experiences) < batch_size * 2:
                if self.cpu_count == 1:
                    experiences.extend(self.run_episode(self.brain, sim))
                else:
                    threads = []
                    for sim_id in range(self.cpu_count):
                        sim = sim_pool[sim_id]
                        thread = threading.Thread(target=self._thread_worker, args=(self.brain, sim, experiences))
                        threads.append(thread)
                        thread.start()
                    for thread in threads:
                        thread.join()
            if e % mod == 0:
                average_reward = sum([exp[2] for exp in experiences]) / len(experiences)
                print(f"Epoch {e} done, average reward: {average_reward}, epsilon: {self.brain.epsilon}")
            self.brain.update(experiences, batch_size=batch_size)
            experiences.clear()
        print(f"Epoch {epoch} done, average reward: {average_reward}, epsilon: {self.brain.epsilon}")
        self.brain.save_weights() # I already have a Gazillion kevins and maurice in my directory


    def test(self, cli_map=False, max_tick=1500):
        sim = self.sim_generator()
        sim.init_episode()

        with torch.no_grad():
            s, _ = sim.get_state()
            a = self.brain.qna(s, learning_on=False)
            r = sim.step(a)
            while True:
                s, done = sim.get_state()
                if done: break
                if cli_map:
                    sim.display_map_cli()
                r += sim.step(self.brain.qna(s, learning_on=False), max_tick=max_tick)
        ticks = sim.ticks
        snake_len = sim.snake_len
        print(f"{colors.CYAN}Simulation ended after {ticks} ticks, with a size of {snake_len}, average reward {r/ticks}{colors.RESET}")
