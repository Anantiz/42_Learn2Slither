import os
import torch
import datetime
import json
from torch import Tensor
from multiprocessing import Process, Manager

import colors
from kevin import Kevin
from maurice import Maurice
from simulator import Simulator


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

    def run_episode(self, agent, sim) -> list[tuple[Tensor, int, float, Tensor]]:
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

    def _parallel_worker(self, agent, sim, requested_experiences_count, shared_experiences, lock):
        '''
        Wrapper for the run_episode method to be used in a thread
        '''
        buff:list = []
        while len(buff) < requested_experiences_count:
            buff.extend(self.run_episode(agent, sim))
        with lock:
            shared_experiences.extend(buff)

    def _parallel_manager(self, sim_pool, batch_size) -> list[tuple[Tensor, int, float, Tensor]]:
        with Manager() as manager:
            shared_experiences = manager.list()  # Shared memory list
            lock = manager.Lock()  # Synchronization lock

            processes = []
            for sim_id in range(self.cpu_count):
                p = Process(target=self._parallel_worker, args=(self.brain, sim_pool[sim_id], batch_size, shared_experiences, lock))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
            return list(shared_experiences)

    def train(self, epoch=1000, batch_size=256):
        mod = 10
        if epoch >= 500:
            mod = 50
        if epoch >= 1000:
            mod = 100
        if isinstance(self.brain, Maurice): # Cuz maurice is a Q-table, he don't do certain things
            self.cpu_count = 1 # No need for threads
            self.batch_size = 8 # No need for batches, q-tables have independent updates

        print(f"{colors.GREEN}Training for {epoch} epochs, with a batch size of {batch_size}, and {self.cpu_count} workers{colors.RESET}")
        if self.cpu_count > 1:
            sim_pool = [self.sim_generator() for _ in range(self.cpu_count)]
            self.brain.share_memory()
            batch_count = self.cpu_count * 2
        else:
            batch_count = 16
            sim = self.sim_generator()

        experiences:list[tuple[Tensor, int, float, Tensor]] = list()
        for e in range(epoch):
            while len(experiences) < batch_size * batch_count:
                if self.cpu_count == 1:
                    experiences.extend(self.run_episode(self.brain, sim))
                else:
                    experiences.extend(self._parallel_manager(sim_pool, batch_size * 2))
            if e % mod == 0:
                average_reward = sum([exp[2] for exp in experiences]) / len(experiences)
                print(f"Epoch {e:0.2f} done, average reward: {average_reward}, epsilon: {self.brain.epsilon:0.4f}, exp_len: {len(experiences)}")
            self.brain.update(experiences, batch_size=batch_size)
            experiences.clear()
        print(f"Epoch {epoch} done, average reward: {average_reward:0.2f}, epsilon: {self.brain.epsilon:0.4f}")
        self.brain.save_weights() # I already have a Gazillion kevins and maurice in my directory

    def test(self, cli_map=False, max_tick=1500):
        '''
        Test the model without training
        '''
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
                    sim.display_map_cli(snake_vision_only=True)
                r += sim.step(self.brain.qna(s, learning_on=False), max_tick=max_tick)
        ticks = sim.ticks
        snake_len = sim.snake_len
        print(f"{colors.CYAN}Simulation ended after {ticks} ticks, with a size of {snake_len}, average reward {r/ticks}{colors.RESET}")

    def test_record(self, record_file_path=None, max_tick=1500, min_acepted_snake_len=0, min_accepted_tick=0, max_retries=5) -> str:
        '''
        Saves a Json record of the simulation for each step.
        :param record_file_path: the path to save the record file
        :param max_tick: the maximum number of ticks before the simulation ends
        :param min_acepted_snake_len: the minimum snake size to accept the record
        :param min_accepted_tick: the minimum number of ticks to accept the record
        :param max_retries: the maximum number of retries before accepting the record regardless of the conditions
        returns: the path to the record file
        '''
        frames = []
        max_len = 0
        redo = True
        sim = self.sim_generator()
        while redo:
            sim.init_episode()
            with torch.no_grad():
                while True:
                    s, done = sim.get_state()
                    if done: break
                    a = self.brain.qna(s, learning_on=False)
                    frames.append(sim.step_record(a, max_tick=max_tick))
                    max_len = max(max_len, sim.snake_len)
            redo = False
            if len(frames) < min_accepted_tick or max_len < min_acepted_snake_len:
                redo = True
                frames.clear()
                if max_retries == 0:
                    print(f"{colors.RED}Max retries reached without satisfaction, record not saved{colors.RESET}")
                    return None
                max_retries -= 1

        try:
            if record_file_path is None:
                record_file_path = f"records/record_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
                if not os.path.exists("records"):
                    os.makedirs("records")
            with open(record_file_path, "w") as f:
                f.write(json.dumps(frames))
        except Exception as e:
            print(f"An error occured while saving the record: {e}")
            return None
        print(f"{colors.CYAN}Record saved as: {record_file_path}{colors.RESET}")
        return record_file_path
