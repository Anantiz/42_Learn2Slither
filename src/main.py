#!python3
import torch
import traceback
import sys
import os
import argparse

from colors import *
from simulator import Simulator, SnakeSimulator
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


# This docstring with a few modif could also be used as the help message for the program when used with 0 arguments
HELP_STRING = '''
List of arguments my program has to handle:

'@' are mutually exclusive arguments but one of them must be present
'%' are optional arguments, and depending on the action, they might just be ignored or required
    Load a model:
        @train <int> ; number of epochs to train, agents learn during training
        @test <int> ; number of tests to run, agents don't learn during tests
        @record:[<min_tick_required>,<len_required>,<max_retries>] ; record a simulation
        @visualize <record_file> ; visualize a record file


        % -brain <maurice|kevin> ; specify the type of brain to use, Kevin is a neural network, Maurice is a Q-table
            default: Maurice cuz Kevin ain't the sharpest tool in the shed`
        % -load_model <path> ; bad files will raise an exception
        % -save_model <path> ; save the model to a file, otherwise it will be named with the current date
        % -board_size <int> <int> ; set the board size
        % -render_snake_vision ; render the snake vision in the CLI
            (Required by the subject but just impractical since it will just slow everything down
            it will only be taken into account for @test, and will be all dumped in the terminal with 0 effort)
            HOWEVER: For the visualizer, it will toggle or not gold dots in the middle of the cells
            that the snake can see.
'''

# Other things to implement:
# - In the visualizer, highlight the snake vision: In the 4 cardinal directions
# from it's head, put a gold dot in the middle of the cell.
# - Handle SIGINT during training, and save the model
# (Add an extension to the model name to indicate interruption)


def init_gym(args):
    load_file = args.load_model
    save_file = args.save_model
    brain_type = args.brain
    board_size = abs(int(args.board_size)) if args.board_size else 10
    lr = args.lr if args.lr else 0.1 if brain_type == "maurice" else 0.005
    decay = args.decay if args.decay else 0.995 if brain_type == "maurice" else 0.999
    min_epsilon = max(
        0.01, 0.015 if args.min_epsilon is None else args.min_epsilon)
    epsilon = max(min_epsilon, min(1, args.epsilon if args.epsilon else 1))
    gamma = max(0.1, min(0.99, args.gamma))
    target_update_freq = max(1, args.target_update_freq)
    batch_size = max(8, min(256, args.batch_size))
    max_tick = max(100, args.max_tick)
    cpus = max(1, min(os.cpu_count(), args.cpu_count))

    kwargs = {}
    if args.render_snake_vision:
        kwargs["render_snake_vision"] = True
    if load_file:
        if not os.path.exists(load_file):
            raise ValueError(
                f"Model file not found: {load_file}, aborting training")
        if not os.access(load_file, os.R_OK):
            raise ValueError(f"Model file not readable: {load_file}, abort")
    if save_file:
        if not os.path.exists(os.path.dirname(save_file)):
            raise ValueError(
                f"Directory not found for save file: {os.path.dirname(save_file)}")
        if not os.access(os.path.dirname(save_file), os.W_OK):
            raise ValueError(
                f"Directory not writable for save file: {os.path.dirname(save_file)}")

    state_shape, action_count = SnakeSimulator().get_io_shape()
    if brain_type == "kevin":
        brain = Kevin(state_shape, action_count, load_model=load_file,
                      lr=lr, decay=decay, epsilon=epsilon, min_epsilon=min_epsilon,
                      gamma=gamma, target_update_freq=target_update_freq)
    elif brain_type == "maurice":
        brain = Maurice(state_shape, action_count, load_model=load_file,
                        lr=lr, decay=decay, epsilon=epsilon, min_epsilon=min_epsilon,
                        gamma=gamma)

    gym = Gym(brain, lambda: SnakeSimulator(
        board_size=board_size), cpu_count=cpus)

    if args.train:
        training_epochs = min(50_000, abs(args.train))
        gym.train(training_epochs, batch_size=batch_size,
                  save_file=args.save_model)
    elif args.test:
        if load_file is None:
            raise ValueError("Cannot test without a model")
        testing_epochs = min(abs(args.test), 1000)
        results = []
        if args.render_snake_vision:
            testing_epochs = 1  # Don't flood the terminal with snake vision
        for _ in range(testing_epochs):
            results.append(gym.test(
                max_tick=max_tick, cli_map=args.render_snake_vision, render_speed=args.render_speed))
        # result: list[tuple[ticks, end_len, r, max_len]]
        avg_ticks = sum([r[0] for r in results]) / len(results)
        avg_len = sum([r[1] for r in results]) / len(results)
        avg_reward = sum([r[2] for r in results]) / len(results)
        avg_max_len = sum([r[3] for r in results]) / len(results)
        best_len = max([r[1] for r in results])
        best_reward = max([r[2] for r in results])
        longest_ticks = max([r[0] for r in results])
        shortest_ticks = min([r[0] for r in results])
        print(f"{GREEN}Average ticks: {avg_ticks:0.2f}\nAverage final length: {avg_len:0.2f}\nAverage reward: {avg_reward:0.2f}\nAverage max length: {avg_max_len:0.2f}{RESET}")
        print(f"{GREEN}Best length: {best_len}\nBest reward: {best_reward}\nLongest ticks: {longest_ticks}\nShortest ticks: {shortest_ticks}{RESET}")

    elif args.record:
        if load_file is None:
            raise ValueError("Cannot record without a model")

        min_acepted_snake_len = max(1, args.min_acepted_snake_len)
        min_accepted_tick = max(1, args.min_accepted_tick)
        max_retries = max(1, min(args.max_retries, 100))
        path = gym.test_record(max_tick=max_tick, min_acepted_snake_len=min_acepted_snake_len,
                               min_accepted_tick=min_accepted_tick, max_retries=max_retries)
        if path:
            print(f"Recorded simulation at: {path}")
        else:
            print(f"Failed to record a simulation")
    elif args.visualize:
        # Here so it doesn't import pygame like a madman everytimes
        from visualizer import Visualizer
        visualizer = Visualizer()
        visualizer.load_game(args.visualize)
        visualizer.start()
    else:
        raise ValueError("Unknown action")


def main():
    parser = argparse.ArgumentParser(
        description=HELP_STRING, formatter_class=argparse.RawTextHelpFormatter)

    # Mutually exclusive group (one of these must be present)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--train', type=int, help='Number of epochs to train, agents learn during training')
    group.add_argument(
        '--test', type=int, help='Number of tests to run, agents don\'t learn during tests')
    group.add_argument('--record', action='store_true',
                       default=False, help='Record a simulation with format')
    group.add_argument('--visualize', type=str, help='Visualize a record file')

    # Optional arguments
    parser.add_argument('--brain', choices=['maurice', 'kevin'], default='maurice',
                        help='Specify the type of brain to use (default: Maurice)')
    parser.add_argument('--load_model', type=str,
                        help='Load a model from the specified path')
    parser.add_argument('--save_model', type=str, default=None,
                        help='Save the model to the specified path')
    parser.add_argument('--board_size', type=int,
                        help='Set the board size (a square board)')
    parser.add_argument('--render_snake_vision', default=False,
                        action='store_true', help='Render the snake vision in the CLI')
    parser.add_argument('--render_speed', type=float,
                        default=0.5, help='Speed of the rendering')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--cpu_count', type=int, choices=range(1, os.cpu_count()),
                        default=1, help='Number of workers to use for training, might crash tho')
    parser.add_argument('--lr', type=float, help='Learning rate for the brain')
    parser.add_argument('--decay', type=float, help='Decay rate for epsilon')
    parser.add_argument('--epsilon', type=float,
                        help='Initial epsilon value for the brain')
    parser.add_argument('--min_epsilon', type=float, default=0.01,
                        help='Minimum epsilon value for the brain')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Gamma value for the brain')
    parser.add_argument('--target_update_freq', default=25,
                        type=int, help='Frequency to update the target network')
    parser.add_argument('--max_tick', type=int, default=1500,
                        help='Maximum tick for a simulation')
    parser.add_argument('--min_acepted_snake_len', default=10,
                        type=int, help='Minimum accepted snake length for a record')
    parser.add_argument('--min_accepted_tick', default=50,
                        type=int, help='Minimum accepted tick for a record')
    parser.add_argument('--max_retries', default=5, type=int,
                        help='Maximum retries for a record')
    parser.add_argument('--debug', action='store_true',
                        default=False, help='Debug mode')

    args = parser.parse_args()

    # Here you can add the logic to handle the parsed arguments
    try:
        init_gym(args)
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
