from colors import *
from random import randint
from collections import deque
import torch
import math

class Simulator:
    """
    Rules: For all simulators, action=0 is the default action when the agent has not yet an action-list. It is thus expected to mostly do nothing.
    """
    def __init__(self):
        pass

    def init_episode(self) -> tuple[int, list[int]]:
        """
        called to initialize the simulator and get the initial state and possible actions for a new episode
        """
        print(f"{RED}Simulator.init() not implemented{RESET}")
        return 0, []

    def step(self, action: int) -> tuple[torch.tensor, float]:
        """
        :param action: action taken by the agent
        :return: next state and possible actions
        """
        print(f"{RED}Simulator.step() not implemented{RESET}")
        return None, 0

class SnakeSimulator(Simulator):

    '''
    Symbols:
     - W: Wall
     - 0: Empty
     - H: Head of the snake
     - S: Body of the snake
     - R: Red apple (bad one, -1)
     - G: Green apple (good one, +1)

    Coordinates are between 0 and map_size - 1
    '''

    # Don't change this
    action_map = {
        0 : "up",
        1 : "left",
        2 : "down",
        3 : "right"
    }

    action_map_rev = {
        "up" : 0,
        "left" : 1,
        "down" : 2,
        "right" : 3
    }

    action_result = {"dead-wall", "dead-self", "dead-red-apple", "red-apple", "green-apple", "nothing"}

    rewards = {
        "dead-wall": -100,
        "dead-self": -100,
        "dead-red-apple": -50,
        "red-apple": -2,
        "green-apple": 10,
        "nothing": 0.1
    }

    def __init__(self, map_size=10, snake_len=3):
        self.map_size = map_size
        self.snake_len = snake_len
        self.snake_body: deque[tuple[int, int]] = deque()
        self.green_apple_pos: set[tuple] = set()
        self.red_apple_pos: set[tuple] = set()

    def get_io_shape(self) -> tuple:
        """ Return: (State-size, Action-size) """
        return (4, 4), len(self.action_map)

    def display_map_cli(self):
        """TODO: Be careful, you are not allowed to turn in this, it should only display the snake vision"""
        print(f"{BOLD}", end="")
        print("W" * (self.map_size + 2), end="")
        print(f"{RESET}")
        for y in range(self.map_size):
            print(f"{BOLD}W{RESET}", end="")
            for x in range(self.map_size):
                pos = (x, y)
                if pos in self.snake_body:
                    print(f"{BLUE}H{RESET}" if pos == self.snake_body[0] else f"{CYAN}S{RESET}", end="")
                elif pos in self.green_apple_pos:
                    print(f"{GREEN}G{RESET}", end="")
                elif pos in self.red_apple_pos:
                    print(f"{RED}R{RESET}", end="")
                else:
                    print("0", end="")
            print(f"{BOLD}W{RESET}")
        print(f"{BOLD}", end="")
        print("W" * (self.map_size + 2), end="")
        print(f"{RESET}\n")

    def spawn_snake(self, initial_size=3):
        """
        :param initial_size: initial size of the snake
        Randomly spawn the sanke, the body shall be contiguous
        """
        if self.map_size < initial_size:
            print(f"{RED}Map size too small for the snake{RESET}")
            raise ValueError
        self.snake_len = initial_size
        x = randint(0, self.map_size - 1)
        y = randint(0, self.map_size - 1)
        # Find a direction to contiguously spawn the snake
        self.snake_body.clear()
        self.snake_body.append((x, y))
        if x > initial_size:
            for i in range(1, initial_size):
                self.snake_body.append((x - i, y))
        elif y > initial_size:
            for i in range(1, initial_size):
                self.snake_body.append((x, y - i))
        elif x < self.map_size - initial_size:
            for i in range(1, initial_size):
                self.snake_body.append((x + i, y))
        else:
            for i in range(1, initial_size):
                self.snake_body.append((x, y + i))

    def spawn_green_apple(self, n=1):
        while n > 0:
            x = randint(0, self.map_size - 1)
            y = randint(0, self.map_size - 1)
            pos = (x, y)
            if pos in self.snake_body or pos in self.red_apple_pos:
                continue
            self.green_apple_pos.add(pos)
            n -= 1

    def spawn_red_apple(self, n=1):
        while n > 0:
            x = randint(0, self.map_size - 1)
            y = randint(0, self.map_size - 1)
            pos = (x, y)
            if pos in self.snake_body or pos in self.green_apple_pos:
                continue
            self.red_apple_pos.add(pos)
            n -= 1

    def move_snake(self, direction: int) -> str:
        """
        :param direction: direction of the movement
        return: action result (dead-wall, dead-self, dead-red-apple, red-apple, green-apple, nothing)
        """
        head: tuple = self.snake_body[0]
        head_x, head_y = head
        # Check directly wall colisions in the match, it is more code duplication but makes less comparisons per call
        match direction:
            case 0: # up
                head_y -= 1
                if head_y < 0 or head_y == self.map_size:
                    return "dead-wall"
            case 1: # left
                head_x -= 1
                if head_x < 0 or head_x == self.map_size:
                    return "dead-wall"
            case 2: # down
                head_y += 1
                if head_y < 0 or head_y == self.map_size:
                    return "dead-wall"
            case 3: # right
                head_x += 1
                if head_x < 0 or head_x == self.map_size:
                    return "dead-wall"
            case _:
                print(f"{RED}Invalid direction, fix your code !{RESET}")
                return None
        new_head_pos = (head_x, head_y)
        if new_head_pos in self.snake_body:
            return "dead-self"
        elif new_head_pos in self.green_apple_pos:
            self.snake_len += 1
            self.green_apple_pos.remove(new_head_pos)
            self.spawn_green_apple()
            self.snake_body.appendleft(new_head_pos)
            return "green-apple"
        elif new_head_pos in self.red_apple_pos:
            self.snake_len -= 1
            if self.snake_len == 0:
                return "dead-red-apple"
            self.red_apple_pos.remove(new_head_pos)
            self.spawn_red_apple()
            self.snake_body.pop()
            self.snake_body.pop()
            self.snake_body.appendleft(new_head_pos)
            return "red-apple"
        else:
            self.snake_body.pop()
            self.snake_body.appendleft(new_head_pos)
            return "nothing"

    def get_state(self) -> torch.tensor:
        """
        Make a 4x4 matrix:

        UP:    Dist-Red; Dist-Green; Dist-Body; Dist-Wall
        LEFT:  _
        DOWN:  _
        RIGHT: _

        if something doesn't show up put distance to 0, later try -1 to check if it works better/worse
        """

        # You can Optimize all of this by simply searching if the apples are in the same row or column as the head
        # instead of checking all the way to the head, BUT, too lazy
        state = torch.zeros((4, 4), dtype=torch.float32)
        head = self.snake_body[0]
        head_x, head_y = head
        # Up state[0]
        for i in range(1, head_y + 1):
            if state[0][0] == 0 and (head_x, head_y - i) in self.red_apple_pos:
                state[0][0] = i
            elif state[0][1] == 0 and (head_x, head_y - i) in self.green_apple_pos:
                state[0][1] = i
            elif state[0][2] == 0 and (head_x, head_y - i) in self.snake_body: # Refer to the closest body part
                state[0][2] = i
        state[0][3] = head_y + 1
        # Left state[1]
        for i in range(1, head_x + 1):
            if state[1][0] == 0 and (head_x - i, head_y) in self.red_apple_pos:
                state[1][0] = i
            elif state[1][1] == 0 and (head_x - i, head_y) in self.green_apple_pos:
                state[1][1] = i
            elif state[1][2] == 0 and (head_x - i, head_y) in self.snake_body:
                state[1][2] = i
        state[1][3] = head_x + 1
        # Down state[2]
        for i in range(1, self.map_size - head_y):
            if state[2][0] == 0 and (head_x, head_y + i) in self.red_apple_pos:
                state[2][0] = i
            elif state[2][1] == 0 and (head_x, head_y + i) in self.green_apple_pos:
                state[2][1] = i
            elif state[2][2] == 0 and (head_x, head_y + i) in self.snake_body:
                state[2][2] = i
        state[2][3] = self.map_size - head_y
        # Right state[3]
        for i in range(1, self.map_size - head_x):
            if state[3][0] == 0 and (head_x + i, head_y) in self.red_apple_pos:
                state[3][0] = i
            elif state[3][1] == 0 and (head_x + i, head_y) in self.green_apple_pos:
                state[3][1] = i
            elif state[3][2] == 0 and (head_x + i, head_y) in self.snake_body:
                state[3][2] = i
        state[3][3] = self.map_size - head_x
        return state

    def init_episode(self) -> tuple[torch.tensor, list[int]]:
        """
        called to initialize the simulator and get the initial state and possible actions
        """
        self.ticks = 0
        self.spawn_snake()
        self.green_apple_pos.clear()
        self.red_apple_pos.clear()
        self.spawn_green_apple(5)
        self.spawn_red_apple()
        return self.get_state(), [0, 1, 2, 3]

    def step(self, action: int) -> tuple[torch.tensor, list[int]]:
        """
        :param action: action taken by the agent
        :return: next state and reward of last action
        """
        if not action:
            return self.get_state(), 0
        self.ticks += 1
        if self.ticks > 1000:
            self.display_map_cli()
            print(f"{RED}Game took too long{RESET}")
            return None, -100
        result = self.move_snake(action)
        if result in {"dead-wall", "dead-self", "dead-red-apple", None}:
            print(f"{MAGENTA}Game ended{RESET}: ticks={self.ticks} snake_len={self.snake_len}")
            return None, self.rewards[result]
        return self.get_state(), (self.rewards[result] + (math.log((self.snake_len - 3)) if self.snake_len > 3 else 0))
