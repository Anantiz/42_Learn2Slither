from colors import *
from random import randint
from collections import deque
import torch

MAGIC_EMPTY_VALUE = 999999

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

    action_map = { 0 : "up", 1 : "left", 2 : "down", 3 : "right"} # Don't change the values
    action_result = {"dead-wall", "dead-self", "dead-red-apple", "red-apple", "green-apple", "nothing"}
    rewards = {
        "dead-wall": -100,
        "dead-self": -100,
        "dead-red-apple": -50,
        "red-apple": -10,
        "green-apple": 100,
        "nothing": -1
    }

    def __init__(self, board_size=10, snake_len=3):
        if board_size > 1000:
            print(f"{RED}Map size too large, keep it below a thousand{RESET}")
            raise ValueError
        if board_size < 5:
            print(f"{RED}Map size too small, keep it above 5{RESET}")
            raise ValueError
        self.board_size = board_size
        self.snake_len = snake_len
        self.snake_body: deque[tuple[int, int]] = deque()
        self.green_apple_pos: set[tuple] = set()
        self.red_apple_pos: set[tuple] = set()
        self.done = False

    def rework_state_representation_kevin(self, state:torch.Tensor) -> torch.Tensor:
        ''' The model performs like a Orangutan on a unicycle, mayhaps it needs glasses '''
        # Current state is a 4x4 matrix where for each direction we have: dist-red(bad), dist-green(good), dist-body, dist-wall
        # And something that is not see has dist=0 which i guess is stupid

        # Suggestions:
        # - Merge body and wall cuz both ways you die, so it's kinda the same thing
        # - FIND and dam WAY to explain the model that the North-values-Input are like NORTH lol, and South are SOUTH ...
        # 1. Make good things (green apples) a positive distance, and bad things (red apples) a negative distance
        #    keep walls as is, and merge body with walls, maybe the agent will want to minimize the value in the apple-category
        #    and maximize the value in the wall/body category, so we go from 4x4 to 4x2
        return state
        new_state = torch.zeros(4, 2)
        for i in range(4):
            new_state[i, 0] = state[i, 1] if state[i, 1] > 0 else -state[i, 0]
            new_state[i, 1] = state[i][2] if state[i][3] == 0 else state[i][3] if state[i][2] == 0 else min(state[i][2], state[i][3])
            if new_state[i, 1] == 0: new_state[i, 1] = MAGIC_EMPTY_VALUE
            if new_state[i, 0] == 0: new_state[i, 0] = MAGIC_EMPTY_VALUE
        return new_state

    def get_io_shape(self) -> tuple:
        """ Return: (State-size, Action-size) """
        return (4, 4), 4

    def display_map_cli(self, snake_vision_only=False):
        """TODO: Be careful, you are not allowed to turn in this, it should only display the snake vision"""
        snake_head_x, snake_head_y = self.snake_body[0]
        print(f"{BOLD}", end="")
        print("W" * (self.board_size + 2), end="")
        print(f"{RESET}")
        for y in range(self.board_size):
            print(f"{BOLD}W{RESET}", end="")
            for x in range(self.board_size):
                if x != snake_head_x and y != snake_head_y:
                    print("?", end="")
                    continue
                pos = (x, y)
                if pos in self.snake_body:
                    print(f"{BLUE}H{RESET}" if pos == self.snake_body[0] else f"{CYAN}S{RESET}", end="")
                elif pos in self.green_apple_pos:
                    print(f"{GREEN}G{RESET}", end="")
                elif pos in self.red_apple_pos:
                    print(f"{RED}R{RESET}", end="")
                else:
                    print(f"{YELLOW}0{RESET}", end="")
            print(f"{BOLD}W{RESET}")
        print(f"{BOLD}", end="")
        print("W" * (self.board_size + 2), end="")
        print(f"{RESET}\n")

    def generate_json_frame(self):
        """
        Will return a dictionary representing the current state of the game
        """
        return {
            "id": self.ticks,
            "msize": [self.board_size, self.board_size],
            "snake": list(self.snake_body),
            "green": list(self.green_apple_pos),
            "red": list(self.red_apple_pos)
        }

    def spawn_snake(self, initial_size=3):
        """
        :param initial_size: initial size of the snake
        Randomly spawn the sanke, the body shall be contiguous
        """
        if self.board_size < initial_size:
            print(f"{RED}Map size too small for the snake{RESET}")
            raise ValueError
        self.snake_len = initial_size
        x = randint(0, self.board_size - 1)
        y = randint(0, self.board_size - 1)
        # Find a direction to contiguously spawn the snake
        self.snake_body.clear()
        self.snake_body.append((x, y))
        if x > initial_size:
            for i in range(1, initial_size):
                self.snake_body.append((x - i, y))
        elif y > initial_size:
            for i in range(1, initial_size):
                self.snake_body.append((x, y - i))
        elif x < self.board_size - initial_size:
            for i in range(1, initial_size):
                self.snake_body.append((x + i, y))
        else:
            for i in range(1, initial_size):
                self.snake_body.append((x, y + i))

    def spawn_green_apple(self, n=1):
        while n > 0:
            x = randint(0, self.board_size - 1)
            y = randint(0, self.board_size - 1)
            pos = (x, y)
            if pos in self.snake_body or pos in self.red_apple_pos:
                continue
            self.green_apple_pos.add(pos)
            n -= 1

    def spawn_red_apple(self, n=1):
        while n > 0:
            x = randint(0, self.board_size - 1)
            y = randint(0, self.board_size - 1)
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
                if head_y < 0 or head_y == self.board_size:
                    return "dead-wall"
            case 1: # left
                head_x -= 1
                if head_x < 0 or head_x == self.board_size:
                    return "dead-wall"
            case 2: # down
                head_y += 1
                if head_y < 0 or head_y == self.board_size:
                    return "dead-wall"
            case 3: # right
                head_x += 1
                if head_x < 0 or head_x == self.board_size:
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

    def get_state(self, rework=False) -> tuple[torch.tensor, bool]:
        """
        Make a 4x4 matrix:

        UP:    Dist-Red; Dist-Green; Dist-Body; Dist-Wall
        LEFT:  _
        DOWN:  _
        RIGHT: _
        if something doesn't show up put distance to 0
        :return: state tensor, done: True if the game is over
        """

        # You can Optimize all of this by simply searching if the apples are in the same row or column as the head
        # instead of checking all the way to the head, BUT, too lazy
        state = torch.zeros((4, 4), dtype=torch.float32)
        if self.done:
            return state, self.done
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
        for i in range(1, self.board_size - head_y):
            if state[2][0] == 0 and (head_x, head_y + i) in self.red_apple_pos:
                state[2][0] = i
            elif state[2][1] == 0 and (head_x, head_y + i) in self.green_apple_pos:
                state[2][1] = i
            elif state[2][2] == 0 and (head_x, head_y + i) in self.snake_body:
                state[2][2] = i
        state[2][3] = self.board_size - head_y
        # Right state[3]
        for i in range(1, self.board_size - head_x):
            if state[3][0] == 0 and (head_x + i, head_y) in self.red_apple_pos:
                state[3][0] = i
            elif state[3][1] == 0 and (head_x + i, head_y) in self.green_apple_pos:
                state[3][1] = i
            elif state[3][2] == 0 and (head_x + i, head_y) in self.snake_body:
                state[3][2] = i
        state[3][3] = self.board_size - head_x
        return state, self.done

    def init_episode(self, green_apple_count=2, red_apple_count=1, initial_size=3) -> tuple[torch.tensor, list[int]]:
        """
        called to initialize the simulator and get the initial state and possible actions
        """
        self.ticks = 0
        self.spawn_snake(initial_size=initial_size)
        self.green_apple_pos.clear()
        self.red_apple_pos.clear()
        self.spawn_green_apple(green_apple_count)
        self.spawn_red_apple(red_apple_count)
        self.done = False
        self.last_action = None
        return self.get_state(), [0, 1, 2, 3]

    def check_if_apple_in_dir(self, apple_set, dir):
        snake_x, snake_y = self.snake_body[0]
        match dir:
            case 0:
                for a in apple_set:
                    if a[1] < snake_y:
                        return True
                return False
            case 1:
                for a in apple_set:
                    if a[0] < snake_x:
                        return True
                    return False
            case 2:
                for a in apple_set:
                    if a[0] > snake_x:
                        return True
                    return False
            case 3:
                for a in apple_set:
                    if a[1] < snake_x:
                        return True
                    return False
        return False

    def get_reward(self, action , r) ->int:
        if r != "nothing":
            return self.rewards[r]
        # Reward if the direction is
        if self.check_if_apple_in_dir(self.green_apple_pos, action):
            return self.rewards["green-apple"] / 20
        if self.check_if_apple_in_dir(self.red_apple_pos, action):
            return self.rewards["red-apple"] / 15
        return self.rewards[r]

    def step_record(self, action: int, max_tick=1500) -> tuple[bool, dict]:
        """
        :param action: action taken by the agent
        :max_tick: maximum number of ticks before the game ends
        return: done, frame
        """
        if action not in {0, 1, 2, 3}:
            print(f"Simulator can't step invalid action")
            raise ValueError
        self.ticks += 1
        if self.ticks == max_tick:
            self.done = True
            return self.generate_json_frame()
        result = self.move_snake(action)
        if result in {"dead-wall", "dead-self", "dead-red-apple", None}:
            self.done = True
        return self.generate_json_frame()

    def step(self, action: int, max_tick=-100) -> tuple[torch.tensor, list[int]]:
        """
        :param action: action taken by the agent
        :max_tick: maximum number of ticks before the game ends, negative values set training mode
        :return: next state and reward of last action
        """
        if action not in {0, 1, 2, 3}:
            print(f"Simulator can't step invalid action")
            return 0
        self.ticks += 1
        if max_tick < 0: # Training
            if self.ticks > max_tick * -(self.snake_len - 2):
                self.done = True
        elif self.ticks == max_tick:
            self.done = True
            return 0
        result = self.move_snake(action)
        if result in {"dead-wall", "dead-self", "dead-red-apple", None}:
            self.done = True
        return self.get_reward(action, result)