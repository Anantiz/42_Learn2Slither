import pygame
import os
import json

'''
This class is responsible for visualizing the game state.
We will load a json file that contains individual game-states and render them on the screen.
'''


class Visualizer:

    def __init__(self, width: int = 800, height: int = 600, default_render_speed=2, snake_color=(0, 0, 255), red_color=(255, 0, 0), green_color=(0, 255, 0), background_color=(0, 0, 0)):
        ''' '''
        if width < 400 or height < 400:
            raise ValueError("Window size is too small")
        if width > 1920 or height > 1080:
            raise ValueError("Window size is too big")
        if type(snake_color) is not tuple or type(red_color) is not tuple or type(green_color) is not tuple or type(background_color) is not tuple:
            raise ValueError("Colors must be tuples")
        if len(snake_color) != 3 or len(red_color) != 3 or len(green_color) != 3 or len(background_color) != 3:
            raise ValueError("Colors must be RGB tuples")
        for i in range(3):
            if not (0 <= snake_color[i] <= 255) or not (0 <= red_color[i] <= 255) or not (0 <= green_color[i] <= 255) or not (0 <= background_color[i] <= 255):
                raise ValueError("Color values must be between 0 and 255")

        self.window_width = width
        self.window_height = height
        self.bottom_margin = self.window_height // 10

        self.current_frame = 0
        self.frames = None
        self.auto_render_speed = 1
        self.default_render_speed = default_render_speed

        self.red_color = red_color
        self.green_color = green_color
        self.background_color = background_color

        self.snake_head_color = snake_color
        self.snake_body_color = (
            background_color[0]*0.2+snake_color[0]*0.8,
            background_color[1]*0.2+snake_color[1]*0.8,
            background_color[2]*0.2+snake_color[2]*0.8)

    def load_game(self, path: str):
        '''
        Will load the game state from the given json file
        A frame should be a dictionary as follow:
        {
            id: 0, # The frame index
            msize: [10, 10], # The map size
            # Tuples of coordinates (x, y)
            "snake": [ # The first element is the head
                [1, 1],
                [1, 2],
                [1, 3]
            ],
            "green": [[2, 2]],
            "red": [[3, 2]],
        '''
        if not os.access(path, os.R_OK):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r") as f:
            self.frames = json.load(f)
        # Check if the file is a valid game state
        self.current_frame = 0
        self.last_frame_index = len(self.frames) - 1
        print(f"Loaded {len(self.frames)} frames")

    def generate_frame(self, index: int):
        '''
        Will return an image of the game state at the given index
        '''

        if self.frames is None:
            raise ValueError("No frames loaded")
        if index < 0 or index >= len(self.frames):
            raise ValueError("Invalid frame index")
        frame = self.frames[index]
        map_size = frame["msize"]
        snake = list(map(lambda x: (x[0], x[1]), frame["snake"]))
        green = list(map(lambda x: (x[0], x[1]), frame["green"]))
        red = list(map(lambda x: (x[0], x[1]), frame["red"]))
        self.game_width = self.window_width
        self.game_height = self.window_height
        cell_width = self.game_width // map_size[0]
        cell_height = (self.game_height - self.bottom_margin) // map_size[1]
        # Now create the pygame surface
        surface = pygame.Surface((self.game_width, self.game_height))
        surface.fill(self.background_color)
        # Draw the snake
        for i, s in enumerate(snake):
            x, y = s
            color = self.snake_head_color if i == 0 else self.snake_body_color
            pygame.draw.rect(surface, color, (x*cell_width,
                             y*cell_height, cell_width, cell_height))
        # Draw the green apple
        for g in green:
            x, y = g
            pygame.draw.rect(surface, self.green_color,
                             (x*cell_width, y*cell_height, cell_width, cell_height))
        # Draw the red apple
        for r in red:
            x, y = r
            pygame.draw.rect(surface, self.red_color, (x*cell_width,
                             y*cell_height, cell_width, cell_height))
        return surface

    def step_forward(self):
        '''
        Will render the next frame
        '''
        if self.current_frame == self.last_frame_index:
            return None
        self.current_frame += 1
        return self.generate_frame(self.current_frame)

    def step_backward(self):
        '''
        Will render the previous frame
        '''
        if self.current_frame == 0:
            return None
        self.current_frame -= 1
        return self.generate_frame(self.current_frame)

    def sim_speed(self, speed: float):
        '''
        Will change the simulation speed
        '''
        self.auto_render_speed += speed
        if self.auto_render_speed < 0.5:
            self.auto_render_speed = 0.5
        elif self.auto_render_speed > 6:
            self.auto_render_speed = 6
        print(f"Simulation speed: {self.auto_render_speed} fps")

    def start(self):
        '''
        Will start the rendering loop
        '''
        if self.frames is None:
            print("No frames loaded")
            return
        pygame.init()
        screen = pygame.display.set_mode(
            (self.window_width, self.window_height))
        pygame.display.set_caption("Snake Game Visualizer")
        print("Press the left and right arrow keys to navigate the frames, press escape to exit")
        clock = pygame.time.Clock()
        running = True
        mode = False
        update = True
        surface = None
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    print("Exiting")
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_LEFT:
                        if not mode:
                            self.step_backward()
                            update = True
                        else:
                            self.sim_speed(-0.5)
                    elif event.key == pygame.K_RIGHT:
                        if not mode:
                            self.step_forward()
                            update = True
                        else:
                            self.sim_speed(1)
                    elif event.key == pygame.K_SPACE:
                        mode = not mode
                        print(f"Mode: {'Auto' if mode else 'Manual'}")
            screen.fill((0, 0, 0))
            if mode or update or not surface:
                if mode:
                    surface = self.step_forward()
                else:
                    surface = self.generate_frame(self.current_frame)
                update = False
            if surface:
                screen.blit(surface, (0, 0))
            else:
                print("No frame to render")
                mode = False
            pygame.display.flip()
            clock.tick(
                (self.auto_render_speed if mode else self.default_render_speed))
        pygame.quit()
