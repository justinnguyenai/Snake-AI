import pygame
import random
import time
import numpy as np
import os

# Initialize Pygame
pygame.init()

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set up the game window
width = 800
height = 600
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Snake")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Snake and food
snake_block = 20
snake_speed = 15
speed_boost_factor = 10

# Fonts
font = pygame.font.SysFont(None, 50)
button_font = pygame.font.SysFont(None, 40)

# Directions
UP = (0, -snake_block)
DOWN = (0, snake_block)
LEFT = (-snake_block, 0)
RIGHT = (snake_block, 0)

# Neural Network constants
INPUT_SIZE = 12
HIDDEN_SIZE = 24
OUTPUT_SIZE = 3

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)

    def forward(self, x):
        h = np.dot(x, self.w1)
        h[h<0] = 0  # ReLU activation
        o = np.dot(h, self.w2)
        return o

    def get_action(self, state):
        q_values = self.forward(state)
        return np.eye(3)[np.argmax(q_values)]

def get_state(snake_list, food_position, current_direction):
    head = snake_list[-1]
    point_l = (head[0] - snake_block, head[1])
    point_r = (head[0] + snake_block, head[1])
    point_u = (head[0], head[1] - snake_block)
    point_d = (head[0], head[1] + snake_block)
    
    dir_l = current_direction == LEFT
    dir_r = current_direction == RIGHT
    dir_u = current_direction == UP
    dir_d = current_direction == DOWN

    state = [
        (dir_r and is_collision(*point_r, snake_list)) or 
        (dir_l and is_collision(*point_l, snake_list)) or 
        (dir_u and is_collision(*point_u, snake_list)) or 
        (dir_d and is_collision(*point_d, snake_list)),

        (dir_u and is_collision(*point_r, snake_list)) or 
        (dir_d and is_collision(*point_l, snake_list)) or 
        (dir_l and is_collision(*point_u, snake_list)) or 
        (dir_r and is_collision(*point_d, snake_list)),

        (dir_d and is_collision(*point_r, snake_list)) or 
        (dir_u and is_collision(*point_l, snake_list)) or 
        (dir_r and is_collision(*point_u, snake_list)) or 
        (dir_l and is_collision(*point_d, snake_list)),
        
        dir_l, dir_r, dir_u, dir_d,
        
        food_position[0] < head[0],
        food_position[0] > head[0],
        food_position[1] < head[1],
        food_position[1] > head[1],
        
        len(snake_list) / (width * height / snake_block**2)
    ]

    return np.array(state, dtype=int)

def load_ai_model(filename):
    file_path = os.path.join(current_dir, filename)
    data = np.load(file_path)
    nn = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    nn.w1 = data['w1']
    nn.w2 = data['w2']
    return nn

# Load the trained AI model
ai_model = load_ai_model('snake_best_ai.npz')  # Make sure this file exists in the same directory

def our_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(window, GREEN, [x[0], x[1], snake_block, snake_block])

def message(msg, color, y_displace=0):
    mesg = font.render(msg, True, color)
    text_rect = mesg.get_rect(center=(width/2, height/2 + y_displace))
    window.blit(mesg, text_rect)

def draw_button(text, x, y, w, h, color, text_color):
    pygame.draw.rect(window, color, (x, y, w, h))
    text_surface = button_font.render(text, True, text_color)
    text_rect = text_surface.get_rect(center=(x + w/2, y + h/2))
    window.blit(text_surface, text_rect)

def is_collision(x, y, snake_list):
    if x >= width or x < 0 or y >= height or y < 0:
        return True
    if [x, y] in snake_list[:-1]:
        return True
    return False

def place_food(snake_list):
    while True:
        foodx = round(random.randrange(0, width - snake_block) / 20.0) * 20.0
        foody = round(random.randrange(0, height - snake_block) / 20.0) * 20.0
        if [foodx, foody] not in snake_list:
            return foodx, foody

def ai_move(snake_list, food_position, current_direction):
    state = get_state(snake_list, food_position, current_direction)
    action = ai_model.get_action(state)
    
    if np.array_equal(action, [1, 0, 0]):
        return current_direction
    elif np.array_equal(action, [0, 1, 0]):
        directions = [UP, RIGHT, DOWN, LEFT]
        current_index = directions.index(current_direction)
        return directions[(current_index + 1) % 4]
    else:
        directions = [UP, RIGHT, DOWN, LEFT]
        current_index = directions.index(current_direction)
        return directions[(current_index - 1) % 4]

def game_loop(player_mode):
    game_over = False
    game_close = False

    x1 = width / 2
    y1 = height / 2

    x1_change, y1_change = 0, 0
    current_direction = RIGHT

    snake_list = [[x1, y1]]
    length_of_snake = 1

    foodx, foody = place_food(snake_list)

    score = 0
    start_time = None
    speed_boost = False
    game_time = 0

    clock = pygame.time.Clock()

    while not game_over:
        speed_boost = pygame.key.get_pressed()[pygame.K_LSHIFT] or pygame.key.get_pressed()[pygame.K_RSHIFT]
        current_speed = snake_speed * speed_boost_factor if speed_boost else snake_speed

        while game_close:
            window.fill(BLACK)
            message(f"Game Over! Score: {score}", RED, -50)
            message("Press Q-Quit or R-Restart", RED, 50)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = True
                    game_close = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_r:
                        return  # Return to the main menu

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if player_mode and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and current_direction != RIGHT:
                    current_direction = LEFT
                elif event.key == pygame.K_RIGHT and current_direction != LEFT:
                    current_direction = RIGHT
                elif event.key == pygame.K_UP and current_direction != DOWN:
                    current_direction = UP
                elif event.key == pygame.K_DOWN and current_direction != UP:
                    current_direction = DOWN
                
                # Start the timer when the snake first moves
                if start_time is None:
                    start_time = time.time()

        if not player_mode:
            if start_time is None:
                start_time = time.time()
            current_direction = ai_move(snake_list, (foodx, foody), current_direction)

        x1_change, y1_change = current_direction
        x1 += x1_change
        y1 += y1_change

        if is_collision(x1, y1, snake_list):
            game_close = True

        snake_head = [x1, y1]
        snake_list.append(snake_head)
        if len(snake_list) > length_of_snake:
            del snake_list[0]

        window.fill(BLACK)
        pygame.draw.rect(window, RED, [foodx, foody, snake_block, snake_block])
        our_snake(snake_block, snake_list)
        
        # Update and display score and time
        if start_time is not None:
            time_increment = clock.get_time() / 1000.0
            if speed_boost:
                time_increment *= speed_boost_factor
            game_time += time_increment
            time_text = f'Time: {int(game_time)}s'
        else:
            time_text = 'Time: 0s'
        
        score_surf = font.render(f'Score: {score} {time_text}', True, WHITE)
        window.blit(score_surf, (10, 10))

        # Display speed boost indicator
        if speed_boost:
            boost_surf = font.render('BOOST', True, RED)
            window.blit(boost_surf, (width - 120, 10))

        pygame.display.update()

        if x1 == foodx and y1 == foody:
            foodx, foody = place_food(snake_list)
            length_of_snake += 1
            score += 1  # Increase score by 1 when food is eaten

        clock.tick(current_speed)

    pygame.quit()
    quit()

def main_menu():
    menu = True

    while menu:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if 300 <= mouse_pos[0] <= 500 and 300 <= mouse_pos[1] <= 350:
                    game_loop(True)  # Player mode
                elif 300 <= mouse_pos[0] <= 500 and 400 <= mouse_pos[1] <= 450:
                    game_loop(False)  # AI mode

        window.fill(BLACK)
        message("Snake", GREEN, -100)
        draw_button("PLAYER", 300, 300, 200, 50, WHITE, BLACK)
        draw_button("AI", 300, 400, 200, 50, WHITE, BLACK)
        pygame.display.update()

main_menu()