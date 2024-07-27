import numpy as np
import random
from enum import Enum
from collections import namedtuple
import time

# Snake game constants
WIDTH = 800
HEIGHT = 600
BLOCK_SIZE = 20

# Neural Network constants
INPUT_SIZE = 12
HIDDEN_SIZE = 24
OUTPUT_SIZE = 3

# Genetic Algorithm constants
POPULATION_SIZE = 500
MUTATION_RATE = 0.1
GENERATIONS = 500

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class SnakeGame:
    def __init__(self, w=WIDTH, h=HEIGHT):
        self.w = w
        self.h = h
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
            y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
            food = Point(x, y)
            if food not in self.snake:
                self.food = food
                break

    def play_step(self, action):
        self.frame_iteration += 1
        
        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

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

def get_state(game):
    head = game.snake[0]
    point_l = Point(head.x - BLOCK_SIZE, head.y)
    point_r = Point(head.x + BLOCK_SIZE, head.y)
    point_u = Point(head.x, head.y - BLOCK_SIZE)
    point_d = Point(head.x, head.y + BLOCK_SIZE)
    
    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        (dir_r and game.is_collision(point_r)) or 
        (dir_l and game.is_collision(point_l)) or 
        (dir_u and game.is_collision(point_u)) or 
        (dir_d and game.is_collision(point_d)),

        (dir_u and game.is_collision(point_r)) or 
        (dir_d and game.is_collision(point_l)) or 
        (dir_l and game.is_collision(point_u)) or 
        (dir_r and game.is_collision(point_d)),

        (dir_d and game.is_collision(point_r)) or 
        (dir_u and game.is_collision(point_l)) or 
        (dir_r and game.is_collision(point_u)) or 
        (dir_l and game.is_collision(point_d)),
        
        dir_l, dir_r, dir_u, dir_d,
        
        game.food.x < game.head.x,
        game.food.x > game.head.x,
        game.food.y < game.head.y,
        game.food.y > game.head.y,
        
        len(game.snake) / (game.w * game.h / BLOCK_SIZE**2)
    ]

    return np.array(state, dtype=int)

def crossover(nn1, nn2):
    child = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    child.w1 = nn1.w1 * 0.5 + nn2.w1 * 0.5
    child.w2 = nn1.w2 * 0.5 + nn2.w2 * 0.5
    return child

def mutate(nn):
    nn.w1 += np.random.randn(*nn.w1.shape) * MUTATION_RATE
    nn.w2 += np.random.randn(*nn.w2.shape) * MUTATION_RATE

def train():
    population = [NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE) for _ in range(POPULATION_SIZE)]
    best_score = 0
    best_fitness = 0
    
    for generation in range(GENERATIONS):
        gen_start_time = time.time()
        fitnesses = []
        scores = []
        
        for nn in population:
            game = SnakeGame()
            while True:
                state = get_state(game)
                action = nn.get_action(state)
                _, game_over, score = game.play_step(action)
                if game_over:
                    break
            
            fitness = score
            fitnesses.append(fitness)
            scores.append(score)
        
        gen_end_time = time.time()
        gen_duration = gen_end_time - gen_start_time
        
        max_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        
        if max_score > best_score or max_fitness > best_fitness:
            best_score = max(best_score, max_score)
            best_fitness = max(best_fitness, max_fitness)
            best_nn = population[np.argmax(fitnesses)]
            np.savez('best_snake_ai.npz', w1=best_nn.w1, w2=best_nn.w2)
        
        print(f"Generation {generation + 1}:")
        print(f"  Duration: {gen_duration:.2f} seconds")
        print(f"  Max Fitness: {max_fitness:.2f}")
        print(f"  Avg Fitness: {avg_fitness:.2f}")
        print(f"  Max Score: {max_score}")
        print(f"  Avg Score: {avg_score:.2f}")
        print(f"  Best Score So Far: {best_score}")
        
        top_performers = np.argsort(fitnesses)[-POPULATION_SIZE//4:]  # Keep top 25%
        new_population = [population[i] for i in top_performers]
        
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(new_population, 2)
            child = crossover(parent1, parent2)
            mutate(child)
            new_population.append(child)
        
        population = new_population
        
        if max_score >= 150:
            print(f"Target score reached in generation {generation + 1}!")
            break

    print("Training complete.")

if __name__ == '__main__':
    train()