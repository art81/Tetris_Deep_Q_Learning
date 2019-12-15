#!/usr/bin/env python
from random import randrange as rand
import pygame, sys
import random
import numpy as np
from collections import deque
from keras.layers import Dense,Flatten,Input,Reshape
from keras.optimizers import Adam

EPISODES = 1000

# The configuration of the tetris board
config = {
	'cell_size':40,
	'cols':		8,
	'rows':		16,
	'delay':	750,
	'maxfps':	30
}

# Template for Deep Q-learning Agent from (https://github.com/keon/deep-q-learning)
class DQNAgent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=2000)
		self.gamma = 0.95	# discount rate
		self.epsilon = 1.0  # exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.model = self._build_model()

	def build_model(self):
		# Neural Net for Deep-Q learning Model
		model = tf.keras.Sequential(name = 'Deep_Q_Learning_Model')
		#State size for tetris is
		model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=self.state_shape))
		model.add(layers.LeakyReLU())
		model.add(layers.Dropout(0.3))

		model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
		model.add(layers.LeakyReLU())
		model.add(layers.Dropout(0.3))

		model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
		model.add(layers.LeakyReLU())

		model.add(layers.Flatten())
		model.add(layers.Dense(config['rows']*config['cols']*64/2, activation="relu"))
		model.add(layers.Dense(config['rows']*config['cols']*64/4, activation="relu"))
		model.add(layers.Dense(self.action_size, activation="softmax"))

		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

		return model

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])  # returns action

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
			  target = reward + self.gamma * \
					   np.amax(self.model.predict(next_state)[0])
			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

# (ADAPTED BY Andrew Tarnoff FOR Q-LEARNING) tetris implementation
#
# Control keys:
# Down - Drop stone faster
# Left/Right - Move stone
# Up - Rotate Stone clockwise
#
# Have fun!

# Copyright (c) 2010 "Kevin Chabowski"<kevin@kch42.de>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

class TetrisEnv(object):
	colors = [
	(0,   0,   0  ),
	(255, 0,   0  ),
	(0,   150, 0  ),
	(0,   0,   255),
	(255, 120, 0  ),
	(255, 255, 0  ),
	(180, 0,   255),
	(0,   220, 220)
	]

	# Define the shapes of the single parts
	tetris_shapes = [
		[[1, 1, 1],
		 [0, 1, 0]],

		[[0, 2, 2],
		 [2, 2, 0]],

		[[3, 3, 0],
		 [0, 3, 3]],

		[[4, 0, 0],
		 [4, 4, 4]],

		[[0, 0, 5],
		 [5, 5, 5]],

		[[6, 6, 6, 6]],

		[[7, 7],
		 [7, 7]]
	]

	def rotate_clockwise(self, shape):
		return [ [ shape[y][x] for y in range(len(shape)) ] for x in range(len(shape[0]) - 1, -1, -1) ]

	def check_collision(self, board, shape, offset):
		off_x, off_y = offset
		for cy, row in enumerate(shape):
			for cx, cell in enumerate(row):
				try:
					if cell and board[ cy + off_y ][ cx + off_x ]:
						return True
				except IndexError:
					return True
		return False

	def remove_row(self, board, row):
		del board[row]
		return [[0 for i in range(config['cols'])]] + board

	def join_matrixes(self, mat1, mat2, mat2_off):
		off_x, off_y = mat2_off
		for cy, row in enumerate(mat2):
			for cx, val in enumerate(row):
				mat1[cy+off_y-1][cx+off_x] += val
		return mat1

	def new_board(self):
		board = [ [ 0 for x in range(config['cols']) ] for y in range(config['rows']) ]
		board += [[ 1 for x in range(config['cols'])]]
		return board

	def __init__(self):
		pygame.init()
		pygame.key.set_repeat(250,25)
		self.width = config['cell_size']*config['cols']
		self.height = config['cell_size']*config['rows']

		self.screen = pygame.display.set_mode((self.width, self.height))
		pygame.event.set_blocked(pygame.MOUSEMOTION) # We do not need
													 # mouse movement
													 # events, so we
													 # block them.
		self.init_game()

	def new_stone(self):
		self.stone = self.tetris_shapes[rand(len(self.tetris_shapes))]
		self.stone_x = int(config['cols'] / 2 - len(self.stone[0])/2)
		self.stone_y = 0

		if self.check_collision(self.board, self.stone, (self.stone_x, self.stone_y)):
			self.gameover = True

	def init_game(self):
		self.board = self.new_board()
		self.new_stone()

	def draw_matrix(self, matrix, offset):
		off_x, off_y  = offset
		for y, row in enumerate(matrix):
			for x, val in enumerate(row):
				if not val == 0:
					pygame.draw.rect(
						self.screen,
						self.colors[val],
						((off_x+x) * config['cell_size'], (off_y+y) * config['cell_size'], config['cell_size'], config['cell_size']),0)
				pygame.draw.rect(self.screen, (0,0,0), ((off_x+x) * config['cell_size'], (off_y+y) * config['cell_size'], config['cell_size'], config['cell_size']),3)

	def move(self, delta_x):
		if not self.gameover:
			new_x = self.stone_x + delta_x
			if new_x < 0:
				new_x = 0
			if new_x > config['cols'] - len(self.stone[0]):
				new_x = config['cols'] - len(self.stone[0])
			if not self.check_collision(self.board,
								   self.stone,
								   (new_x, self.stone_y)):
				self.stone_x = new_x

	def drop(self):
		if not self.gameover:
			self.stone_y += 1
			if self.check_collision(self.board, self.stone, (self.stone_x, self.stone_y)):
				self.board = self.join_matrixes(self.board, self.stone, (self.stone_x, self.stone_y))
				self.new_stone()
				while True:
					for i, row in enumerate(self.board[:-1]):
						if 0 not in row:
							self.board = self.remove_row(self.board, i)
							break
					else:
						break

	def full_drop(self):
		if not self.gameover:
			startY = self.stone_y
			while not self.check_collision(self.board, self.stone, (self.stone_x, startY)):
				startY += 1
			#Update board the create a new piece
			self.board = self.join_matrixes(self.board,self.stone,(self.stone_x, startY))
			self.new_stone()
			while True:
				for i, row in enumerate(self.board[:-1]):
					if 0 not in row:
						self.board = self.remove_row(self.board, i)
						break
				else:
					break

	def rotate_stone(self):
		if not self.gameover:
			new_stone = self.rotate_clockwise(self.stone)
			#Maybe add checks for collisions at all adjacent positions of the stone??
			if not self.check_collision(self.board, new_stone, (self.stone_x, self.stone_y)):
				self.stone = new_stone

	def start_game(self):
		if self.gameover:
			self.init_game()
			self.gameover = False

def runTetrisGame(tetrisEnv):
	key_actions = {
		'LEFT':	 lambda:tetrisEnv.move(-1),
		'RIGHT': lambda:tetrisEnv.move(+1),
		'DOWN':	 tetrisEnv.full_drop,
		'UP':	 tetrisEnv.rotate_stone,
	}

	tetrisEnv.gameover = False

	pygame.time.set_timer(pygame.USEREVENT+1, config['delay'])
	dont_burn_my_cpu = pygame.time.Clock()
	while 1:
		tetrisEnv.screen.fill((255,255,255))
		if tetrisEnv.gameover:
			print("Game Over :(")
			break
		else:
			tetrisEnv.draw_matrix(tetrisEnv.board, (0,0))
			tetrisEnv.draw_matrix(tetrisEnv.stone, (tetrisEnv.stone_x, tetrisEnv.stone_y))
		pygame.display.update()

		for event in pygame.event.get():
			if event.type == pygame.USEREVENT+1:
				tetrisEnv.drop()
			elif event.type == pygame.KEYDOWN:
				for key in key_actions:
					if event.key == eval("pygame.K_" + key):
						key_actions[key]()

		dont_burn_my_cpu.tick(config['maxfps'])

if __name__ == "__main__":
	#env = gym.make('CartPole-v1')
	tetrisEnv = TetrisEnv()
	runTetrisGame(tetrisEnv)
	state_size = (config['rows'], config['cols'], 1)
	action_size = 4 #Left, Right, Rotate (up), drop to the bottom (down)
	agent = DQNAgent(state_size, action_size)
	# agent.load("./save/cartpole-dqn.h5")
	done = False
	batch_size = 32

	for e in range(EPISODES):
		state = env.reset()
		state = np.reshape(state, [1, state_size])
		for time in range(500):
			# env.render()
			action = agent.act(state)
			next_state, reward, done, _ = env.step(action)
			reward = reward if not done else -10
			next_state = np.reshape(next_state, [1, state_size])
			agent.remember(state, action, reward, next_state, done)
			state = next_state
			if done:
				print("episode: {}/{}, score: {}, e: {:.2}"
					  .format(e, EPISODES, time, agent.epsilon))
				break
			if len(agent.memory) > batch_size:
				agent.replay(batch_size)
		# if e % 10 == 0:
		#	 agent.save("./save/cartpole-dqn.h5")
