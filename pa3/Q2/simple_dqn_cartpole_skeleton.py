import gym
import random
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class DQN:
	REPLAY_MEMORY_SIZE = 100000			# number of tuples in experience replay  
	EPSILON_START = 1   		 		# epsilon of epsilon-greedy exploation
	EPSILON_DECAY = 0.995 	 			# exponential decay multiplier for epsilon
	EPSILON_MIN = 0.001					# minimum value of epsilon allowed
	HIDDEN1_SIZE = 20 					# size of hidden layer 1
	HIDDEN2_SIZE = 20 					# size of hidden layer 2
	HIDDEN3_SIZE = 20 					# size of hidden layer 3
	EPISODES_NUM = 500	 				# number of episodes to train on. Ideally shouldn't take longer than 2000
	MAX_STEPS = 200 					# maximum number of steps in an episode 
	LEARNING_RATE = 0.001				# learning rate and other parameters for SGD/RMSProp/Adam
	MINIBATCH_SIZE = 32				# size of minibatch sampled from the experience replay
	DISCOUNT_FACTOR = 0.999				# MDP's gamma
	TARGET_UPDATE_FREQ = 100		# number of steps (not episodes) after which to update the target networks 
	LOG_DIR = './logs' 					# directory wherein logging takes place


	# Create and initialize the environment
	def __init__(self, env, seed, plot=True, replay_req=True, target_req=True):
		self.env = gym.make(env)
		assert len(self.env.observation_space.shape) == 1
		self.input_size = self.env.observation_space.shape[0]		# In case of cartpole, 4 state features
		self.output_size = self.env.action_space.n					# In case of cartpole, 2 actions (right/left)
		self.set_seed(seed) 		# Seed for all the RNG while training for reproducability
		self.plot = plot
		self.replay_req = replay_req
		self.target_req = target_req

		if  not self.replay_req:
			self.REPLAY_MEMORY_SIZE = 1
			self.MINIBATCH_SIZE = 1
		
		if not self.target_req:
			self.TARGET_UPDATE_FREQ = 1

	def set_seed(self, seed):
		self.env.seed(seed)
		self.env.action_space.seed(seed)
		np.random.seed(seed)
		random.seed(seed)
		tf.set_random_seed(seed)

	# Create the Q-network
	def initialize_network(self):
		# placeholder for the state-space input to the q-network
		self.x = tf.placeholder(tf.float32, [None, self.input_size])

		############################################################
		# Design your q-network here.
		# 
		# Add hidden layers and the output layer. For instance:
		# 
		# with tf.name_scope('output'):
		#	W_n = tf.Variable(
		# 			 tf.truncated_normal([self.HIDDEN_n-1_SIZE, self.output_size], 
		# 			 stddev=0.01), name='W_n')
		# 	b_n = tf.Variable(tf.zeros(self.output_size), name='b_n')
		# 	self.Q = tf.matmul(h_n-1, W_n) + b_n
		#
		#############################################################

		# Your code here
		with tf.name_scope('Q_network'):
			W1 = tf.Variable(tf.truncated_normal([self.input_size, self.HIDDEN1_SIZE], stddev=0.1), name='W1'),
			b1 = tf.Variable(tf.zeros(self.HIDDEN1_SIZE), name='b1'),
			W2 = tf.Variable(tf.truncated_normal([self.HIDDEN1_SIZE, self.HIDDEN2_SIZE], stddev=0.1), name='W2'),
			b2 = tf.Variable(tf.zeros(self.HIDDEN2_SIZE), name='b2'),
			W3 = tf.Variable(tf.truncated_normal([self.HIDDEN2_SIZE, self.HIDDEN3_SIZE], stddev=0.1), name='W3'),
			b3 = tf.Variable(tf.zeros(self.HIDDEN3_SIZE), name='b3'),
			W4 = tf.Variable(tf.truncated_normal([self.HIDDEN3_SIZE, self.output_size], stddev=0.1), name='W4'),
			b4 = tf.Variable(tf.zeros(self.output_size), name='b4')

			self.weights = [W1, b1, W2, b2, W3, b3, W4, b4]

			h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)
			h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
			h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
			self.Q = tf.squeeze(tf.matmul(h3, W4) + b4)

		############################################################
		# Next, compute the loss.
		#
		# First, compute the q-values. Note that you need to calculate these
		# for the actions in the (s,a,s',r) tuples from the experience replay's minibatch
		#
		# Next, compute the l2 loss between these estimated q-values and 
		# the target (which is computed using the frozen target network)
		#
		############################################################

		# Your code here
		self.action = tf.placeholder(tf.int32, [None], name='actions')
		self.one_hot_action = tf.one_hot(self.action, self.output_size, name='one_hot_actions')
		self.Q_vals = tf.reduce_sum(tf.multiply(self.Q, self.one_hot_action), axis=1)
		self.target_vals = tf.placeholder(tf.float32, [None], name='target_values')

		self.loss = tf.losses.mean_squared_error(self.Q_vals, self.target_vals)
		############################################################
		# Finally, choose a gradient descent algorithm : SGD/RMSProp/Adam. 
		#
		# For instance:
		# optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
		# global_step = tf.Variable(0, name='global_step', trainable=False)
		# self.train_op = optimizer.minimize(self.loss, global_step=global_step)
		#
		############################################################

		# Your code here
		
		optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
		global_step = tf.Variable(0, name='global_step', trainable=False)

		# gradients, variables = zip(*optimizer.compute_gradients(self.loss))
		# gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
		# self.train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
		
		self.train_op = optimizer.minimize(self.loss, global_step=global_step)

		############################################################

	def train(self, episodes_num=EPISODES_NUM):
		
		#Initialize real-time plotter
		if self.plot:
			plt.ion() ## Note this correction
			fig =plt.figure()
			ax = fig.add_subplot(111)
			ax.set_ylim(0, 200)
			ax.set_xlabel("Episodes")
			ax.set_ylabel("100 episode moving average reward")
		# Alternatively, you could use animated real-time plots from matplotlib 
		# (https://stackoverflow.com/a/24228275/3284912)
		
		# Initialize the TF session
		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())
		
		############################################################
		# Initialize other variables (like the replay memory)
		############################################################

		# Your code here
		replay_memory = []
		total_steps = 0
		self.ep_rewards = []
		self.EPSILON = self.EPSILON_START

		#Initialising Target weights
		target_weights = self.session.run(self.weights)

		############################################################
		# Main training loop
		# 
		# In each episode, 
		#	pick the action for the given state, 
		#	perform a 'step' in the environment to get the reward and next state,
		#	update the replay buffer,
		#	sample a random minibatch from the replay buffer,
		# 	perform Q-learning,
		#	update the target network, if required.
		#
		#
		#
		# You'll need to write code in various places in the following skeleton
		#
		############################################################

		for episode in range(episodes_num):
		  
			state = self.env.reset()

			############################################################
			# Episode-specific initializations go here.
			############################################################
			#
			# Your code here
			
			episode_length = 0
			episode_loss = 0
			############################################################

			while True:

				############################################################
				# Pick the next action using epsilon greedy and and execute it
				############################################################

				# Your code here
				if np.random.uniform(0, 1) < self.EPSILON:
					action = self.env.action_space.sample()
				else:
					Q_values = self.session.run(self.Q, feed_dict={self.x : state.reshape(1, state.shape[0])})
					action = np.argmax(Q_values)
					
				if self.EPSILON > self.EPSILON_MIN:
					self.EPSILON *= self.EPSILON_DECAY
	
				############################################################
				# Step in the environment. Something like: 
				# next_state, reward, done, _ = self.env.step(action)
				############################################################

				# Your code here
				next_state, reward, done, _ = self.env.step(action)
				
				episode_length += 1
				total_steps += 1

				curr_exp = [state, action, reward, next_state, done]
				
				############################################################
				# Update the (limited) replay buffer. 
				#
				# Note : when the replay buffer is full, you'll need to 
				# remove an entry to accommodate a new one.
				############################################################

				# Your code here
				if len(replay_memory) == self.REPLAY_MEMORY_SIZE:
					replay_memory.pop(0)

				replay_memory.append(curr_exp)

				############################################################
				# Sample a random minibatch and perform Q-learning (fetch max Q at s') 
				#
				# Remember, the target (r + gamma * max Q) is computed    
				# with the help of the target network.
				# Compute this target and pass it to the network for computing 
				# and minimizing the loss with the current estimates
				#
				############################################################

				# Your code here
				if len(replay_memory) >= self.MINIBATCH_SIZE:
					sampled_exps = np.array(random.sample(replay_memory, self.MINIBATCH_SIZE))
					
					states = np.stack(sampled_exps[:, 0], axis=0)
					actions = sampled_exps[:, 1]
					rewards = sampled_exps[:, 2]
					next_states = np.stack(sampled_exps[:, 3], axis=0)
					teriminal = sampled_exps[:, 4]
					
					feed = {self.x : next_states}
					feed.update(zip(self.weights, target_weights))
					
					if self.replay_req:
						Q_old = np.max(self.session.run(self.Q, feed_dict=feed), axis=1)
					else:
						Q_old = np.max(self.session.run(self.Q, feed_dict=feed))

					targets = np.where(teriminal, rewards, rewards + self.DISCOUNT_FACTOR * Q_old)
					
					_, loss_val = self.session.run([self.train_op, self.loss], 
													feed_dict={self.x : states,
															   self.target_vals : targets,
															   self.action : actions})
					episode_loss += loss_val
				############################################################
				# Update target weights. 
				#
				# Something along the lines of:
				# if total_steps % self.TARGET_UPDATE_FREQ == 0:
				# 	target_weights = self.session.run(self.weights)
				############################################################

				# Your code here
				if total_steps % self.TARGET_UPDATE_FREQ == 0:
					target_weights = self.session.run(self.weights)

				############################################################
				# Break out of the loop if the episode ends
				#
				# Something like:
				# if done or (episode_length == self.MAX_STEPS):
				# 	break
				#
				############################################################

				# Your code here
				if done or (episode_length == self.MAX_STEPS):
					break
				else:
					state = next_state

			############################################################
			# Logging. 
			#
			# Very important. This is what gives an idea of how good the current
			# experiment is, and if one should terminate and re-run with new parameters
			# The earlier you learn how to read and visualize experiment logs quickly,
			# the faster you'll be able to prototype and learn.
			#
			# Use any debugging information you think you need.
			# For instance :

			self.ep_rewards.append(episode_length)
			running_avg_reward = np.mean(self.ep_rewards[-100:])
			
			print("Training: Episode = %d, Length = %d, 100_len_average = %d, Global step = %d" \
				 % (episode, episode_length, running_avg_reward, total_steps))	
			
			if self.plot:	
				plt.scatter(episode, running_avg_reward, color='blue', linewidths=0.25)
				plt.show()
				plt.pause(0.0001)

	# Simple function to visually 'test' a policy
	def playPolicy(self, render=True):
		
		done = False
		steps = 0
		state = self.env.reset()
		if render:
			self.env.render()
		# we assume the CartPole task to be solved if the pole remains upright for 200 steps
		while not done and steps < 200: 
			if render:	
				self.env.render()				
			q_vals = self.session.run(self.Q, feed_dict={self.x: [state]})
			action = q_vals.argmax()
			state, _, done, _ = self.env.step(action)
			steps += 1
		
		return steps


if __name__ == '__main__':
	os.makedirs("results/learning_curve_best/", exist_ok=True)
	# Create and initialize the model
	seeds = [1, 345, 50, 278]
	for run, seed in enumerate(seeds):
		dqn = DQN('CartPole-v0', seed, plot=True)
		dqn.initialize_network()

		print("\nStarting training...\n")
		dqn.train()
		print("\nFinished training...\nCheck out some demonstrations\n")

		# Visualize the learned behaviour for a few episodes
		results = []
		for i in range(20):
			episode_length = dqn.playPolicy(render=False)
			print("Test steps = ", episode_length)
			results.append(episode_length)
		print("Mean steps = ", sum(results) / len(results))	

		np.save("results/learning_curve_best/run_" + str(run) + ".npy", dqn.ep_rewards)
		np.save("results/learning_curve_best/test_run_" + str(run) + ".npy", results)

	print("\nFinished.")
	print("\nCiao, and hasta la vista...\n")