import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.optimizers
from keras import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
from tensorflow.keras.layers import LeakyReLU
import matplotlib.pyplot as plt
import time


class ReplayBuffer(object):  # agents memory
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size  # max size of memory
        self.input_shape = input_shape  # the input of the environment
        self.discrete = discrete  # the action space discrete or continuous
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = action
        else:
            self.action_memory[index] = action
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)  # so you do not read over the end of the array
        batch = np.random.choice(max_mem, batch_size)  # get a random -> we do not select the same over and over again

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
        # training not to be sequential-> correlation-> you get caught up on one part of the training


def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = keras.Sequential()
    model.add(keras.layers.Dense(fc1_dims, input_shape=(input_dims,)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(keras.layers.Dense(fc2_dims))
    model.add(LeakyReLU(alpha=0.1))
    model.add(keras.layers.Dense(n_actions, activation=None))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')

    return model


class DDQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dims, fname, epsilon_dec=0.02,
                 epsilon_end=0.01, mem_size=20000, replace_target=1000):
        # two networks one chooses the action another action evaluates that action
        # only train the target network, replace the weight of the target network every 100 games
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, True)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256)
        self.q_target = build_dqn(alpha, n_actions, input_dims, 256, 256)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = \
                                            self.memory.sample_buffer(self.batch_size)

            q_next = self.q_target.predict(new_state)
            q_eval = self.q_eval.predict(new_state)

            q_pred = self.q_eval.predict(state)

            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action[:, 0]] = reward +\
                                                    self.gamma * q_next[batch_index, max_actions.astype(int)] * done

            _ = self.q_eval.fit(state, q_target, verbose=0)

            if self.memory.mem_cntr % self.replace_target == 0:
                self.update_network_parameters()  # look up mi ez
                print("updated the weights")

    def decrease_epsilon(self):
        # change instead of self.epsilon * self.epsilon_dec --> self.epsilon - self.epsilon_dec
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def update_network_parameters(self):
        self.q_target.set_weights(self.q_eval.get_weights())

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)

        # if in evaluation mode:
        if self.epsilon <= self.epsilon_min:
            self.update_network_parameters()

    def evaluate(self, state):
        state = state[np.newaxis, :]
        actions = self.q_eval.predict(state)
        action = np.argmax(actions)

        return action

    @staticmethod
    def plot_training_data(self, training_scores, length_of_training_episodes, n_games, start_time):
        x = [i + 1 for i in range(n_games)]
        plt.figure(1)
        plt.plot(x, training_scores, label="Training accuracy")
        plt.title('Scores of the agent in each episode')
        plt.xlabel('Number of episodes')
        plt.ylabel('Achieved score')
        plt.legend()
        plt.show()

        # plot the length of the episodes
        plt.figure(2)
        plt.plot(x, length_of_training_episodes, label="Length of training episode")
        # plt.plot(x, length_of_test_episodes, label="Length of testing episode")
        plt.title('Episode lengths of the agent')
        plt.xlabel('Number of episodes')
        plt.ylabel('Length of the episodes')
        plt.show()

        print('\n', 'The highest received training award of the agent: {}'.format(np.max(training_scores)))
        print('\n', 'The average episode length of training the agent: {}'.format(np.mean(length_of_training_episodes)))
        print('\n', "--- %s seconds ---" % (time.time() - start_time))