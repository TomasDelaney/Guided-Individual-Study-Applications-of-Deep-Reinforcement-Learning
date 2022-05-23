import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()

        self.fc1 = Dense(fc1_dims, activation=keras.layers.LeakyReLU(alpha=0.01))
        self.fc2 = Dense(fc2_dims, activation=keras.layers.LeakyReLU(alpha=0.01))
        self.fc3 = Dense(n_actions, activation='softmax')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()

        self.fc1 = Dense(fc1_dims, activation=keras.layers.LeakyReLU(alpha=0.01))
        self.fc2 = Dense(fc2_dims, activation=keras.layers.LeakyReLU(alpha=0.01))
        self.fc3 = Dense(1, activation=None)  # actual value of the state according to the Deep neural network

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
