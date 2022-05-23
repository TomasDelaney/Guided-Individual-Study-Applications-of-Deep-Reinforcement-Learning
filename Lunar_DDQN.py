import gym
from gym import wrappers  # records the episodes just an extra
import matplotlib.pyplot as plt
import numpy as np
from ddqn_keras import DDQNAgent
import time


if __name__ == '__main__':
    start_time = time.time()
    env = gym.make('LunarLander-v2')
    ddqn_agent = DDQNAgent(alpha=0.005, gamma=0.99, n_actions=env.action_space.n, epsilon=1.0, fname='ddqn_model_v1.h5',
                           batch_size=64, input_dims=8)
    training = False
    evaluate = True
    n_games = 5
    episode_scores = []
    eps_history = []
    length_of_games = []

    # env = wrappers.Monitor(env, 'tmp/lunar-lander', video_callable=lambda episode_id: True, force=True)
    if training:
        for i in range(1, n_games+1):
            done = False
            score = 0
            episode_length = 0
            observation = env.reset()
            while not done:
                action = ddqn_agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                score += reward
                episode_length += 1
                ddqn_agent.remember(observation, action, reward, observation_, done)
                observation = observation_
                ddqn_agent.learn()

            # finished the episode
            ddqn_agent.decrease_epsilon()

            # append to the lists which we plot + epsilon list
            eps_history.append(ddqn_agent.epsilon)
            episode_scores.append(score)
            length_of_games.append(episode_length)

            avg_score = np.mean(episode_scores)
            print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)
            print('episode length: {}'.format(episode_length))

            # saving the model
            if i % 10 == 0:
                ddqn_agent.save_model()
                print("Saved after {} iterations.".format(i))

        ddqn_agent.plot_training_data(episode_scores, length_of_games, n_games, start_time)
        print(eps_history)

    if evaluate:
        ddqn_agent.load_model()  # make sure the hyper params match in the initializing
        for i in range(n_games):
            done = False
            score = 0
            episode_length = 0
            observation = env.reset()

            while not done:
                env.render()
                action = ddqn_agent.evaluate(observation)
                observation_, reward, done, info = env.step(action)
                score += reward
                ddqn_agent.remember(observation, action, reward, observation_, done)
                observation = observation_
                episode_length += 1

            episode_scores.append(score)
            length_of_games.append(episode_length)
            avg_score = np.mean(episode_scores)
            print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)
            print('episode length: {}'.format(episode_length))

        # plotting the results
        x = [i + 1 for i in range(n_games)]
        plt.plot(x, episode_scores, label='Score of the agent')
        plt.title("Evaluation of the agent")
        plt.xlabel('Episodes')
        plt.ylabel('Scores')
        plt.legend()
        plt.show()
