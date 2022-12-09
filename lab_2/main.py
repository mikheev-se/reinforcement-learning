import gym
import matplotlib.pyplot as plt
import numpy as np

from gym.envs.toy_text.frozen_lake import generate_random_map


env = gym.make('FrozenLake-v1',
               desc=generate_random_map(size=5), is_slippery=False)


def play_game(env, policy, t_max, render=False):
    states, actions = [], []
    sum_reward = 0.
    valid_actions = np.arange(start=env.action_space.start,
                              stop=env.action_space.n + env.action_space.start,
                              step=1)

    state = env.reset()

    for t in range(t_max):
        # выберите с помощью numpy действие (столбец) для данного состояния (строка)
        # при этом выбор должен осуществляться на основе вероятностей (значения строки матрицы)
        action = np.random.choice(a=valid_actions, p=policy[state])
        new_state, reward, done, i = env.step(action)

        if render:
            env.render()

        states.append(state)
        actions.append(action)
        sum_reward += reward
        state = new_state

        if done:
            break

    return states, actions, sum_reward


policy = np.array([[1/env.action_space.n for _ in range(env.action_space.n)]
                  for _ in range(env.observation_space.n)])

rewards = np.array([play_game(env, policy, 1000)[2] for _ in range(200)])
plt.hist(rewards)


def select_elites(states_lists, actions_lists, rewards):
    # выберите выигрышные стратегии (верните списки наблюдений и действий для выигрышных стратегий)
    # для данного окружения, при выигрышной стратегии, суммарная награда за игру будет равна 1
    elite_ids = np.nonzero(rewards)

    elite_states = states_lists[elite_ids]
    elite_actions = actions_lists[elite_ids]

    # зная, какие стратегии выигрышные, сформируйте списки соответствующих состояний и действий
    # получатся пары (состояние, действие)

    return elite_states, elite_actions


def update_policy(env, elite_states, elite_actions):
    # создайте матрицу нулей, в которой количество строк - это количество состояний,
    new_policy = np.zeros((env.observation_space.n, env.action_space.n))
    # а количество столбцов - количество действий

    # увеличьте на 1 значения ячеек в new_policy для каждой пары (состояние, действие)
    for elite_state, elite_action in zip(elite_states, elite_actions):
        new_policy[elite_state,
                   elite_action] = new_policy[elite_state, elite_action] + 1

    sum_of_rows = new_policy.sum(axis=1)
    for idx, row_sum in enumerate(sum_of_rows):
        if row_sum == 0:
            # сделайте равновероятными все действия для данного состояния
            new_policy[idx] = 1/env.action_space.n
        else:
            # поделите все элементы данной строки на row_sum
            new_policy[idx] = new_policy[idx]/row_sum

    return new_policy


# здесь можно поиграть с гиперпараметрами
t_max = 10000
n_games = 150
learning_rate = 0.25
epochs = 250

for i in range(epochs):
    games = [play_game(env, policy, t_max)
             for _ in range(n_games)]    # сыграйте n_games игр
    states_lists, actions_lists, rewards_lists = zip(*games)
    elite_states, elite_actions = select_elites(np.array(states_lists), np.array(
        actions_lists), np.array(rewards_lists))   # выберите элитные состояния и действия
    # получите промежуточную стратегию
    new_policy = update_policy(env, elite_states, elite_actions)
    policy = learning_rate * new_policy + (1 - learning_rate) * policy

state = env.reset()

while True:
    state, reward, done, i = env.step(np.random.choice(a=np.arange(start=env.action_space.start,
                                                                   stop=env.action_space.n + env.action_space.start,
                                                                   step=1), p=policy[state]))

    env.render()

    if done:
        break
