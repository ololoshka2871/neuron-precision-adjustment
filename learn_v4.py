#!/usr/bin/env python


import os
import pickle

import numpy as np

from keras.optimizers import Adam

import gym
import gym_quarz

from controllers.controller_v4 import NNController


#def build_agent(model, actions):
#    policy = BoltzmannQPolicy()
#    memory = SequentialMemory(limit=50000, window_length=1)
#    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
#                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
#    return dqn
#
#
#def train(q, q_target, replay_buffer, optimizer, batch_size, gamma, updates_number=10):
#    """
#    Обучение сети
#    :param q: Q-policy Сеть
#    :param q_target: Q-target Сеть
#    :param replay_buffer: Буфер для хранения истории
#    :param optimizer: Оптимизатор
#    :param batch_size: Размер мини-батча
#    :param gamma: Коэффициент дисконтирования
#    :param updates_number: Количество обновлений сети
#    """
#    for i in range(updates_number):
#        # Молучаем batch_size прошлых действий из буфера
#        state, action, reward, next_state, done = replay_buffer.sample(
#            batch_size)
#
#        # Получить полезность, для каждого действия из сети
#        q_out = q_network.predict(state)
#        q_a = q_out.gather(1, action)  # ?
#
#        # Получаем значение максимальной полезности сети и считаем target
#        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
#        target = reward + gamma * max_q_prime * done
#
#        # Считаем ошибку
#        loss = F.smooth_l1_loss(q_a, target.detach())
#
#        # Обновляем сеть
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#
#
def learn_main(max_iterations: int,
               checkpoint_file: str,
               batch_size: int,
               lerning_rate=0.001,
               gamma=0.99,
               epsilon=0.8,
               buffer_max_size=10000,
               target_update_interval=10,
               replay_buffer_start_size=1000,
               ) -> None:
    #try:
    #    with open(checkpoint_file, "rb") as cp_file:
    #        cp = pickle.load(cp_file)  # type: ignore
    #except FileNotFoundError:
    #    # Start a new learning process
    #    max_steps = 200
    #    pass

    env = gym.make("gym_quarz/QuartzEnv-v3")
    actions: int = env.action_space.n  # type: ignore
    states: int = env.observation_space.shape[0] # type: ignore
    
    dqn = NNController(states, actions)

    dqn.compile(Adam(lr=1e-3), metrics=['mse'])
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

    scores = dqn.test(env, nb_episodes=3, visualize=True)
    print(np.mean(scores.history['episode_reward']))

    env.close()


if __name__ == '__main__':
    import argparse

    # parse argumants
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, help='Max iterations', default=0)
    parser.add_argument(
        'file', type=str, help='Simulation history file', nargs='?', default='learn_v4.ckl')
    args = parser.parse_args()

    learn_main(args.m,
               checkpoint_file=args.file,
               lerning_rate=0.0005,
               gamma=0.98,
               buffer_max_size=50000,
               batch_size=32,
               target_update_interval=10,
               replay_buffer_start_size=2000,
               )
