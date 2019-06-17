"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import random
import numpy as np
from maze_env import env
from RL_brain import DeepQNetwork

RL = DeepQNetwork(n_actions=3, n_features=1, learning_rate=0.001, e_greedy=1,
                  replace_target_iter=20, memory_size=3000,
                  e_greedy_increment=0.05, output_graph=True)

total_steps = 0
oldstate = 1.4

action = 0
observation = np.array([oldstate])

ep_r = 0

Env = env()

def action2press_coefficient(oldstate, action):
    # Gap =
    observation = oldstate
    # 调低力度
    if action == 0:
        observation = observation - 0.02
    # 调高力度
    elif action == 2:
        observation = observation + 0.02
    else:
        pass
    return observation

cnt = 1
while True:
    # 重置的意思
    # env.render()
    action = RL.choose_action(observation)
    observation_ = action2press_coefficient(observation, action)
    reward = Env.step(observation_, action)

    print('cnt: ', cnt)
    cnt = cnt + 1
    print('action: ', action)
    print ('old observation:', observation,', new observation :', observation_)
    print('reward: ', reward)

    RL.store_transition(observation, action, reward, observation_)

    if total_steps > 10:
        RL.learn()

    ep_r += reward

    observation = observation_
    total_steps += 1

# RL.plot_cost()




























