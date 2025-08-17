"""
1 vs 1 No limit Texas Hold'em, testing of game tree manipulation to carry out counterfactual regret minimization. 
"""

import tensorflow as tfxs
from rlcard import make
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed, tournament, Logger
import numpy as np

env = make(
    'no-limit-holdem',
    config={
        'game_num_players': 2,
        'seed': 0,
        'allow_step_back': True
    }
)

random_agent_1 = RandomAgent(num_actions=env.num_actions) 
random_agent_2 = RandomAgent(num_actions=env.num_actions)
env.set_agents([random_agent_1, random_agent_2]) 

total_trajectories = []
payoff_list = []

for episode in range(10):
    trajectories, payoffs = env.run(is_training=False)
    total_trajectories.append(trajectories)
    payoff_list.append(payoffs)

final_traj = total_trajectories[-1]
for rounds in final_traj[0]:
    if type(rounds) != np.int64:
        print("==raw_obs==\n")
        print(rounds['raw_obs'])
        print("\n")
        print("==action_record==\n")
        print(rounds['action_record'])
        print("\n")
    #print(rounds)
    #print(rounds['action_record'])
