import tensorflow as tf
import rlcard

from rlcard.agents import RandomAgent
from pettingzoo.classic import texas_holdem_no_limit_v6

import numpy as np
from pettingzoo.classic import texas_holdem_no_limit_v6
# from supersuit import pad_observations_v0, pad_action_space_v0

# ---------- 1.  initialise a 4-player table ----------
env = texas_holdem_no_limit_v6.env(
        num_players=4,         
        render_mode="human")       # change to "human" for ascii output
env.reset(seed=41)

print(env.action_spaces)


# ---------- 2.  simple fixed-strategy bots ----------
def random_bot(obs, mask):
    return np.random.choice(np.flatnonzero(mask))

def tight_bot(obs, mask):
    # Example: Fold pre-flop unless both hole cards are same rank
    cards = np.flatnonzero(obs["observation"][:52])
    if len(cards) == 2 and (cards[0] % 13 == cards[1] % 13):
        return 1 if 1 in np.flatnonzero(mask) else np.flatnonzero(mask)[0]
    return 0


bot_policy = {
    "player_1": random_bot,
    "player_2": tight_bot,
    "player_3": random_bot,
    # player_0: human or learner
}


for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    print(env.last())
    
    if termination or truncation:
        action = None
    else:
        mask = observation["action_mask"]
        if agent == "player_0":
            # For human: show mask and prompt for input
            print(f"\n{agent} legal actions: {np.flatnonzero(mask)} (0=Fold, 1=Chk/Call, 2=½Pot, 3=Pot, 4=All-in)")
            while True:
                try:
                    a = int(input(f"{agent}, choose action id ➜ "))
                    if a in np.flatnonzero(mask):
                        action = a
                        break
                except Exception:
                    pass
        else:
            # Use bot policy
            action = bot_policy[agent](observation, mask)

            print(f"{agent} chose action {action}")

    env.step(action)

env.close()


if __name__ == "__main__":
    pass
    # # Print TensorFlow version
    # print("TensorFlow version:", tf.__version__)

    # # List available physical devices
    # print("Available physical devices:")
    # for device in tf.config.list_physical_devices():
    #     print(device)

    # # Check if GPU is available
    # print("Is GPU available:", tf.config.list_physical_devices('GPU'))

