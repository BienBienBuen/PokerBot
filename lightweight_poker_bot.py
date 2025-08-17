"""
A lightweight implementation of a poker bot based on the "Deep Counterfactual
Regret Minimization" paper.

This bot learns to play Kuhn Poker, a simplified 3-card poker game.

Key Components from the paper:
1.  **Game Environment**: Kuhn Poker is used as the testbed.
2.  **Neural Networks**: Simple MLPs are used instead of large tables to store:
    - V(I): The cumulative regrets for taking actions in an infoset I.
    - Π(I): The average strategy for an infoset I.
3.  **Monte Carlo CFR (MCCFR)**: The bot uses an "external sampling" approach
    to traverse the game tree, which is efficient for large games.
4.  **Memory Buffers**: Two replay buffers store training data:
    - Advantage Memory (Mv): Stores (infoset, instantaneous_regret) tuples.
    - Strategy Memory (MΠ): Stores (infoset, strategy_played) tuples.
    These use reservoir sampling to keep a fixed-size sample of past experiences.
5.  **Training Loop**: The bot iteratively plays against itself. In each iteration,
    it traverses the game tree, collects data, and then trains its networks
    on batches of data sampled from the memory buffers.
"""
import random
import numpy as np
import torch
import torch.nn as nn
import collections
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('poker_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# --- 1. Kuhn Poker Game Environment ---

class Kuhn_Poker:
    """
    Represents the Kuhn Poker game environment.
    - 3 cards (J, Q, K represented as 0, 1, 2)
    - 2 players
    - 1 chip ante
    - 1 bet size
    - Actions: 0 (pass/fold), 1 (bet/call)
    """
    def __init__(self):
        self.cards = list(range(3))
        # Non-terminal histories where a player has to act
        self.history_map = {
            "": 0, "p": 1, "b": 2, "pb": 3
        }
        self.num_actions = 2

    def is_terminal(self, history):
        # pp: pass, pass
        # bp: bet, pass (fold)
        # pbp: pass, bet, pass (fold)
        # bb: bet, bet (call)
        # pbb: pass, bet, bet (call)
        return history in ["pp", "bp", "pbp", "bb", "pbb"]

    def get_payoff(self, history, hands):
        """Calculates payoff for player 0."""
        if history == "bp":  # Player 1 bets, Player 2 folds
            return 1
        if history == "pbp": # Player 1 passes, Player 2 bets, Player 1 folds
            return -1

        # Reached a showdown
        p0_card, p1_card = hands
        payoff = 1 if history == "pp" else 2

        if p0_card > p1_card:
            return payoff
        else:
            return -payoff

    def get_infoset(self, card, history):
        """Converts card and history into a one-hot encoded vector."""
        card_vec = np.zeros(len(self.cards))
        card_vec[card] = 1
        
        history_vec = np.zeros(len(self.history_map))
        if history in self.history_map:
            history_vec[self.history_map[history]] = 1
            
        return torch.tensor(np.concatenate((card_vec, history_vec)), dtype=torch.float32, device=device)


# --- 2. Neural Network Model ---

class Poker_MLP(nn.Module):
    """A simple Multi-Layer Perceptron for value and policy networks."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


# --- 3. Replay Buffer with Reservoir Sampling ---

class ReservoirBuffer:
    """Stores experiences using reservoir sampling."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, data):
        if len(self.memory) < self.capacity:
            self.memory.append(data)
        else:
            # Reservoir sampling
            i = random.randint(0, self.position)
            if i < self.capacity:
                self.memory[i] = data
        self.position += 1

    def sample(self, batch_size):
        return random.sample(self.memory, min(len(self.memory), batch_size))

    def __len__(self):
        return len(self.memory)


# --- 4. Deep CFR Agent ---

class Deep_CFR_Agent:
    def __init__(self, game, hidden_size=128, memory_size=100000, learning_rate=0.001):
        # Add validation
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if memory_size <= 0:
            raise ValueError("memory_size must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
            
        self.game = game
        self.infoset_size = len(game.cards) + len(game.history_map)
        self.num_actions = game.num_actions
        self.learning_rate = learning_rate

        # Create value networks (one per player) and a single policy network
        self.value_nets = [
            Poker_MLP(self.infoset_size, hidden_size, self.num_actions).to(device) for _ in range(2)
        ]
        self.policy_net = Poker_MLP(self.infoset_size, hidden_size, self.num_actions).to(device)

        # Create memory buffers
        self.value_memories = [ReservoirBuffer(memory_size) for _ in range(2)]
        self.policy_memory = ReservoirBuffer(memory_size)

        # Optimizers with configurable learning rate
        self.value_optimizers = [
            torch.optim.Adam(net.parameters(), lr=learning_rate) for net in self.value_nets
        ]
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def _get_strategy_from_regrets(self, regret_values):
        """Applies regret matching to get a strategy."""
        positive_regrets = torch.clamp(regret_values, min=0)
        sum_positive_regrets = torch.sum(positive_regrets)
        if sum_positive_regrets > 0:
            return positive_regrets / sum_positive_regrets
        else:
            # Default to uniform random strategy if no positive regrets
            return torch.full((self.num_actions,), 1.0 / self.num_actions, device=device)

    def _traverse(self, history, hands, traverser, iteration):
        """
        Recursive traversal of the game tree using external sampling.
        This corresponds to Algorithm 2 in the Deep CFR paper.
        """
        player = len(history) % 2

        if self.game.is_terminal(history):
            payoff_p0 = self.game.get_payoff(history, hands)
            # The value of the game is from the traverser's perspective
            return payoff_p0 if traverser == 0 else -payoff_p0

        infoset = self.game.get_infoset(hands[player], history)

        # Get strategy for the current node
        value_net = self.value_nets[player]
        with torch.no_grad():
            # Detach to prevent gradients from flowing into the value network during traversal
            regret_values = value_net(infoset).detach()
        strategy = self._get_strategy_from_regrets(regret_values)

        if player == traverser:
            # --- Traverser's turn ---
            # Explore all actions to calculate counterfactual values
            action_values = torch.zeros(self.num_actions, device=device)
            for action in range(self.num_actions):
                next_history = history + ('p' if action == 0 else 'b')
                action_values[action] = self._traverse(next_history, hands, traverser, iteration)

            # Calculate node value and instantaneous regrets
            node_value = torch.dot(action_values, strategy)
            instant_regrets = action_values - node_value

            # Store experience in the value memory for the traverser
            # The paper weights by iteration number 't' (Linear CFR)
            self.value_memories[player].push((iteration, infoset, instant_regrets))
            
            return node_value
        else:
            # --- Opponent's or Chance's turn ---
            # Store the current strategy for training the average policy network
            self.policy_memory.push((iteration, infoset, strategy.detach()))

            # Sample one action and proceed
            action = torch.multinomial(strategy, 1).item()
            next_history = history + ('p' if action == 0 else 'b')
            return self._traverse(next_history, hands, traverser, iteration)

    def _train_net(self, memory, network, optimizer, is_value_net):
        """Trains either a value network or the policy network."""
        if len(memory) < 1000: # Don't train until we have some data
            return 0.0  # Return loss for monitoring
        
        try:
            batch = memory.sample(min(1024, len(memory)))  # Don't sample more than available
            
            # Unpack batch and create tensors
            weights, infosets, targets = zip(*batch)
            weights = torch.tensor(weights, dtype=torch.float32, device=device).view(-1, 1)
            infosets = torch.stack(infosets).to(device)
            targets = torch.stack(targets).to(device)

            # Add epsilon to prevent division by zero
            weights = weights + 1e-8
            weights /= weights.sum()

            # Forward pass
            predictions = network(infosets)

            # Calculate loss (weighted MSE)
            if not is_value_net:
                # For policy net, use cross-entropy loss instead of MSE on probabilities
                loss = -(weights * targets * torch.log(torch.softmax(predictions, dim=1) + 1e-8)).sum(dim=1).mean()
            else:
                loss = (weights * (predictions - targets)**2).sum(dim=1).mean()

            # Add gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return float('inf')

    def train(self, num_iterations=200, num_traversals_per_iter=200, checkpoint_freq=50):
        """
        Main training loop. Corresponds to Algorithm 1 in the Deep CFR paper.
        """
        logger.info("Starting training...")
        logger.info(f"Iterations: {num_iterations}, Traversals per iter: {num_traversals_per_iter}")
        
        training_losses = {'value': [], 'policy': []}
        
        try:
            for t in range(1, num_iterations + 1):
                iter_value_losses = []
                
                for traverser in range(2):  # Alternate traverser
                    # Re-initialize value network from scratch (as per paper)
                    self.value_nets[traverser] = Poker_MLP(self.infoset_size, 128, self.num_actions).to(device)
                    self.value_optimizers[traverser] = torch.optim.Adam(
                        self.value_nets[traverser].parameters(), lr=self.learning_rate
                    )

                    for _ in range(num_traversals_per_iter):
                        # 1. Sample a game state (deal cards)
                        hands = random.sample(self.game.cards, 2)
                        # 2. Run a traversal
                        self._traverse(history="", hands=hands, traverser=traverser, iteration=t)

                    # 3. Train the value network for the current traverser
                    loss = self._train_net(
                        self.value_memories[traverser], 
                        self.value_nets[traverser], 
                        self.value_optimizers[traverser], 
                        is_value_net=True
                    )
                    iter_value_losses.append(loss)

                # 4. Train the average policy network
                policy_loss = self._train_net(
                    self.policy_memory, self.policy_net, self.policy_optimizer, is_value_net=False
                )

                # Store losses
                training_losses['value'].append(np.mean(iter_value_losses))
                training_losses['policy'].append(policy_loss)

                if t % 10 == 0:
                    logger.info(f"Iteration {t}/{num_iterations} - Value Loss: {np.mean(iter_value_losses):.4f}, Policy Loss: {policy_loss:.4f}")
                    self.evaluate()

                # Checkpoint saving
                if t % checkpoint_freq == 0:
                    self.save_checkpoint(f"checkpoint_iter_{t}.pt", t, training_losses)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        logger.info("Training finished.")
        return training_losses

    def save_checkpoint(self, filename, iteration, losses):
        """Save model checkpoint."""
        try:
            checkpoint = {
                'iteration': iteration,
                'policy_net_state': self.policy_net.state_dict(),
                'policy_optimizer_state': self.policy_optimizer.state_dict(),
                'losses': losses,
                'hyperparameters': {
                    'hidden_size': 128,
                    'memory_size': len(self.value_memories[0].memory),
                    'learning_rate': self.learning_rate
                }
            }
            torch.save(checkpoint, filename)
            logger.info(f"Checkpoint saved: {filename}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(filename, map_location=device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state'])
            logger.info(f"Checkpoint loaded: {filename}")
            return checkpoint['iteration'], checkpoint['losses']
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return 0, {'value': [], 'policy': []}

    def get_action(self, card, history, temperature=0.0):
        """Get an action from the trained average policy network."""
        if not isinstance(card, int) or card not in self.game.cards:
            raise ValueError(f"Invalid card: {card}")
        if not isinstance(history, str):
            raise ValueError("History must be a string")
            
        infoset = self.game.get_infoset(card, history)
        with torch.no_grad():
            logits = self.policy_net(infoset)
            if temperature > 0:
                logits = logits / temperature
            strategy = nn.functional.softmax(logits, dim=0)
        return strategy

    def evaluate(self):
        """
        Prints the learned strategy for key Player 1 infosets and compares
        them to the known Nash Equilibrium for Kuhn Poker.
        """
        logger.info("--- Evaluating Player 1 Strategy ---")
        # Nash equilibrium probabilities for betting with a given card at the start.
        nash_bet_probs = {
            0: 0.0,    # Jack: Never bet
            1: 1/3,    # Queen: Bet with probability 1/3
            2: 1.0     # King: Always bet
        }
        card_names = {0: "Jack", 1: "Queen", 2: "King"}

        logger.info(f"{'Card':<8} | {'Learned Bet %':<15} | {'Nash Bet %':<12}")
        logger.info("-" * 45)

        for card in self.game.cards:
            # We are evaluating Player 1's opening move, so history is empty
            strategy = self.get_action(card, "")
            learned_bet_prob = strategy[1].item() # Prob of action 1 (bet)
            nash_prob = nash_bet_probs[card]

            logger.info(f"{card_names[card]:<8} | {learned_bet_prob:15.2%} | {nash_prob:12.2%}")
        logger.info("-" * 45)

# Add configuration management
class Config:
    def __init__(self):
        self.NUM_ITERATIONS = int(os.getenv('NUM_ITERATIONS', '200'))
        self.TRAVERSALS_PER_ITER = int(os.getenv('TRAVERSALS_PER_ITER', '500'))
        self.HIDDEN_SIZE = int(os.getenv('HIDDEN_SIZE', '128'))
        self.MEMORY_SIZE = int(os.getenv('MEMORY_SIZE', '1000000'))
        self.LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.001'))
        self.CHECKPOINT_FREQ = int(os.getenv('CHECKPOINT_FREQ', '50'))

if __name__ == '__main__':
    # --- 5. Training and Evaluation ---
    try:
        config = Config()
        
        # Set seeds for reproducibility
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Initialize game and agent
        kuhn_game = Kuhn_Poker()
        agent = Deep_CFR_Agent(
            game=kuhn_game, 
            hidden_size=config.HIDDEN_SIZE, 
            memory_size=config.MEMORY_SIZE,
            learning_rate=config.LEARNING_RATE
        )

        # Train the agent
        losses = agent.train(
            num_iterations=config.NUM_ITERATIONS, 
            num_traversals_per_iter=config.TRAVERSALS_PER_ITER,
            checkpoint_freq=config.CHECKPOINT_FREQ
        )

        # Save final model
        agent.save_checkpoint('final_model.pt', config.NUM_ITERATIONS, losses)

        # Final evaluation
        logger.info("Final evaluation:")
        agent.evaluate()
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise