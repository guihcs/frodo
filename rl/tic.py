import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

# --- Tic Tac Toe Environment ---
class TicTacToeGame:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)  # 0 = empty, 1 = player 1, -1 = player 2
        self.current_player = 1

    def get_legal_actions(self):
        # Returns list of (i, j) for empty positions.
        return list(zip(*np.where(self.board == 0)))

    def make_move(self, action):
        i, j = action
        if self.board[i, j] != 0:
            raise ValueError("Invalid move!")
        self.board[i, j] = self.current_player
        self.current_player *= -1

    def is_terminal(self):
        return self.get_winner() is not None or len(self.get_legal_actions()) == 0

    def get_winner(self):
        # Check rows, columns and diagonals
        for player in [1, -1]:
            if any(np.all(self.board[row, :] == player) for row in range(3)):
                return player
            if any(np.all(self.board[:, col] == player) for col in range(3)):
                return player
            if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
                return player
        # Draw if board full and no winner
        if len(self.get_legal_actions()) == 0:
            return 0  # Draw
        return None

    def clone(self):
        # Create a deep copy of the game state.
        new_game = TicTacToeGame()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        return new_game

    def get_state(self):
        # Returns a tensor or np.array representation of the state.
        # Here, we use the board from the perspective of the current player.
        return self.current_player * self.board.flatten()

# --- Neural Network ---
class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 64)
        # Policy head: 9 outputs for each board position.
        self.policy_head = nn.Linear(64, 9)
        # Value head: single output.
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        # x should be a tensor of shape [batch, 9]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Policy: use log softmax for numerical stability
        policy = torch.log_softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value

# --- MCTS Implementation ---
class MCTSNode:
    def __init__(self, game_state, parent=None):
        self.game_state = game_state  # instance of TicTacToeGame
        self.parent = parent
        self.children = {}  # key: action, value: MCTSNode
        self.N = {}         # visit counts for each action
        self.W = {}         # total value for each action
        self.P = {}         # prior probability for each action
        self.is_expanded = False

    def expand(self, net):
        # Expand node using the network predictions.
        state_tensor = torch.tensor(self.game_state.get_state(), dtype=torch.float32).unsqueeze(0)
        log_policy, value = net(state_tensor)
        policy = torch.exp(log_policy).detach().numpy().flatten()

        legal_actions = self.game_state.get_legal_actions()
        # Only keep legal moves.
        policy_mask = np.zeros(9)
        for (i, j) in legal_actions:
            policy_mask[i * 3 + j] = 1

        policy = policy * policy_mask
        if policy.sum() > 0:
            policy /= policy.sum()
        else:
            # If network gives zero probability to all legal moves, assign uniform.
            policy = policy_mask / policy_mask.sum()

        for (i, j) in legal_actions:
            a = i * 3 + j
            self.P[a] = policy[a]
            self.N[a] = 0
            self.W[a] = 0.0

        self.is_expanded = True
        return value.item()

def mcts_search(root, net, num_simulations=50, c_puct=1.0):
    # Perform num_simulations of MCTS starting from root.
    for _ in range(num_simulations):
        node = root
        path = []
        # Selection
        while node.is_expanded and not node.game_state.is_terminal():
            best_score = -float('inf')
            best_action = None
            total_N = sum(node.N.values())
            for a in node.P.keys():
                # UCB score
                q = node.W[a] / (node.N[a] + 1e-8)
                u = c_puct * node.P[a] * np.sqrt(total_N) / (1 + node.N[a])
                score = q + u
                if score > best_score:
                    best_score = score
                    best_action = a
            path.append((node, best_action))
            # Take the action and move to the child node.
            i, j = divmod(best_action, 3)
            next_state = node.game_state.clone()
            next_state.make_move((i, j))
            if best_action not in node.children:
                node.children[best_action] = MCTSNode(next_state, parent=node)
            node = node.children[best_action]

        # Evaluation
        if node.game_state.is_terminal():
            winner = node.game_state.get_winner()
            if winner == 0:
                value = 0
            else:
                # value from the perspective of the player who just moved at parent node.
                value = 1 if winner == node.game_state.current_player * -1 else -1
        else:
            value = node.expand(net)

        # Backpropagation
        for parent, action in reversed(path):
            parent.N[action] += 1
            parent.W[action] += value
            value = -value  # switch perspective

    # After simulations, create improved policy.
    counts = np.array([root.N.get(a, 0) for a in range(9)])
    if counts.sum() > 0:
        pi = counts / counts.sum()
    else:
        pi = np.ones(9) / 9
    return pi

# --- Self-play and Training Loop ---
def self_play(net, num_simulations=50):
    memory = []  # stores (state, pi, z) tuples
    game = TicTacToeGame()
    while not game.is_terminal():
        root = MCTSNode(game.clone())
        pi = mcts_search(root, net, num_simulations=num_simulations)
        state = game.get_state()
        memory.append((state, pi, None))  # z to be filled after game outcome
        # Choose move: sample from pi restricted to legal moves
        legal = game.get_legal_actions()
        legal_idx = [i*3+j for (i, j) in legal]
        pi_legal = np.array([pi[i] if i in legal_idx else 0 for i in range(9)])
        if pi_legal.sum() > 0:
            pi_legal /= pi_legal.sum()
        else:
            pi_legal = np.ones(9) / len(legal_idx)
        move = np.random.choice(9, p=pi_legal)
        i, j = divmod(move, 3)
        game.make_move((i, j))
    # Determine game outcome from the perspective of each state.
    winner = game.get_winner()
    for idx, (state, pi, _) in enumerate(memory):
        # The perspective is that of the player who moved at that state.
        if winner == 0:
            z = 0
        else:
            # Flip sign based on state representation (since we multiplied board by current player)
            z = 1 if winner == 1 else -1
        memory[idx] = (state, pi, z)
    return memory

def train(net, optimizer, memory, batch_size=32, epochs=10):
    net.train()
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        np.random.shuffle(memory)
        for i in range(0, len(memory), batch_size):
            batch = memory[i:i+batch_size]
            states, pis, zs = zip(*batch)
            states = torch.tensor(np.array(states), dtype=torch.float32)
            target_pis = torch.tensor(np.array(pis), dtype=torch.float32)
            target_zs = torch.tensor(np.array(zs), dtype=torch.float32).unsqueeze(1)

            pred_logpis, pred_vs = net(states)
            # Policy loss: cross entropy (note that pred_logpis are log probabilities)
            policy_loss = -torch.mean(torch.sum(target_pis * pred_logpis, dim=1))
            # Value loss:
            value_loss = loss_fn(pred_vs, target_zs)
            # Total loss (add L2 reg if desired)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: Loss {loss.item()}")

def evaluate(new_net, old_net, num_games=20, num_simulations=50):
    new_wins = 0
    old_wins = 0
    draws = 0

    for game_idx in range(num_games):
        game = TicTacToeGame()
        # Alternate who goes first: new_net if game_idx is even, old_net otherwise.
        while not game.is_terminal():
            # Select network based on the current player and game index
            if game.current_player == 1:
                net = new_net if game_idx % 2 == 0 else old_net
            else:
                net = old_net if game_idx % 2 == 0 else new_net

            root = MCTSNode(game.clone())
            pi = mcts_search(root, net, num_simulations=num_simulations)
            legal = game.get_legal_actions()
            legal_idx = [i * 3 + j for (i, j) in legal]
            pi_legal = np.array([pi[i] if i in legal_idx else 0 for i in range(9)])
            if pi_legal.sum() > 0:
                pi_legal /= pi_legal.sum()
            else:
                pi_legal = np.ones(len(legal_idx)) / len(legal_idx)
            move = np.random.choice(9, p=pi_legal)
            i, j = divmod(move, 3)
            game.make_move((i, j))

        winner = game.get_winner()
        if winner == 0:
            draws += 1
        else:
            # Determine win relative to new_net's perspective.
            if (winner == 1 and game_idx % 2 == 0) or (winner == -1 and game_idx % 2 == 1):
                new_wins += 1
            else:
                old_wins += 1

    return new_wins / num_games, old_wins / num_games, draws / num_games




# --- Putting it all together ---
if __name__ == "__main__":
    net1 = TicTacToeNet()
    net2 = TicTacToeNet()
    optimizer = optim.Adam(net1.parameters(), lr=0.01)

    # Self-play to generate training data
    training_memory = []
    for iteration in range(1000):  # number of self-play games
        memory = self_play(net1, num_simulations=50)
        training_memory.extend(memory)
        # Optionally, limit memory size, shuffle, etc.
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: training on {len(training_memory)} examples")
            train(net2, optimizer, training_memory, batch_size=32, epochs=5)

            new_wins, old_wins, draws = evaluate(net2, net1, num_games=20, num_simulations=50)

            if new_wins > 0.55:
                net1 = copy.deepcopy(net2)
                print("Net1 updated")


