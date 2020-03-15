from node import Node
from games import Draughts
from network import Network
import numpy as np
import math
import tensorflow as tf
from threading import Thread
from multiprocessing import cpu_count
from copy import copy
import pickle
import time
from tqdm import tqdm
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint
import keras
import os


class AlphaZeroConfig(object):
    '''Configuration settings'''

    def __init__(self):
        ### Self-Play
        self.num_actors = 1

        self.num_sampling_moves = 30
        self.max_moves = 256
        self.num_simulations = 256

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Training
        self.training_steps = int(128)
        self.checkpoint_interval = 16
        self.window_size = int(1e5)
        self.batch_size = 4096
        self.epochs = 25
        self.checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"

        self.weight_decay = 1e-4
        self.momentum = 0.9
        # Schedule for chess and shogi, Go starts at 2e-2 immediately.
        self.learning_rate = 2e-2
        self.cores = cpu_count()
        self.job_size = 128


class Game(object):
    def __init__(self, history=None, board=None, to_play=None, logic=None):
        self.child_visits = []
        self.num_actions = 32 * 8
        self.history_len = 4
        self.width = 8
        self.number_of_pieces = 4  # normal, king for each colour
        self.constant_planes = 2  # colour, total_moves
        self.planes_per_board = self.number_of_pieces + self.constant_planes

        if not logic:
            self.logic = Draughts()  # game specific logic
        else:
            self.logic = logic
        if board is None:
            self.board = self.logic.start_state()
        else:
            self.board = board

        self.history = history or [self.board] * self.history_len

        if not to_play:
            self.to_play = 1
        else:
            self.to_play = to_play

    def terminal(self):
        return not self.logic.winner(self.board) == None

    def terminal_value(self):
        return self.logic.winner(self.history[-1])

    def xy_to_board(self, move):
        '''Convert a set of move coords to a board'''
        moves_since_start = self.board.moves_since_start + 1
        board = self.board.copy()
        while move:
            start_x, start_y, end_x, end_y = move[:4]
            move = move[4:]

            if end_y == 7 and self.to_play == board[start_y][start_x] == -1:
                board[start_y][start_x], board[end_y][end_x] = 0, -2
            elif end_y == 0 and self.to_play == board[start_y][start_x] == 1:
                board[start_y][start_x], board[end_y][end_x] = 0, 2
            else:
                board[start_y][start_x], board[end_y][end_x] = 0, board[
                    start_y][start_x]

            # check if jump performed
            if abs(start_x - end_x) == 2 and abs(start_y - end_y) == 2:
                mid_x = start_x + ((end_x - start_x) // 2)
                mid_y = start_y + ((end_y - start_y) // 2)
                board[mid_y][mid_x] = 0

        board.moves_since_start = moves_since_start
        return board

    def is_jump(self, move):
        start_x, start_y, end_x, end_y = move
        return abs(start_x - end_x) == 2 and abs(start_y - end_y) == 2

    def board_to_xy(self, out_board, in_board):
        '''Convert a board into a set of move coords'''
        now_zero = (out_board == 0) & (in_board != 0)
        was_zero = (out_board != 0) & (in_board == 0)

        end_loc = np.argwhere(was_zero)[0]

        now_zero_index = np.argwhere(now_zero)
        for piece, index in zip(in_board[now_zero], now_zero_index):
            # find a piece that is now 0 and is the same color as the end piece
            if np.sign(piece) == np.sign(out_board[was_zero]):
                start_loc = index
                break
        start_y, start_x = start_loc
        end_y, end_x = end_loc

        return start_x, start_y, end_x, end_y

    def raw_to_xy(self, start_loc):
        '''Convert a flattened index to an x,y index'''
        return start_loc % 8, start_loc // 8

    def legal_actions(self, as_xy=True, only_jumps=False):
        '''Generate all of the legal moves from the current state'''
        moves = self.logic.moves(self.board,
                                 self.to_play,
                                 only_jumps=only_jumps)
        if as_xy:
            try:
                return [self.board_to_xy(move, self.board) for move in moves]
            except Exception as e:
                print(self.board)
                print(self.to_play)
                print(only_jumps)
        else:
            return moves

    def clone(self):
        return Game(history=copy(self.history),
                    board=copy(self.board),
                    to_play=copy(self.to_play),
                    logic=copy(self.logic))

    def apply(self, action, change_player=True):
        if type(action) is tuple:
            action = self.xy_to_board(action)
        self.history.append(action)
        self.board = action
        if change_player:
            self.to_play *= -1

    def store_search_statistics(self, root, network):
        children = root.children
        sum_visits = sum(child.visits for child in children.values())
        child_visits = []
        for key in network.policy_keys:
            if key in children:
                child_visits.append(children[key].visits / sum_visits)
            else:
                child_visits.append(0)
        self.child_visits.append(child_visits)

    def binarize(self, image, to_play):
        '''Convert the current state into a series of binary feature planes'''
        features = np.zeros(
            (self.width, self.width, self.history_len * self.planes_per_board))

        for i, board in enumerate(image):
            index = i * self.planes_per_board
            features[:, :, index] = board == -1  # black_pawn
            features[:, :, index + 1] = board == 1  # white_pawn
            features[:, :, index + 2] = board == -2  # black_king
            features[:, :, index + 3] = board == 2  # white_king
            features[:, :, index + 4] = np.full(fill_value=to_play * (-1**i),
                                                shape=(self.width,
                                                       self.width))  # colour
            features[:, :, index + 5] = np.full(
                fill_value=board.moves_since_start,
                shape=(self.width, self.width))  # total_moves

        return [features]  # Need to wrap in a list because the NN expects
        # several inputs

    def make_image(self, state_index):
        '''Convert a historical state into a series of binary feature planes'''
        image = self.history[-self.history_len + state_index:state_index]
        if np.count_nonzero(image) == 0:
            return None
        return self.binarize(image, self.to_play)

    def make_target(self, state_index):
        return (self.terminal_value(), self.child_visits[state_index])

    def to_play(self):
        return len(self.history) % 2


class ReplayBuffer(object):
    def __init__(self, config):
        self.filename = 'games_{}.pickle'.format(time.time())
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = self.import_games()

    def import_games(self):
        '''Returns the latest batch of saved games'''
        games = [(float(f[6:-7]), f) for f in os.listdir() if 'games_' in f]
        if games:
            _, filename = max(games)
            with open(filename, 'rb') as f:
                games = pickle.loads(f.read())

            for game_index, game in enumerate(games):
                for i, board in enumerate(game.history):
                    if i > 3:
                        games[game_index].history[i].moves_since_start = i - 3
                    else:
                        games[game_index].history[i].moves_since_start = 0
                final_board = copy(board)
                final_board.winner = None
                winner = game.logic.winner(final_board)
                for i, board in enumerate(game.history):
                    games[game_index].history[i].winner = winner
        else:
            games = []
        return games

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)
        buffer_str = pickle.dumps(self.buffer)
        with open(self.filename, 'wb') as f:
            f.write(buffer_str)

    def sample_batch(self):
        images = []
        target_policies = []
        target_values = []
        for game in self.buffer:
            number_of_games = len(game.history[3:])
            for i in range(3, number_of_games - 1):
                image = game.make_image(i)
                if image:
                    target_value, target_policy = game.make_target(i)
                    if target_value and np.count_nonzero(target_policy):
                        images.append(image[0])
                        target_policies.append(target_policy)
                        target_values.append(target_value)

        images = np.array(images)
        target_policies = np.array(target_policies)
        target_values = np.array(target_values)
        return images, target_policies, target_values


class SharedStorage(object):
    def __init__(self):
        self._networks = []
        self.filename = 'networks_{}.pickle'.format(time.time())

    def latest_network(self, ):
        if self._networks:
            return self._networks[-1]
        else:
            network = Network()
            checkpoints = [(float(f[8:-5]), f) for f in os.listdir()
                           if '.ckpt' in f]
            _, latest = max(checkpoints)
            network.model.load_weights(latest)
            return network

    def save_network(self, network):
        self._networks.append(network)
        network.model.save_weights('network_{}.ckpt'.format(time.time()))


# AlphaZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def alphazero(config):
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)
    storage, _ = train_network(config, storage, replay_buffer)
    os.system('clear')
    for i in tqdm(range(config.training_steps),
                  desc='Main',
                  ascii=True,
                  position=0):

        _, storage = launch_job(run_selfplay, config, storage, replay_buffer)


def launch_job(function, config, storage, replay_buffer):
    replay_buffer = run_selfplay(config, storage, replay_buffer)
    storage, _ = train_network(config, storage, replay_buffer)
    return replay_buffer, storage


##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config, storage, replay_buffer):
    for i in tqdm(range(config.job_size), desc='Games', ascii=True,
                  position=1):
        network = storage.latest_network()
        game = play_game(config, network)
        replay_buffer.save_game(game)
    return replay_buffer


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config, network):
    game = Game()
    with tqdm(total=config.max_moves, desc='Moves', ascii=True,
              position=2) as pbar:
        while not game.terminal() and len(game.history) < config.max_moves:
            action, root = run_mcts(config, game, network)
            game.apply(action)
            game.store_search_statistics(root, network)
            pbar.update(1)
    return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config, game, network):
    root = Node(game.board, prior=0)
    evaluate(root, game, network)
    root = add_exploration_noise(config, root)
    for _ in range(config.num_simulations):
        node = root
        scratch_game = game.clone()
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node)
            scratch_game.apply(action)
            search_path.append(node)
        value, node = evaluate(node, scratch_game, network)
        backpropagate(search_path, value, scratch_game.to_play)
    return select_action(config, game, root), root


def select_action(config, game, root):
    actions = []
    counts = []
    visit_counts = []
    for action, child in root.children.items():
        counts.append(child.visits)
        actions.append(action)
        visit_counts.append((child.visits, action))
    if len(game.history) < config.num_sampling_moves:
        action = softmax_sample(actions, counts)
    else:
        _, action = max(visit_counts)
    return action


def softmax_sample(actions, counts):
    probs = []
    if len(counts) == 1:
        return actions[0]
    for count in counts:
        probs.append(np.exp(count) / sum(np.exp(counts)))
    action_index = np.random.choice(len(actions), 1, p=probs)[0]
    return actions[action_index]


def select_child(config, node):
    ucbs = []
    for action, child in node.children.items():
        ucb = ucb_score(config, node, child)
        ucbs.append((ucb, action, child))
        _, action, child = max(ucbs)
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config, parent, child):
    pb_c = math.log((parent.visits + config.pb_c_base + 1) /
                    config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visits) / (child.visits + 1)

    prior_score = pb_c * child.prior
    value_score = child.score()
    return prior_score + value_score


def generate_jumps(policy, scratch_game, network):
    new_dict = {}
    for action in policy:
        if scratch_game.is_jump(action):
            jump_game = scratch_game.clone()
            jump_game.apply(action, change_player=False)
            value, policy_logits = network.inference(jump_game.make_image(-1))
            jumps = jump_game.legal_actions(action, only_jumps=True)
            jumps = [j for j in jumps if action[-2:] == j[:2]]
            if jumps:
                new_policy = {a: math.exp(policy_logits[a]) for a in jumps}
                new_policy = generate_jumps(new_policy, jump_game, network)
                new_dict[action] = new_policy, policy[action]
            else:
                new_dict[action] = policy[action]
        else:
            new_dict[action] = policy[action]
    return new_dict


def flatten(dictionary, parent_key=(), parent_prob=1):
    '''Flatten the nested probability dictionary'''
    items = []
    for key, value in dictionary.items():
        if type(value) is tuple:
            new_key = parent_key + key
            new_prob = parent_prob * value[1]
            items.extend(
                flatten(value[0], parent_key=new_key,
                        parent_prob=new_prob).items())
        else:
            new_key = parent_key + key
            items.append((new_key, value * parent_prob))
    return dict(items)


# Use the neural network to obtain a value and policy prediction.
def evaluate(node, game, network):
    value, policy_logits = network.inference(game.make_image(-1))

    # Expand the node
    node.to_play = game.to_play

    policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}

    scratch_game = game.clone()
    policy = generate_jumps(policy, scratch_game, network)
    policy = flatten(policy)

    policy_sum = sum(policy.values())
    for action, p in policy.items():
        new_board = game.xy_to_board(action)
        node.children[action] = Node(new_board, prior=p / policy_sum)
    return value, node


# At the end of a simulation, propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path, value, to_play):
    for node in search_path:
        node.wins += value
        node.visits += 1


# At the start of each search, add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config, node):
    actions = node.children.keys()
    noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
    frac = config.root_exploration_fraction
    for action, noise in zip(actions, noise):
        node.children[action].prior = node.children[action].prior * (
            1 - frac) + noise * frac
    return node


######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########


def train_network(config, storage, replay_buffer, return_network=False):
    network = Network()
    optimiser = keras.optimizers.SGD(config.learning_rate)
    images, target_policies, target_values = replay_buffer.sample_batch()

    network, history = update_weights(optimiser, network, images,
                                      target_policies, target_values, config)
    storage.save_network(network)
    if return_network:
        return network
    else:
        return storage, history


def cross_entropy_with_logits(y_true, y_pred):
    loss = categorical_crossentropy(y_true, y_pred, from_logits=True)
    # loss = K.print_tensor(result, message='losses')
    return loss


def update_weights(optimiser, network, images, target_policies, target_values,
                   config):
    network.model.compile(optimizer=optimiser,
                          loss={
                              'value': 'mean_squared_error',
                              'policy': cross_entropy_with_logits
                          })
    history = network.model.fit(images, {
        'value': target_values,
        'policy': target_policies
    },
                                epochs=config.epochs)

    return network, history


if __name__ == '__main__':
    config = AlphaZeroConfig()
    alphazero(config)
