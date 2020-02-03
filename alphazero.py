from node import Node
from games import Draughts


class AlphaZeroConfig(object):
    '''Configuration settings'''

    def __init__(self):
        ### Self-Play
        self.num_actors = 5000

        self.num_sampling_moves = 30
        self.max_moves = 512
        self.num_simulations = 800

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Training
        self.training_steps = int(700e3)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = 4096

        self.weight_decay = 1e-4
        self.momentum = 0.9
        # Schedule for chess and shogi, Go starts at 2e-2 immediately.
        self.learning_rate_schedule = {
            0: 2e-1,
            100e3: 2e-2,
            300e3: 2e-3,
            500e3: 2e-4
        }


class Game(object):
    def __init__(self, history=None, board=None, to_play=None, logic=None):
        self.history = history or []
        self.child_visits = []
        self.num_actions = 256  # (8 * 4) * (4+4) = board spaces * move per peice
        if not logic:
            self.logic = Draughts()  # game specific logic
        else:
            self.logic = logic
        if not board:
            self.board = self.logic.start_state()
        else:
            self.board = board

        if not to_play:
            self.to_play = 1
        else:
            self.to_play = to_play

    def terminal(self):
        return self.logic.winner() == None

    def terminal_value(self, to_play):
        return self.logic.winner()

    def legal_actions(self):
        return self.logic.moves(self.board, self.to_play)

    def clone(self):
        return Game(history=self.history,
                    board=self.board,
                    to_play=self.to_play,
                    logic=self.logic)

    def apply(self, action):
        self.history.append(action)

    def store_search_statistics(self, root):
        sum_visits = sum(child.visits for child in root.children)
        self.child_visits.append([
            root.children[a].visit_count /
            sum_visits if a in root.children else 0
            for a in range(self.num_actions)
        ])

    def make_image(self):
        image = [self.board]
        image.extend(history[-7:])
        image.append(self.to_play)
        return image

    def make_target(self, state_index: int):
        return (self.terminal_value(state_index % 2),
                self.child_visits[state_index])

    def to_play(self):
        return len(self.history) % 2


class Network(object):
    def __init__(self):
        pass

    def inference(self, image):
        return (-1, {})  # Value, Policy

    def get_weights(self):
        # Returns the weights of this network.
        return []
