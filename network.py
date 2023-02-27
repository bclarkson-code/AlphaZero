import numpy as np
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import ReLU
from keras.layers import Add
from keras.layers import Lambda
from keras.initializers import normal
from keras.regularizers import l2
import pyximport
pyximport.install()
from games import Draughts
from threading import Thread
from queue import Queue


class Network(object):
    def __init__(self):
        self.width = 8
        self.number_of_pieces = 4  # normal, king for each colour
        self.constant_planes = 2  # colour, total_moves
        self.planes_per_board = self.number_of_pieces + self.constant_planes
        self.history_len = 4
        self.n_actions = 32 * 8
        self.game_logic = Draughts()
        self.n_kernels = 32
        self.shared_layers = 3
        self.kernel_size = (3, 3)
        self.initialiser = normal(mean=0, stddev=0.01)
        self.regularisation = 0.01
        self.crossentropy_constant = 1e-8
        self.model = self.make_uniform()
        self.policy_keys = [self.index_to_xy(index) for index in range(self.n_actions)]

        # self.input_queue = Queue()
        # self.output_queue = Queue()
        # thread = Thread(target=self.predict_from_queue, daemon=True)
        # thread.start()

    def index_to_xy(self, index):
        """Convert an output index into a set of move coords.
        This is done by converting the index into octal, taking the first
        2 digits as the start position and the
        final digit as a direction and distance"""
        directions = {0: (-1, -1), 1: (1, -1), 2: (-1, 1), 3: (1, 1)}
        move = index % 8
        index -= move
        index *= 2
        octal_index = oct(index)[2:].zfill(3)[:2]
        start_y, start_x = [int(o) for o in octal_index]
        if start_y % 2 == 0:
            start_x += 1
        distance = (move // 4) + 1
        direction = directions[move % 4]

        end_x = start_x + (direction[0] * distance)
        end_y = start_y + (direction[1] * distance)

        return start_x, start_y, end_x, end_y

    def xy_to_index(self, move):
        directions = {(-1, -1): 0, (1, -1): 1, (-1, 1): 2, (1, 1): 3}
        start_x, start_y, end_x, end_y = move
        vector = end_x - start_x, end_y - start_y

        if abs(vector[0]) == 2:
            distance = 1
        else:
            distance = 0

        direction = directions[vector]
        index = start_x * 64 + start_y * 8 + distance * 4 + direction
        return index

    def output_features(self, in_board, output, to_move):
        """Convert the policy output plane from the neural network into a board
        state. Output plane is a 64 * 63 = 992 vector corresponding to moving a
        piece from any valid square to any other valid square

        Each position from 0 to 31 corresponds to a position on the following
        board:
        |  |0 |  |1 |  |2 |  |3 |
        |4 |  |5 |  |6 |  |7 |  |
        |  |8 |  |9 |  |10|  |11|
        |12|  |13|  |14|  |15|  |
        |  |16|  |17|  |18|  |19|
        |20|  |21|  |22|  |23|  |
        |  |24|  |25|  |26|  |27|
        |28|  |29|  |30|  |31|  |

        An index in the output vector corresponds to:
            start_index + 32 * end_index
        Thus, the mapping from index to start_index, end_index is as
        follows:
            start_index = index % 32
            end_index = index // 32
        The mapping from a board_index to x, y is as follows:
            y = index // 4
            x = (2 * index) % 8 + y % 2
        As a piece cannot move onto itself:
        if end_index >= start_index:
            end_index += 1
        """

        def raw_to_xy(raw):
            """Converts a raw, unraveled numpy index into an x,y coord"""
            x = raw % 8
            y = raw // 8
            return x, y

        def board_to_xy(board, in_board, to_move):
            """Convert a board into a set of move coords"""
            diff = np.sign(board - in_board)
            start_loc = diff.argmin()
            start_x, start_y = raw_to_xy(start_loc)
            end_loc = diff.argmax()
            end_x, end_y = raw_to_xy(end_loc)

            return start_x, start_y, end_x, end_y

        moves = self.game_logic.moves(in_board, to_move)
        legal_move_coords = [board_to_xy(move, in_board, to_move) for move in moves]
        index = np.argmax(output)
        out_move_coords = self.index_to_xy(index)
        while out_move_coords not in legal_move_coords:
            output[index] = 0
            index = output.argmax()
            out_move_coords = self.index_to_xy(index)

        start_x, start_y, end_x, end_y = out_move_coords
        new_board = in_board.copy()
        new_board[end_y][end_x] = new_board[start_y][start_x]
        new_board[start_y][start_x] = 0
        return new_board

    def conv_block(self, in_layer):
        """
        Build a convolutional block of network layers
        """
        layer = Conv2D(
            self.n_kernels,
            self.kernel_size,
            padding="same",
            kernel_initializer=self.initialiser,
            kernel_regularizer=l2(self.regularisation),
            bias_regularizer=l2(self.regularisation),
        )(in_layer)
        layer = BatchNormalization()(layer)
        return ReLU()(layer)

    def resid_block(self, in_layer):
        """
        Build a residual block of network layers
        """
        layer = self.conv_block(in_layer)
        layer = Conv2D(
            self.n_kernels,
            self.kernel_size,
            padding="same",
            kernel_initializer=self.initialiser,
            kernel_regularizer=l2(self.regularisation),
            bias_regularizer=l2(self.regularisation),
        )(in_layer)
        layer = BatchNormalization()(layer)
        return Add()([layer, in_layer])

    def build_shared(self):
        """
        Build the shared residual stack
        """
        self.inputs = Input(
            shape=(self.width, self.width, self.planes_per_board * self.history_len)
        )
        shared = self.conv_block(self.inputs)
        for _ in range(self.shared_layers):
            shared = self.resid_block(shared)
        return shared

    def build_policy(self, shared):
        """
        Add the policy head onto the shared residual stack
        """
        layer = Conv2D(
            self.n_kernels,
            self.kernel_size,
            padding="same",
            kernel_initializer=self.initialiser,
            kernel_regularizer=l2(self.regularisation),
            bias_regularizer=l2(self.regularisation),
        )(shared)
        layer = BatchNormalization()(layer)
        layer = ReLU()(layer)
        layer = Flatten()(layer)
        layer = Dense(
            self.n_actions,
            activation="softmax",
            kernel_regularizer=l2(self.regularisation),
            bias_regularizer=l2(self.regularisation),
        )(layer)
        return Lambda(
            lambda x: x + self.crossentropy_constant,
            name="policy",
        )(layer)

    def build_value(self, shared):
        """
        Add the value head onto the shared residual stack
        """
        layer = Conv2D(
            self.n_kernels,
            self.kernel_size,
            padding="same",
            kernel_initializer=self.initialiser,
            kernel_regularizer=l2(self.regularisation),
            bias_regularizer=l2(self.regularisation),
        )(shared)
        layer = BatchNormalization()(layer)
        layer = ReLU()(layer)
        layer = Flatten()(layer)
        layer = Dense(
            self.n_kernels,
            kernel_regularizer=l2(self.regularisation),
            bias_regularizer=l2(self.regularisation),
        )(layer)
        return Dense(
            1,
            activation="tanh",
            name="value",
            kernel_initializer=self.initialiser,
            kernel_regularizer=l2(self.regularisation),
            bias_regularizer=l2(self.regularisation),
        )(layer)

    def make_uniform(self):
        """
        Actaully build the network with uniform outputs
        """
        self.shared = self.build_shared()
        self.policy = self.build_policy(self.shared)
        self.value = self.build_value(self.shared)
        self.model = Model(inputs=self.inputs, outputs=[self.policy, self.value])
        self.model.make_predict_function()
        return self.model

    def inference(self, image, as_dict=True):
        """
        Given a board state, predict the probability of the current player
        winning (value) and the probability that each of the next moves will
        result in a win (policy)
        """
        pred = self.model.predict([image])

        policy = pred[0][0]
        if as_dict:
            policy_logits = {
                self.policy_keys[i]: policy_logit
                for i, policy_logit in enumerate(policy)
            }
        else:
            policy_logits = policy
        value = pred[1][0][0]

        return value, policy_logits

    # def inference(self, image):
    #     '''
    #     Given a board state, predict the probability of the current player
    #     winning (value) and the probability that each of the next moves will
    #     result in a win (policy)
    #     '''
    #     self.input_queue.put(image)
    #     pred = self.output_queue.get()
    #     policy = pred[0][0]
    #     policy_logits = {
    #         self.policy_keys[i]: policy_logit
    #         for i, policy_logit in enumerate(policy)
    #     }
    #     value = pred[1][0][0]
    #
    #     return value, policy_logits
    #
    # def generate_from_queue(self):
    #     while True:
    #         yield self.input_queue.get()
    #
    # def predict_from_queue(self):
    #     # graph = tf.Graph()
    #     # with graph.as_default():
    #     #     self.model._make_predict_function()
    #     while True:
    #         for image in self.generate_from_queue():
    #             inference = self.model.predict([image])
    #             self.output_queue.put(inference)

    def get_weights(self):
        """
        Return a list of the weights of both the value model and the policy
        model
        """
        return self.model.get_weights()

    def set_weights(self, weights):
        """
        Set the weights of the network
        """
        self.model.set_weights(weights)
