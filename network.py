import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import ReLU
from keras.layers import Add
import pyximport
pyximport.install()
from c_draughts import C_Draughts


class Network(object):
    def __init__(self):
        self.history_len = 8
        self.width = 8
        self.number_of_pieces = 4  # normal, king for each colour
        self.constant_planes = 4  # colour, total_moves, no_progress, last_move_capture
        self.planes_per_board = self.number_of_pieces + self.constant_planes
        self.c_logic = C_Draughts()
        self.n_kernels = 64
        self.shared_layers = 5
        self.kernel_size = (3,3)

    def input_features(self, image, last_move_capture=False):
        # Convert the raw input boards into feature planes
        colour = image[-1]
        features = np.zeros(
            (self.width, self.width, self.history_len * self.planes_per_board))
        for i, board in enumerate(image[:-1]):
            index = i * self.planes_per_board
            features[index] = np.where(board == -1, board)  # black_pawn
            features[index + 1] = np.where(board == 1, board)  # white_pawn
            features[index + 2] = np.where(board == -2, board)  # black_king
            features[index + 3] = np.where(board == 2, board)  # white_king
            features[index + 4] = np.full(colour * (-1**i),
                                          size=(self.width,
                                                self.width))  # colour
            features[index + 5] = np.full(board.moves_since_start,
                                          size=(self.width,
                                                self.width))  # total_moves
            moves_since_progess = max(
                [board.moves_since_king, board.moves_since_capture])
            features[index + 6] = np.full(moves_since_progress,
                                          size=(self.width,
                                                self.width))  # no_progress
            features[index + 7] = np.full(last_move_capture,
                                          size=(self.width, self.width))
        return features

    def output_features(self, in_board, output, to_play):
        '''
        Convert the output plane from the neural network into a board state
        output plane is a 32 * 3 * 4 set of planes (number of
        available spaces * number of squares to move
        (move + king move + capture) * directions (NE, NW, SE, SW))

        Each position from 0 to 31 corresponds to a position on the following
        board:
        |0 |  |1 |  |2 |  |3 |  |
        |  |4 |  |5 |  |6 |  |7 |
        |8 |  |9 |  |10|  |11|  |
        |  |12|  |13|  |14|  |15|
        |16|  |17|  |18|  |19|  |
        |  |20|  |21|  |22|  |23|
        |24|  |25|  |26|  |27|  |
        |  |28|  |29|  |30|  |31|
        The mapping from (x, y) to index is as follows:
        let i = x + (y * 8)

        index = {
            i / 2   if i % 2 == y % 2,
            None    if i % 2 != y % 2
        }

        the mapping from index to x,y is as follows:
        y = floor(index / 4)
        x = (2 * index) % 8 + y % 2

        '''

        def move_to_xy(index):
            y = index // 4
            x = (2 * index) % 8 + y % 2
            return x, y

        def in_bounds(x, y):
            '''Checks whether a given set of x,y coords are inside the board'''
            return 0 <= x and x < 8 and 0 <= y and y < 8

        def is_legal(old_board, new_board):
            return new_board in self.c_logic.get_moves(old_board, to_play)

        def to_move(index,
                    n_squares = 12,
                    n_directions = 3):
            square = index // n_squares
            index -= square * n_squares
            direction = index // n_directions
            distance = index - direction * n_directions
            x, y = move_to_xy(square)
            return square, direction, distance

        directions = {
            0: np.array([1, 1]),
            1: np.array([1, -1]),
            2: np.array([-1, 1]),
            3: np.array([-1, -1]),
        }

        legal_move = False
        while not legal_move:
            index = output.argmax()
            x, y, direction, distance = to_move(index)
            new_board = old_board.copy()
            piece = new_board[x][y]
            direction = directions[direction]
            new_x = direction[0] + x
            new_y = direction[1] + y

            # ensure new location is in board range
            if not in_bounds(new_x, new_y):
                output[index][distance][direction] = 0
                continue

            new_board[x][y] = 0
            new_board[new_x][new_y] = piece
            # ensure new move is legal
            if is_legal(old_board, new_board):
                return new_board
            else:
                output[index][distance][direction] = 0
                continue

    def conv_block(self, in_layer):
        '''
        Build a convolutional block of network layers
        '''
        layer = Conv2D(self.n_kernels,
                       self.kernel_size,
                       padding='same')(in_layer)
        layer = BatchNormalization()(layer)
        return ReLU()(layer)

    def resid_block(self, in_layer):
        '''
        Build a residual block of network layers
        '''
        layer = conv_block(in_layer)
        layer = Conv2D(self.n_kernels,
                       self.kernel_size,
                       padding='same')(in_layer)
        layer = BatchNormalization()(layer)
        return Add()([layer, in_layer])

    def build_shared(self):
        '''
        Build the shared residual stack
        '''
        inputs = Input(shape=(self.width,
                              self.width,
                              self.planes_per_board))
        shared = self.conv_block(inputs)
        for layer_num in range(self.shared_layers):
            shared = self.resid_block(shared)
        return shared

    def build_policy(self, shared):
        '''
        Add the policy head onto the shared residual stack
        '''
        layer = Conv2D(k, (3, 3), padding='same')(shared)
        layer = BatchNormalization()(layer)
        layer = ReLU()(layer)
        layer = Flatten()(layer)
        return Dense(384, activation='sigmoid')(layer)]

    def build_value(self, shared):
        '''
        Add the value head onto the shared residual stack
        '''
        layer = Conv2D(k, (3, 3), padding='same')(shared)
        layer = BatchNormalization()(layer)
        layer = ReLU()(layer)
        layer = Flatten()(layer)
        layer = Dense(k)(layer)
        return Dense(1, activation='tanh')(layer)

    def build(self):
        shared = self.build_shared()
        policy = self.build_policy(shared)
        value = self.build_policy(shared)
