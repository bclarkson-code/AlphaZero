import numpy as np
from node import Node
import sys
import numba
import re
import pyximport
pyximport.install(language_level=2)
from c_draughts import C_Draughts


class TicTacToe(object):
    """
    A class of functions that can be used to generate next moves from a node and test for a win.
    """

    def __init__(self):
        self.board_width = 3
        self.player_ids = [-1, 1]
        self.max_moves = 10

    def get_next_player(self, player):
        '''
        gets the Id of the next player
        '''
        return player * -1

    def get_winner(self, board):
        '''
        If a board has a winner, the index of the winner is returned.
        Returns -1 if loss, 1 if win, 0 if draw and nan if game is not finished
        '''
        for p_id in self.player_ids:
            win_array = np.array([p_id] * self.board_width, dtype=np.int8)
            for i in range(self.board_width):
                # check rows
                if np.array_equal(board[i], win_array):
                    return p_id
                #check columns
                elif np.array_equal(board[:, i], win_array):
                    return p_id
                # check leading diagonal
                elif np.array_equal(np.diagonal(board), win_array):
                    return p_id
                # check non-leading diagonal
                elif np.array_equal(np.diagonal(np.flipud(board)), win_array):
                    return p_id
        # return nan if no wins losses or draws
        for i in np.nditer(board):
            if i == 0:
                return np.nan
        # must be a draw so return 0
        return 0

    def get_moves(self, board, player):
        '''
        Return a list of all possible moves from a given board
        '''
        moves = []
        for x in range(self.board_width):
            for y in range(self.board_width):
                if board[x][y] == 0:
                    copy = board.copy()
                    copy[x][y] = player
                    moves.append(copy)
        return moves

    def display_board(self, board):
        '''
        Nicely display the current board
        '''
        print('  0 1 2')
        for x, row in enumerate(board):
            sys.stdout.write(str(x))
            for val in row:
                if val == 1:
                    sys.stdout.write('|X')
                elif val == -1:
                    sys.stdout.write('|O')
                else:
                    sys.stdout.write('| ')
            print('|')

    def human_go(self, board):
        '''
        Allow a human to take a turn
        '''
        coord_pattern = re.compile('[0-{}],[0-{}]'.format(
            board.shape[0], board.shape[1]))
        print('Enter Coordinates of your go then press enter.')
        input_str = input('(space seperated, 0-2 with origin in top left)\n')

        if not coord_pattern.match(input_str):
            print('That is not in the right format, please try again...')
            return self.human_go(board)
        else:
            y, x = [int(coord) for coord in input_str.split(',')]
        if board[x][y] != 0:
            print('That square is already taken, please try again')
            self.human_go()
        else:
            board[x][y] = -1
            return board

    def start_state(self):
        '''
        Returns the inital board state (an empty 3 * 3 grid)
        '''
        return np.zeros((3, 3), dtype=np.int8)


class Connect4(object):
    """
    A class of functions that can be used to generate next moves from a node and test for a win.
    """

    def __init__(self):
        self.board_size = np.array((6, 7))
        self.player_ids = np.array([-1, 1])
        self.max_moves = (6 * 7) + 1

    def get_next_player(self, player):
        '''
        gets the Id of the next player
        '''
        return player * -1

    def human_go(self, board):
        '''
        Allow a human to take a turn
        '''
        coord_pattern = re.compile('[0-{}]$'.format(board.shape[1]))
        print('Enter Column and press enter.')
        input_str = input('(from 0-6)\n')
        if not coord_pattern.match(input_str):
            print('That is not in the right format, please try again...')
            return self.human_go(board)
        else:
            col = int(input_str)
        if board[0][col] != 0:
            print('That column is already full, please try again')
            self.human_go()
        else:
            for row in board[::-1]:
                if row[col] == 0:
                    row[col] = -1
                    return board

    @staticmethod
    @numba.jit()
    def get_winner_c(board, player_ids, board_size):
        '''
        Uses numba to quickly get the winner
        '''
        for p_id in player_ids:
            width, height = board_size
            # Check vertical lines
            for y in range(height):
                for x in range(width - 3):
                    if (board[x][y] == p_id and board[x + 1][y] == p_id
                            and board[x + 2][y] == p_id
                            and board[x + 3][y] == p_id):
                        return p_id

            # Check horizontal lines
            for y in range(height - 3):
                for x in range(width):
                    if (board[x][y] == p_id and board[x][y + 1] == p_id
                            and board[x][y + 2] == p_id
                            and board[x][y + 3] == p_id):
                        return p_id

            # Check leading diagonals
            for y in range(height - 3):
                for x in range(width - 3):
                    if (board[x][y] == p_id and board[x + 1][y + 1] == p_id
                            and board[x + 2][y + 2] == p_id
                            and board[x + 3][y + 3] == p_id):
                        return p_id

            # Check non-leading diagonals
            for y in range(height - 3):
                for x in range(3, width):
                    if (board[x][y] == p_id and board[x - 1][y + 1] == p_id
                            and board[x - 2][y + 2] == p_id
                            and board[x - 3][y + 3] == p_id):
                        return p_id
        # return nan if no wins losses or draws and some cells contain 0
        for i in np.nditer(board):
            if i == 0:
                return np.nan
        # must be a draw so return 0
        return 0

    @staticmethod
    @numba.jit()
    def get_moves_c(board, player, height, width):
        '''
        Return a list of all possible moves from a given board
        '''
        moves = []
        for x in range(width):
            copy = board.copy()
            for y in range(height):
                if board[height - y - 1][x] == 0:
                    copy[height - y - 1][x] = player
                    moves.append(copy)
                    break
        return moves

    def get_winner(self, board):
        '''
        If a board has a winner, the index of the winner is returned.
        Returns -1 if loss, 1 if win, 0 if draw and nan if game is not finished
        '''
        ids = self.player_ids
        board_size = self.board_size
        return self.get_winner_c(board, ids, board_size)

    def get_moves(self, board, player):
        '''
        Return a list of all possible moves from a given board
        '''
        width, height = self.board_size
        return self.get_moves_c(board, player, width, height)

    def display_board(self, board):
        '''
        Nicely print the current board in a form that humans can understand
        '''
        width = self.board_size[1]
        top_row_index = ' '.join([str(i) for i in range(width)])
        print('  {}'.format(top_row_index))
        for x, row in enumerate(board):
            sys.stdout.write(str(x))
            for val in row:
                if val == 1:
                    sys.stdout.write('|X')
                elif val == -1:
                    sys.stdout.write('|O')
                else:
                    sys.stdout.write('| ')
            print('|')

    def start_state(self):
        '''
        Returns the inital board state (an empty 3 * 3 grid)
        '''
        return np.zeros(self.board_size, dtype=np.int8)


class Draughts(object):
    def __init__(self):
        self.c_logic = C_Draughts()

    def moves(self, in_board, to_move, only_jumps=False):
        if self.winner(in_board) != None:
            return []
        boards, number_of_boards = self.c_logic.get_moves(
            in_board, to_move, only_jumps=only_jumps)
        out_boards = []
        if number_of_boards == 0 and only_jumps:
            return []
        for i in range(number_of_boards):
            out_board = DraughtsBoard(boards[i])
            out_board.moves_since_start = in_board.moves_since_start + 1
            out_board.winner = in_board.winner
            out_boards.append(out_board)
        if not out_boards:
            in_board.winner = to_move * -1
            return in_board
        return out_boards

    def next_player(self, player):
        return player * -1

    def winner(self, board):
        if board.moves_since_start > 256:
            return 0
        elif board.winner:
            return board.winner
        else:
            return self.c_logic.get_winner(board)

    def _in_bounds(self, x, y):
        '''Checks whether a given set of x,y coords are inside the board'''
        return 0 <= x < 8 and 0 <= y < 8

    def human_go(self, board, to_move):
        coord_pattern = re.compile('[0-7],[0-7]')
        input_str = input(
            'Enter Coordinates of the piece you wish to move then press enter (space seperated)'
        )
        # invalid input
        if not coord_pattern.match(input_str):
            print('That is not in the right format, please try again...')
            return self.human_go(board)
        else:
            y, x = [int(coord) for coord in input_str.split(',')]
            # in bounds of board
            if not self._in_bounds(x, y):
                print(
                    'Those coordinates are not inside the board, please try again...'
                )
                return self.human_go(board)
            # on different side to to_move
            elif not board[x][y] * to_move > 0:
                print('That is not your peice, please try again...')
                return self.human_go(board)
            else:
                input_str = input(
                    'Enter Coordinates (space seperated) of the location you wish to move it to then press enter'
                )
                # invalid input
                if not coord_pattern.match(input_str):
                    print(
                        'That is not in the right format, please try again...')
                    return self.human_go(board)
                else:
                    new_y, new_x = [
                        int(coord) for coord in input_str.split(',')
                    ]
                    # in bounds of board
                    if not self._in_bounds(new_x, new_y):
                        print(
                            'Those coordinates are not inside the board, please try again...'
                        )
                        return self.human_go(board)
                    # new square not empty
                    elif board[new_x][new_y] != 0:
                        print('That square is not empty, please try again...')
                        return self.human_go(board)
                    # if diagonal move
                    if new_x - x == new_y - y:
                        # if simple move
                        if abs(new_x - x) == 1:
                            board[new_x][new_y], board[x][y] = board[x][y], 0
                            return board
                        # if jump
                        elif abs(new_x - x) == 2:
                            between_x = (x + new_x) // 2
                            between_y = (y + new_y) // 2
                            if board[between_x][between_y] * to_move < 0:
                                board[new_x][new_y], board[x][y] = board[x][
                                    y], 0
                                board[between_x][between_y] = 0
                                return board
                    print('Invalid move, please try again...')
                    return self.human_go(board)

    def start_state(self):
        raw_array = np.array(
            [[0, -1, 0, -1, 0, -1, 0, -1], [-1, 0, -1, 0, -1, 0, -1, 0],
             [0, -1, 0, -1, 0, -1, 0, -1], [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0],
             [0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0]],
            dtype=np.int16)
        return DraughtsBoard(raw_array)

    def display_board(self, board):
        '''
        Nicely display the current board
        '''
        print('  0 1 2 3 4 5 6 7')
        for x, row in enumerate(board):
            sys.stdout.write(str(x))
            for val in row:
                if val == 1:
                    sys.stdout.write('|b')
                elif val == -1:
                    sys.stdout.write('|w')
                elif val == 2:
                    sys.stdout.write('|B')
                elif val == -2:
                    sys.stdout.write('|W')
                else:
                    sys.stdout.write('| ')
            print('|')


class DraughtsBoard(np.ndarray):
    """
    Modify the standard numpy array to contain some new, draughts
    specific, attributes:
     - moves since start
     - moves since capture
     - moves since king
     - if a given player has won the game
     - history of the last 8 board states
    """

    def __new__(cls, input_array, info=None):
        # Convert input_array into a DraughtsBoard instance
        obj = np.asarray(input_array).view(cls)
        # add the new attributes to the created instance
        obj.moves_since_start = 0
        obj.winner = 0
        # Finally, return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        # Add the extra attributes to the object
        self.moves_since_start = getattr(obj, 'moves_since_start', None)
        self.winner = getattr(obj, 'winner', None)
