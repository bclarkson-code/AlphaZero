import numpy as np
import math


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


class Node(object):
    def __init__(self, board):
        self.board = board
        self.children = []
        terminal_value = rules.get_winner(self.board)
        self.value = None
        if np.isnan(terminal_value):
            self.is_terminal = False
            self.winner = None
        else:
            self.is_terminal = True
            self.winner = terminal_value

    def legal_moves(self, maximising):
        if maximising:
            player = 1
        else:
            player = -1
        self.children = [
            Node(board) for board in rules.get_moves(self.board, player)
        ]
        return self.children


def minimax(node, maximising_player):
    global visited
    global rules
    visited[node.board.__str__()] = node.board

    if node.is_terminal:
        return node.winner

    elif node.value is not None:
        return node.value

    elif maximising_player:
        node.value = -math.inf
        for child in node.legal_moves(maximising_player):
            node.value = max(node.value, minimax(child, False))
        return node.value

    elif not maximising_player:
        node.value = math.inf
        for child in node.legal_moves(maximising_player):
            node.value = min(node.value, minimax(child, True))
        return node.value


if __name__ == '__main__':
    global visited
    global rules
    visited = {}  # store board states in a hash table to avoid repeats
    rules = TicTacToe()

    start_state = np.zeros((3, 3))
    root = Node(start_state)
    score = minimax(root, -1)
    number_of_nodes = len(visited.keys())
    print('The final score for an optimally played game of TicTacToe is {}'.
          format(score))
    print('The number of nodes in the TicTacToe game tree is {}'.format(
        number_of_nodes))
