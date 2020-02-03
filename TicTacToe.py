import numpy as np
import numba
import sys
import re
import os

class Node(object):
    '''
    A node in the tree of possible moves. Can generate children 
    and calculate whether the current board has been won by a player.
    
    As a convention, Machine goes first and has ID 1, human has id -1
    '''
    def __init__(self, board, player, parents=[]):
        self.board = board
        self.board_width = 3
        self.player = player
        self.moves = [] 
        self.children = []
        self.next_player = self.get_next_player(self.player)
        self.winner = self.get_winner(self.board)
        self.end_state = not np.isnan(self.winner)
        self.wins = 0
        self.losses = 0
        self.visits = 0
        self.UCB = np.inf
        self.parents = parents
        self.expanded = False
        
    def get_score(self):
        '''
        Returns the fraction of wins compared to the number of simulated games
        '''
        denominator = self.wins + self.losses
        # If no games have been simulated, guess that the score is 0.5
        if denominator == 0:
               return 0.5
        else:
            return self.wins / denominator
    
    def get_winner(self, board):
        '''
        If a board has a winner, the index of the winner is returned. 
        Returns -1 if loss, 1 if win, 0 if draw and nan if game is not finished
        '''
        X_win = np.array([1] * self.board_width)
        O_win = np.array([-1] * self.board_width)
        win_arrays =[X_win, O_win]
        for i, winner in enumerate([1, -1]):
            win_array = win_arrays[i]
            for i in range(self.board_width):
                # check rows
                if np.array_equal(board[i], win_array):
                    return winner
                #check columns
                elif np.array_equal(board[:,i], win_array):
                    return winner
                # check leading diagonal
                elif np.array_equal(np.diagonal(board), win_array):
                    return winner
                # check non-leading diagonal
                elif np.array_equal(np.diagonal(np.flipud(board)), win_array):
                    return winner
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
        
    def get_next_player(self, player):
        '''
        Given a player, generate the id of the next player
        '''
        return player * -1
        
    def _generate_children(self):
        '''
        Generates a new node for each possible move from the current board
        '''
        self.moves = self.get_moves(self.board, self.player)
        for move in self.moves:
            parents = self.parents
            parents.append(self)
            self.children.append(
                Node(move, 
                     self.next_player,
                     parents=parents))
        self.expanded = True
            
    def calculate_UCBs(self):
        '''
        Calculate the UCB score for each child
        '''
        if len(self.children) == 0:
            self._generate_children()
        UCBs = []
        if self.visits == 0:
            ln_N = np.inf
        else:
            ln_N = np.log(self.visits)
        for child in self.children:
            if child.visits == 0:
                child.UCB = np.inf
            else:
                child.UCB = child.get_score() + (2 * np.sqrt(ln_N/child.visits))
            UCBs.append(child.UCB)
        return UCBs
        
                
    def play_out(self, print_path=False):
        '''
        Play out a given board state - selecting moves at random until the game ends
        '''
        board = self.board.copy()
        player = self.player
        moves = [np.nan]
        while np.isnan(self.get_winner(board))and len(moves) > 0:
            moves = self.get_moves(board, player)
            if print_path:
                for move in moves:
                    self.display_board(move)
                print()
            board_id = np.random.randint(len(moves))
            board = moves[board_id]
            
            player = self.get_next_player(player)
        return self.get_winner(board)
                
    def display_board(self, board=np.array([[np.inf] * 3] * 3)):
        '''
        Print the current board state
        '''
        if board[0][0] == np.inf:
            board = self.board
        string = str(board)
        string = string.replace('0', ' ')
        string = string.replace('-1', ' O')
        string = string.replace('1', 'X')
        print(string)

            
class Game(object):
    def __init__(self):
        self.board = np.zeros((3,3), dtype=np.int8)
        self.iterations = 1000
        self.board_width = 3
        self.coord_pattern = re.compile('[0-2],[0-2]')
        
    def human_go(self):
        '''
        Allow a human to take a turn
        '''
        print('Enter Coordinates of your go then press enter.')
        input_str = input('(space seperated, 0-2 with origin in top left)\n')
        if not self.coord_pattern.match(input_str):
            print('That is not in the right format, please try again...')
            self.human_go()
        else:
            y,x = [int(coord) for coord in input_str.split(',')]
        if self.board[x][y] != 0:
            print('That square is already taken, please try again')
            self.human_go()
        else:
            self.board[x][y] = -1
        
    def machine_go(self):
        '''
        Use MCTS to generate the next move for the machine
        '''
        print('Machine thinking...')
        self.board = self.get_next_move(self.board)
        
    def get_next_move(self, board, player=1):
        '''Explore the possiblity tree to generate the next move for the machine'''
        root_node = Node(board, player)
        root_node._generate_children()

        def index_max(values):
            return max(range(len(values)), key=values.__getitem__)

        for iterations in range(self.iterations):
            node = root_node
            path = [node] # record the nodes that are traversed
            #follow the nodes with the lowest UCB until a node that is un-expanded is reached
            while node.expanded:
                node.visits += 1
                UCBs = node.calculate_UCBs()
                node = node.children[index_max(UCBs)]
                path.append(node)
            node.visits += 1
            if not node.end_state:
                node._generate_children()
                winner = node.play_out()
            else:
                winner = node.winner
                if winner == node.player:
                    node.score = 1
                elif winner == node.player * -1:
                    node.score = 0
                else:
                    node.score = 0.5
            if winner:
                for node in path:
                    if node.player == winner:
                        node.losses += 1
                    else:
                        node.wins += 1
        scores = [child.get_score() for child in root_node.children]
        best_index = index_max(scores)
        return root_node.children[best_index].board
        
    def display_board(self):
        '''
        Nicely display the current board
        '''
        print('  0 1 2')
        for x, row in enumerate(self.board):
            sys.stdout.write(str(x))
            for val in row:
                if val == 1:
                    sys.stdout.write('|X')
                elif val == -1:
                    sys.stdout.write('|O')
                else:
                    sys.stdout.write('| ')
            print('|')
            
    def get_winner(self, board):
        '''
        If a board has a winner, the index of the winner is returned. 
        Returns -1 if loss, 1 if win, 0 if draw and nan if game is not finished
        '''
        X_win = np.array([1] * self.board_width)
        O_win = np.array([-1] * self.board_width)
        win_arrays =[X_win, O_win]
        for i, winner in enumerate([1, -1]):
            win_array = win_arrays[i]
            for i in range(self.board_width):
                # check rows
                if np.array_equal(board[i], win_array):
                    return winner
                #check columns
                elif np.array_equal(board[:,i], win_array):
                    return winner
                # check leading diagonal
                elif np.array_equal(np.diagonal(board), win_array):
                    return winner
                # check non-leading diagonal
                elif np.array_equal(np.diagonal(np.flipud(board)), win_array):
                    return winner
        # return nan if no wins losses or draws
        for i in np.nditer(board):
            if i == 0:
                return np.nan
        # must be a draw so return 0
        return 0
            
    def run_game(self):
        '''
        Allow the human and machine to play agianst each other until one of them wins
        '''
        player = 1
        while np.isnan(self.get_winner(self.board)):
            if player == 1:
                self.machine_go()
            elif player == -1:
                self.human_go()
            player *= -1
            self.display_board()
        if self.get_winner(self.board) == 0:
            print('Draw, Everyone lost')
        elif self.get_winner(self.board) == 1:
            print('Machine Won! All hail our new robot overlords')
        elif self.get_winner(self.board) == -1:
            print('You won! Humanity is safe, for the moment...')
            
        _ = input('Press Enter to play again or Ctrl+C to quit')
        os.system('cls')
        self.run_game()
        
       
if __name__ == '__main__':
	g = Game()
	print('Welcome to the Monte-Carlo')
	print('tree search TicTacToe AI')
	print('AI plays first')
	g.run_game()