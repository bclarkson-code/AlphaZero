import numpy as np


class Node(object):
    '''
    A node in the tree of possible moves. Can generate children
    and calculate whether the current board has been won by a player.

    As a convention, Machine goes first and has ID 1, human has id -1
    '''

    def __init__(self, board, player, parent=None):
        self.board = board
        self.player = player
        self.children = []
        self.winner = np.nan
        self.end_state = np.nan
        self.wins = 0
        self.visits = 0
        self.UCB = np.inf
        self.parent = parent
        self.expanded = False
        self.hash = str(self.board)

    def score(self):
        '''
        Returns the sum of wins compared to the number of simulated games
        '''
        if self.visits == 0:
            return 0
        return (1 + (self.wins / self.visits)) * 0.5

    def expanded(self):
        '''
        Returns True if node has children
        '''
        return len(self.children) > 0
