import numpy as np
import numba
import sys
import re
import os
from games import TicTacToe, Connect4, Draughts
from node import Node
import argparse
import tqdm
from threading import Thread
import time


class InputError(Exception):
    pass


class MonteCarloTreeSearch(object):
    def __init__(self, game, prog_bar_pos=0):
        self.game = game
        self.iterations = 100
        self.hash_table = {}
        self.print_path = False
        self.prog_bar_pos = prog_bar_pos

    def calculate_UCB(self, node):
        '''
        Calculate the UCB score for a node
        '''
        if not node.expanded:
            return np.inf

        if node.visits == 0:
            return np.inf
        else:
            ln_N = np.log(node.parent.visits)

        return node.get_score() + (np.sqrt((2 * ln_N) / node.visits))

    def index_max(self, values):
        '''
        Return the index of the largest value in an array. Select at random if
        draw.
        '''
        max_val = max(values)
        max_indices = [ind for ind, val in enumerate(values) if val == max_val]
        return np.random.choice(max_indices)

    def get_highest_UCB(self, node):
        '''
        Return the child with the highest UCB from a node
        '''
        UCBs = []
        for child in node.children:
            child.UCB = self.calculate_UCB(child)
            UCBs.append(child.UCB)
        try:
            max_index = self.index_max(UCBs)
        except:
            print(UCBs)
            raise ValueError

        return node.children[max_index]

    def backpropogate(self, path, winner):
        '''
        Backpropogate the result of the simulation along the path taken to the leaf node
        '''
        for node in path:
            node.wins += winner * node.player * -1
            node.visits += 1

    def simulate(self, root_node):
        '''
        Traverse the tree, following the path of largest UCB until a leaf not is reached.
        Simulate a random game from that leaf node and backpropagate the result back up the tree
        '''
        node = root_node
        path = [node]  # record the nodes that are traversed

        #follow the nodes with the highest UCB until a node that is un-expanded is reached
        while node.expanded and not node.end_state:
            node = self.get_highest_UCB(node)
            path.append(node)

        self.expand(node)
        winner = self.play_out(node)
        self.backpropogate(path, winner)

    def expand(self, node):
        '''
        Generates a new node for each possible move from the current node
        '''
        moves = self.game.get_moves(node.board, node.player)
        next_player = self.game.get_next_player(node.player)
        for move in moves:
            new_node = Node(move, next_player, parent=node)
            node.children.append(new_node)
            self.add_to_table(new_node)
        node.expanded = True
        return node

    def add_to_table(self, node):
        '''
        Adds a node to the hash table
        '''
        key = str(node.board)
        self.hash_table[key] = node

    def get_next_move(self, board, player=1):
        '''
        Explore the possiblity tree to generate the next move for the machine
        '''

        # If a node is already in the tree, select it, otherwise create a new node
        if str(board) in self.hash_table:
            root_node = self.hash_table[str(board)]
        else:
            root_node = Node(board, player)
            self.hash_table[root_node.hash] = root_node

        # Follow the path of highest UCB to a leaf, expand it, and simulate
        # a random game from that point
        loading_bar = tqdm.tqdm(range(self.iterations),
                                file=sys.stdout,
                                leave=True,
                                position=self.prog_bar_pos)
        for iterations in loading_bar:
            self.simulate(root_node)

        # Look through the scores for each possible next move and select
        # the one with highest score
        scores = [child.get_score() for child in root_node.children]
        best_index = self.index_max(scores)

        return root_node.children[best_index].board, root_node

    def game_not_finished(self, board):
        return np.isnan(self.game.get_winner(board))

    def play_out(self, node, print_path=False):
        '''
        Play out a given board state - selecting moves at random until the game ends
        '''
        if np.isnan(node.end_state):
            node.winner = self.game.get_winner(node.board)
            node.end_state = not np.isnan(node.winner)

        if node.end_state:
            return node.winner
        else:
            player = node.player
            board = node.board
            moves = [np.nan]

            # Choose moves at random until a game is won
            while self.game_not_finished(board) and len(moves) > 0:
                moves = self.game.get_moves(board, player)

                # if no possible moves, return draw
                if not moves:
                    return 0

                board_index = np.random.randint(len(moves))
                board = moves[board_index]
                player = self.game.get_next_player(player)
                # Allow printing of path for debug
                if self.print_path:
                    self.game.display_board(board)
                    print()
            return self.game.get_winner(board)


class Interface(object):
    """
    Allow a human to interact with the AI in the form of a game
    """

    def __init__(self, game, parallel=True):
        self.board = game.start_state()
        self.game = game
        self.parallel = parallel
        if parallel:
            self.searcher = ParallelMCTS(game)
        else:
            self.searcher = MonteCarloTreeSearch(game)
        self.show_probabilities = True

    def print_path(self, print_path=True):
        '''
        Prints paths followed by each simulation
        '''
        if not self.parallel:
            self.searcher.print_path = print_path

    def set_iterations(self, iters):
        '''
        Set the number of simulations that the MCTS wiill run before selecting the best move
        '''
        self.searcher.iterations = iters

    def human_go(self):
        '''
        Let a human enter a move
        '''
        self.board = self.game.human_go(self.board)

    def machine_go(self):
        '''
        Use MCTS to generate the next move for the machine
        '''
        print('Machine thinking...')
        self.board, node = self.searcher.get_next_move(self.board, player=1)

        for child in node.children:
            if self.show_probabilities:
                print('{}/{} = {}'.format(child.wins, child.visits,
                                          child.get_score()))
                print(child.board)
                print()

    def run_game(self):
        '''
        Allow the human and machine to play agianst each other until one of them wins
        '''
        player = 1
        while np.isnan(self.game.get_winner(self.board)):
            if player == 1:
                self.machine_go()
            elif player == -1:
                self.human_go()
            player *= -1
            os.system('cls')
            self.game.display_board(self.board)
        if self.game.get_winner(self.board) == 0:
            print('Draw, Everyone lost')
        elif self.game.get_winner(self.board) == 1:
            print('Machine Won! All hail our new robot overlords')
        elif self.game.get_winner(self.board) == -1:
            print('You won! Humanity is safe, for the moment...')

        _ = input('Press Enter to play again or Ctrl+C to quit')
        os.system('cls')


class ParallelMCTS(object):
    def __init__(self, game, start_player=1):
        self.game = game
        start_board = self.game.start_state()
        root_node = Node(start_board, start_player)
        root_node_key = str(start_board)
        self.master_tree = {root_node_key: root_node}
        self.number_of_threads = 4
        self.iterations = 100
        self.clear_screen = True

    def generate_tree(self,
                      game,
                      tree=None,
                      start_board=None,
                      player=1,
                      iterations=100,
                      results=[None],
                      index=0):
        '''
        Generate a MCTS tree from an existing tree. If no tree exists, create
        a new tree from a start node
        '''
        # Ensure each thread has a different random seed
        start_time = time.time()
        seed = int(time.time() + (index * 100))
        np.random.RandomState(seed)

        # Create an empty tree if one isn't passed
        if not tree:
            if not start_board:
                start_board = game.start_state()

            root_node = Node(start_board, player)
            key = str(root_node.board)
            tree = {key: root_node}

        # Initialise the tree searcher
        MCTS = MonteCarloTreeSearch(game, prog_bar_pos=index)
        MCTS.iterations = iterations
        MCTS.hash_table = tree

        # Run the simulations
        _, _ = MCTS.get_next_move(start_board, player=player)
        results[index] = MCTS.hash_table

    def combine_trees(self, trees):
        '''
        Combine several trees into one
        '''
        master_tree = {}
        # Iterate over each tree
        for tree in trees:
            # Iterate over each node in a tree
            for key, node in tree.items():
                if key not in master_tree:
                    master_tree[key] = node
                else:
                    master_tree[key].wins += node.wins
                    master_tree[key].visits += node.visits
        return master_tree

    def index_max(self, values):
        '''
        Return the index of the largest value in an array. Select at random if
        draw.
        '''
        max_val = max(values)
        max_indices = [ind for ind, val in enumerate(values) if val == max_val]
        return np.random.choice(max_indices)

    def get_next_move(self, board, player):
        '''
        Build several trees in parallel and combine them to calculate the
        best next move
        '''
        threads = [None] * self.number_of_threads
        trees = [None] * self.number_of_threads

        # Build a tree in each thread
        for i in range(self.number_of_threads):
            kwargs = {
                'tree': self.master_tree,
                'start_board': board,
                'player': player,
                'iterations': self.iterations,
                'results': trees,
                'index': i
            }
            args = (self.game, )
            threads[i] = Thread(target=self.generate_tree,
                                args=args,
                                kwargs=kwargs)
            threads[i].start()

        # Wait for each thread to finish
        for i in range(self.number_of_threads):
            threads[i].join()

        if self.clear_screen:
            #os.system('cls')
            pass
        #combine the results
        self.master_tree = self.combine_trees(trees)

        # Look through the scores for each possible next move and select
        # the one with highest score
        root_node_key = str(board)
        root_node = self.master_tree[root_node_key]
        scores = [child.get_score() for child in root_node.children]
        best_index = self.index_max(scores)

        return root_node.children[best_index].board, root_node


if __name__ == '__main__':
    # Parse arguments to allow options

    parser = argparse.ArgumentParser()
    parser.add_argument('game',
                        help='The game to be played: tictactoe or connect4')
    parser.add_argument("--iter", help="Number of iterations", default=1000)
    parser.add_argument("--no-parallel",
                        help="Runs simulations without parallelisation",
                        action='store_true',
                        default=False)
    parser.add_argument("--print-path",
                        help="Displays the path taken by each simulation",
                        action='store_true',
                        default=False)
    parser.add_argument(
        "--show-probs",
        help="Shows the probabilities calculated by the bot for each move",
        action='store_true',
        default=False)
    parser.add_argument(
        "--no-clear",
        help="If True, the screen is not cleared between rounds",
        action='store_true',
        default=False)
    parser.add_argument(
        "--cores",
        help="The number of threads that will be run concurrently",
        default=4)
    args = parser.parse_args()

    # Get game to play
    if args.game == 'tictactoe':
        game_rules = TicTacToe()
    elif args.game == 'connect4':
        game_rules = Connect4()
    elif args.game == 'draughts':
        game_rules = Draughts()

    # Decide whether to show probabilities from argument
    print('Welcome to the Monte-Carlo tree search AI')
    print()
    print('AI plays first')
    print()
    while True:
        interface = Interface(game_rules, parallel=not args.no_parallel)
        interface.set_iterations(int(args.iter))
        interface.print_path(args.print_path)
        interface.show_probabilities = args.show_probs
        interface.clear_screen = not args.no_clear
        interface.number_of_threads = args.cores
        interface.run_game()
