cimport numpy as np
import numpy as pnp

cdef class Moves:
    '''Object to store a set of board moves'''
    cdef short moves[32][8][8], number_of_boards

    def __init__(self):
        self.number_of_boards = 0 # number of stored boards

    cdef append(self, short[:, :] board):
        '''Add a board to the store'''
        cdef short x, y

        for x in range(8):
            for y in range(8):
                self.moves[self.number_of_boards][x][y] = board[x][y]

        self.number_of_boards += 1
        assert self.number_of_boards < 32

    cdef get_boards(self):
        '''Return an array of every stored baord state'''
        return pnp.array(self.moves)

    cdef reset(self):
        '''Reset the state of moves to its initalised state'''
        self.number_of_boards = 0

cdef class C_Draughts:
    cdef Moves moves

    def __init__(self):
        self.moves = Moves()

    cdef get_steps(self, short piece):
        '''Generate the possible steps (e.g [-1,-1]) for a piece'''
        cdef short steps[4][2]
        cdef short number_of_steps

        if piece == 1:
            steps[0][0], steps[0][1] = -1, 1
            steps[1][0], steps[1][1] = -1, -1
            number_of_steps = 2
        elif piece == -1:
            steps[0][0], steps[0][1] = 1, 1
            steps[1][0], steps[1][1] = 1, -1
            number_of_steps = 2
        else:
            steps[0][0], steps[0][1] = 1, 1
            steps[1][0], steps[1][1] = 1, -1
            steps[2][0], steps[2][1] = -1, 1
            steps[3][0], steps[3][1] = -1, -1
            number_of_steps = 4
        return steps, number_of_steps

    def get_winner(self, short[:, :] board):
        '''If a player is the only remaining player, they have won so return their id'''
        cdef short x,y, plus_tot, minus_tot

        plus_tot = 0
        minus_tot = 0

        for x in range(8):
            for y in range(8):
                if board[x][y] > 0:
                    plus_tot += 1
                elif board[x][y] < 0:
                    minus_tot += 1
        if plus_tot == 0:
            return 1
        elif minus_tot == 0:
            return -1
        else:
            return None

    cdef in_bounds(self, short x, short y):
        '''Checks whether a given set of x,y coords are inside the board'''
        return 0 <= x and x < 8 and 0 <= y and y < 8

    cdef valid_move(self, short x, short y, short[:, :] board):
        '''Test whether a given x, y is empty and within the board'''
        if self.in_bounds(x, y):
            if board[x][y] == 0:
                return True
            else:
                return False
        else:
            return False

    cdef valid_jump(self,
                    short between_x,
                    short between_y,
                    short new_x,
                    short new_y,
                    short[:, :] board,
                    short to_move):
        '''Test whether a given jump is valid'''
        if self.in_bounds(between_x, between_y):
            if board[between_x][between_y] * to_move < 0:
                if self.valid_move(new_x, new_y, board):
                    return True
        return False

    cdef generate_moves(self, short x, short y, short[:,:] board, short to_move):
        '''Generate every possible move for a given board state'''
        cdef short number_of_moves, move_index, new_x, new_y, piece
        cdef short move[2]

        if board[x][y] * to_move > 0: # if piece is on same side as to_move
            steps, number_of_moves = self.get_steps(to_move)
            for move_index in range(number_of_moves):
                move = steps[move_index]
                new_x = x + move[0]
                new_y = y + move[1]
                if self.valid_move(new_x, new_y, board):
                    #store the value of the piece (in case it is promoted)
                    piece = board[x][y]

                    # If a piece reaches the other side of the board it is promoted
                    if new_x == 7  and to_move == 1 or new_x == 0 and to_move == -1 and abs(board[x][y]) == 1:
                        board[x][y], board[new_x][new_y] = 0, board[x][y] + to_move
                    else:
                        board[x][y], board[new_x][new_y] = 0, board[x][y]
                    self.moves.append(board)

                    #return the board to its original state
                    board[new_x][new_y], board[x][y] = 0, piece

    cdef generate_jumps(self, short x, short y, short[:,:] board, short to_move):
        '''Generate every possible jump from a given board state'''
        cdef short move[2]
        cdef short number_of_moves, move_index
        cdef short between_x, between_y, new_x, new_y
        cdef short piece, between_piece

        if board[x][y] * to_move > 0: # if piece is on same side as to_move
            steps, number_of_moves = self.get_steps(to_move)

            for move_index in range(number_of_moves):
                move = steps[move_index]
                between_x = x + move[0]
                between_y = y + move[1]
                new_x = x + (move[0] * 2)
                new_y = y + (move[1] * 2)

                # If jumping over piece on other team and next space is free
                if self.valid_jump(between_x, between_y, new_x, new_y, board, to_move):
                    #store the value of the piece (in case it is promoted)
                    piece = board[x][y]
                    between_piece = board[between_x][between_y]

                    # If a piece reaches the other side of the board it is promoted
                    if (new_x == 0  and to_move == 1) or (new_x == 7 and to_move == -1) and abs(board[x][y]) == 1:
                        board[x][y], board[new_x][new_y] = 0, board[x][y] + to_move
                    else:
                        board[x][y], board[new_x][new_y] = 0, board[x][y]

                    # remove piece that is jumped over
                    board[between_x][between_y] = 0
                    self.moves.append(board)

                    # multiple jumps are allowed
                    self.generate_jumps(new_x, new_y, board, board[new_x][new_y])

                    #return the board to its original state
                    board[new_x][new_y], board[x][y] = 0, piece
                    board[between_x][between_y] = between_piece

    def get_moves(self, short[:,:] board, short to_move):
        '''Return all possible moves from a given board'''
        cdef short x, y

        self.moves.reset()
        for x in range(8):
            for y in range(8):
                self.generate_moves(x, y, board, to_move)
                self.generate_jumps(x, y, board, to_move)
        return self.moves.get_boards(), self.moves.number_of_boards
