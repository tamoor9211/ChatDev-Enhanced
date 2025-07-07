'''
This module defines the Gomoku game logic.
'''
class Gomoku:
    '''
    Represents the Gomoku game.
    '''
    def __init__(self, board_size=15):
        '''
        Initializes the Gomoku game.
        Args:
            board_size (int): The size of the game board (default: 15).
        '''
        self.board_size = board_size
        self.board = self.create_board()
        self.current_player = 1  # Player 1 starts
        self.game_over = False
    def create_board(self):
        '''
        Creates an empty game board.
        Returns:
            list[list[int]]: A 2D list representing the game board.
        '''
        return [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
    def place_stone(self, row, col):
        '''
        Places a stone for the current player on the board.
        Args:
            row (int): The row index.
            col (int): The column index.
        Returns:
            bool: True if the stone was successfully placed, False otherwise.
        '''
        if self.is_valid_move(row, col) and not self.game_over:
            self.board[row][col] = self.current_player
            return True
        return False
    def is_valid_move(self, row, col):
        '''
        Checks if a move is valid.
        Args:
            row (int): The row index.
            col (int): The column index.
        Returns:
            bool: True if the move is valid, False otherwise.
        '''
        return 0 <= row < self.board_size and 0 <= col < self.board_size and self.board[row][col] == 0
    def check_win(self, row, col):
        '''
        Checks if the current move results in a win.
        Args:
            row (int): The row index.
            col (int): The column index.
        Returns:
            bool: True if the current move results in a win, False otherwise.
        '''
        player = self.board[row][col]
        # Check horizontal
        count = 0
        for i in range(max(0, col - 4), min(self.board_size, col + 5)):
            if self.board[row][i] == player:
                count += 1
                if count == 5:
                    return True
            else:
                count = 0
        # Check vertical
        count = 0
        for i in range(max(0, row - 4), min(self.board_size, row + 5)):
            if self.board[i][col] == player:
                count += 1
                if count == 5:
                    return True
            else:
                count = 0
        # Check diagonal (top-left to bottom-right)
        count = 0
        for i in range(-4, 5):
            r = row + i
            c = col + i
            if 0 <= r < self.board_size and 0 <= c < self.board_size:
                if self.board[r][c] == player:
                    count += 1
                    if count == 5:
                        return True
                else:
                    count = 0
        # Check diagonal (top-right to bottom-left)
        count = 0
        for i in range(-4, 5):
            r = row + i
            c = col - i
            if 0 <= r < self.board_size and 0 <= c < self.board_size:
                if self.board[r][c] == player:
                    count += 1
                    if count == 5:
                        return True
                else:
                    count = 0
        return False
    def is_board_full(self):
        '''
        Checks if the board is full.
        Returns:
            bool: True if the board is full, False otherwise.
        '''
        for row in self.board:
            if 0 in row:
                return False
        return True
    def switch_player(self):
        '''
        Switches the current player.
        '''
        self.current_player = 3 - self.current_player  # Switches between 1 and 2
    def get_current_player(self):
        '''
        Returns the current player.
        Returns:
            int: The current player (1 or 2).
        '''
        return self.current_player
    def get_board(self):
        '''
        Returns the game board.
        Returns:
            list[list[int]]: The game board.
        '''
        return self.board
    def reset_game(self):
        '''
        Resets the game to its initial state.
        '''
        self.board = self.create_board()
        self.current_player = 1
        self.game_over = False