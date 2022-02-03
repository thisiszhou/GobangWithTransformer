import numpy as np
from chess_board.cons import GAME_PLAYER
from typing import Optional, Tuple
from loguru import logger


class ChessBoard(object):
    def __init__(self, row: int, col: int, goal_chess_num: int = 5):
        """
        :param row:  row of chess board
        :param col:  col of chess board
        """
        assert row >= goal_chess_num and col >= goal_chess_num
        self.row = row
        self.col = col
        self.goal_chess_num = goal_chess_num
        self.board = np.zeros((row, col), dtype=int)
        self.steps = []

        # status variable
        self.current_player = GAME_PLAYER.EMPTY
        self.last_move = None
        self.game_end = True
        self.reset()

    def reset(self):
        self.board[...] = GAME_PLAYER.EMPTY
        self.last_move = None
        self.game_end = False
        self.current_player = GAME_PLAYER.PLAYER_ONE
        self.steps = []

    def move(self, mv_row: int, mv_col: int) -> Tuple[int, Optional[GAME_PLAYER]]:
        """
        :return: status, winner
        status: -1: move failed; 0: move succeed and game go on; 1: move succeed and return winner; 2: move succeed and
        tie(has no winner)
        """
        if self.game_end:
            logger.error('Game is not reset !')
            return -1, None
        if self.board[mv_row, mv_col] != GAME_PLAYER.EMPTY:
            logger.error(f'board already has a chess in {mv_row, mv_col}')
            return -1, None
        self.last_move = (mv_row, mv_col)
        self.steps.append((mv_row, mv_col))
        self.board[mv_row, mv_col] = self.current_player
        has_winner = self.is_current_player_winner()
        if has_winner == 0:
            self.reverse_player()
            return 0, None
        else:
            self.game_end = True
            return has_winner, None if has_winner == 2 else self.current_player

    def is_current_player_winner(self) -> int:
        """
        :param board_map: 棋盘，0为空位置，1，2分别对应玩家1，2下的棋子
        :param player_value: 当前玩家下棋标记value
        :return: 是否已经获胜 0: 未获胜 1: 获胜 2: 平局
        """
        row, col = self.board.shape
        # 1. 判断是否横向连续五子
        for x in range(row - self.goal_chess_num + 1):
            for y in range(col):
                if (self.board[x: x + self.goal_chess_num, y] == self.current_player).all():
                    return 1

        # 2. 判断是否纵向连续五子
        for x in range(row):
            for y in range(col - self.goal_chess_num + 1):
                if (self.board[x, y: y + self.goal_chess_num] == self.current_player).all():
                    return 1

        # 3. 判断有斜五子
        for x in range(row - self.goal_chess_num + 1):
            for y in range(col - self.goal_chess_num + 1):
                # 判断是否有左上 - 右下的连续五子
                index_x = [x + delta for delta in range(self.goal_chess_num)]
                index_y = [y + delta for delta in range(self.goal_chess_num)]
                if (self.board[index_x, index_y] == self.current_player).all():
                    return 1
                # 判断是否有右上-左下的连续五子
                index_x = [x + delta for delta in range(self.goal_chess_num)]
                index_y = [y + delta for delta in range(self.goal_chess_num - 1, -1, -1)]
                try:
                    if (self.board[index_x, index_y] == self.current_player).all():
                        return 1
                except:
                    print(f"index_x: {index_x}, index_y: {index_y}")
                    raise

        # 4. 判断是否为平局
        if (self.board[...] != GAME_PLAYER.EMPTY).all():  # 棋盘中没有剩余的格子，判断为平局
            return 2

        return 0

    def reverse_player(self):
        if self.game_end:
            raise ValueError("game end is True!")
        if self.current_player == GAME_PLAYER.PLAYER_ONE:
            self.current_player = GAME_PLAYER.PLAYER_TWO
        else:
            self.current_player = GAME_PLAYER.PLAYER_ONE
        print("self.current_player:", self.current_player)
        return self.current_player

    def shape(self):
        return self.board.shape

    def get_steps(self):
        return self.steps

    def get_chess_value(self, r, c):
        return self.board[r, c]

    def is_empty(self, r, c):
        return self.board[r, c] == GAME_PLAYER.EMPTY


