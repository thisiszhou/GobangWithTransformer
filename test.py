import numpy as np
from chess_board.cons import GAME_PLAYER
from typing import Optional, Tuple, List
from loguru import logger
import copy


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
        self.empty_board = np.ones((row, col), dtype=np.float)
        self.steps = []

        # status variable
        self.current_player = GAME_PLAYER.EMPTY
        self.oppo_player = GAME_PLAYER.EMPTY
        self.last_oppo_move = None
        self.current_move = None
        self.game_end = True
        self.reset()

        self.winner = None
        # variables for transformer
        self.last_ten_board = np.array([np.zeros((row, col)) for _ in range(10)], dtype=np.float)
        self.last_five_current_board = np.array([np.zeros((row, col), dtype=int) for _ in range(5)], dtype=np.float)
        self.last_five_opposite_board = np.array([np.zeros((row, col), dtype=int) for _ in range(5)], dtype=np.float)

    def reset(self):
        self.board[...] = GAME_PLAYER.EMPTY
        self.empty_board[...] = 0
        self.winner = None
        self.last_move = None
        self.game_end = False
        self.current_player = GAME_PLAYER.PLAYER_ONE
        self.oppo_player = GAME_PLAYER.PLAYER_TWO
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
        self.update_last_window_step(mv_row, mv_col)
        self.board[mv_row, mv_col] = self.current_player
        self.empty_board[mv_row, mv_col] = -10
        has_winner, _ = self.is_player_winner(self.current_player, self.board, self.goal_chess_num)
        if has_winner == 0:
            self.reverse_player()
            return 0, None
        else:
            self.game_end = True
            self.winner = GAME_PLAYER.PLAYER_TWO if has_winner == 2 else self.current_player
            return has_winner, self.winner

    def get_valid_board(self):
        return self.empty_board

    def update_last_window_step(self, r, c):
        self.current_move = r, c
        self.steps.append(self.current_move)
        # update last window
        tmp_board = np.array(self.board, dtype=np.float)
        tmp_board[tmp_board == self.current_player] = 100
        tmp_board[tmp_board == self.oppo_player] = -100
        tmp_board /= 100
        self.last_ten_board[:-1] = -self.last_ten_board[1:]
        self.last_ten_board[-1] = tmp_board

        # update last five
        self.last_five_current_board, self.last_five_opposite_board = -self.last_five_opposite_board, -self.last_five_current_board
        # update last five opposite
        oppo_tmp = np.zeros((self.row, self.col), dtype=int)
        if self.last_move is not None:
            oppo_tmp[self.last_move] = -1
        self.last_five_opposite_board[:-1] = self.last_five_opposite_board[1:]
        self.last_five_opposite_board[-1] = oppo_tmp
        self.last_move = self.current_move
        # update end

    def get_last_train_window_output(self):
        label = None if self.winner is None else 1
        return np.array(self.last_ten_board), np.array(self.last_five_current_board), np.array(
            self.last_five_opposite_board), self.current_move, label

    def get_last_pred_window_output(self):
        """
        :return:
        board: [1, last_steps, board_size, board_size]
        me: [1, half last_steps, board_size, board_size]
        oppo: [1, half last_steps, board_size, board_size]
        """

        last_ten_board = copy.deepcopy(self.last_ten_board)
        tmp_board = np.array(self.board, dtype=np.float)
        tmp_board[tmp_board == self.current_player] = 100
        tmp_board[tmp_board == self.oppo_player] = -100
        tmp_board /= 100
        last_ten_board[:-1] = -last_ten_board[1:]
        last_ten_board[-1] = tmp_board
        # update last five
        last_five_current_board, last_five_opposite_board = copy.deepcopy(
            -self.last_five_opposite_board), copy.deepcopy(-self.last_five_current_board)
        oppo_tmp = np.zeros((self.row, self.col), dtype=int)
        if self.last_move is not None:
            oppo_tmp[self.last_move] = -1
        last_five_opposite_board[:-1] = last_five_opposite_board[1:]
        last_five_opposite_board[-1] = oppo_tmp
        return np.expand_dims(last_ten_board, axis=0), np.expand_dims(last_five_current_board, axis=0), np.expand_dims(
            last_five_opposite_board, axis=0)

    def search_current_player_certain_step(self) -> List[Tuple[int, int]]:
        # 1查看自己是否会获胜
        _, next_steps = self.is_player_winner(self.current_player, self.board, self.goal_chess_num - 1)
        if len(next_steps) > 0:
            return next_steps
        # 2查看对手是否会马上获胜
        _, next_steps = self.is_player_winner(self.oppo_player, self.board, self.goal_chess_num - 1)
        if len(next_steps) > 0:
            return next_steps
        # 3查看自己是否有冲四
        next_steps = self.search_current_player_livefour_step(self.board, self.current_player, self.oppo_player)
        if len(next_steps) > 0:
            return next_steps
        # 4查看对手是否有冲四、以及自己是否有冲五
        next_steps = self.search_current_player_livefour_step(self.board, self.oppo_player, self.current_player)
        if len(next_steps) > 0:
            return next_steps
        return []

    def check_board_livethree(self, check_board, player, oppo) -> Optional[int]:
        if check_board[0] == oppo or check_board[-1] == oppo:
            return None
        board_status = check_board[1:-1] == player
        if np.sum(board_status) >= self.goal_chess_num - 2:
            ids, = np.where(board_status == 0)
            code = 1 + ids[0]
            return code
        return None

    def search_player_one_side_four_step(self, board, player, oppo) -> List[Tuple[int, int]]:
        row, col = board.shape
        four_in_line_steps = set()
        # 1. 判断是否横向连续五子
        for x in range(row - self.goal_chess_num + 1):
            for y in range(col):
                check_board = board[x: x + self.goal_chess_num + 1, y]
                code = self.check_board_livethree(check_board, player, oppo)
                if code is not None and self.is_empty(x + code, y):
                    four_in_line_steps.add((x + code, y))

        # 2. 判断是否纵向连续五子
        for x in range(row):
            for y in range(col - self.goal_chess_num):
                check_board = board[x, y: y + self.goal_chess_num + 1]
                code = self.check_board_livethree(check_board, player, oppo)
                if code is not None and self.is_empty(x, y + code):
                    four_in_line_steps.add((x, y + code))

        # 3. 判断有斜五子
        for x in range(row - self.goal_chess_num):
            for y in range(col - self.goal_chess_num):
                # 判断是否有左上 - 右下的连续五子
                index_x = [x + delta for delta in range(self.goal_chess_num + 1)]
                index_y = [y + delta for delta in range(self.goal_chess_num + 1)]
                check_board = board[index_x, index_y]
                code = self.check_board_livethree(check_board, player, oppo)
                if code is not None and self.is_empty(x + code, y + code):
                    four_in_line_steps.add((x + code, y + code))

                # 判断是否有右上-左下的连续五子
                index_x = [x + delta for delta in range(self.goal_chess_num + 1)]
                index_y = [y + delta for delta in range(self.goal_chess_num, -1, -1)]
                check_board = board[index_x, index_y]
                code = self.check_board_livethree(check_board, player, oppo)
                if code is not None and self.is_empty(x + code, y + self.goal_chess_num - code):
                    four_in_line_steps.add((x + code, y + self.goal_chess_num - code))
        return list(four_in_line_steps)

    def search_player_livefour_step(self, board, player, oppo) -> List[Tuple[int, int]]:
        row, col = board.shape
        four_in_line_steps = set()

        # 1. 判断是否横向连续五子
        for x in range(row - self.goal_chess_num):
            for y in range(col):
                check_board = board[x: x + self.goal_chess_num + 1, y]
                code = self.check_board_livethree(check_board, player, oppo)
                if code is not None and self.is_empty(x + code, y):
                    four_in_line_steps.add((x + code, y))

        # 2. 判断是否纵向连续五子
        for x in range(row):
            for y in range(col - self.goal_chess_num):
                check_board = board[x, y: y + self.goal_chess_num + 1]
                code = self.check_board_livethree(check_board, player, oppo)
                if code is not None and self.is_empty(x, y + code):
                    four_in_line_steps.add((x, y + code))

        # 3. 判断有斜五子
        for x in range(row - self.goal_chess_num):
            for y in range(col - self.goal_chess_num):
                # 判断是否有左上 - 右下的连续五子
                index_x = [x + delta for delta in range(self.goal_chess_num + 1)]
                index_y = [y + delta for delta in range(self.goal_chess_num + 1)]
                check_board = board[index_x, index_y]
                code = self.check_board_livethree(check_board, player, oppo)
                if code is not None and self.is_empty(x + code, y + code):
                    four_in_line_steps.add((x + code, y + code))

                # 判断是否有右上-左下的连续五子
                index_x = [x + delta for delta in range(self.goal_chess_num + 1)]
                index_y = [y + delta for delta in range(self.goal_chess_num, -1, -1)]
                check_board = board[index_x, index_y]
                code = self.check_board_livethree(check_board, player, oppo)
                if code is not None and self.is_empty(x + code, y + self.goal_chess_num - code):
                    four_in_line_steps.add((x + code, y + self.goal_chess_num - code))
        return list(four_in_line_steps)

    def is_player_winner(self, player, board, goal_num) -> Tuple[int, List]:
        """
        :param player:
        :param goal_num:
        :param board:  棋盘，0为空位置，1，-1分别对应玩家1，2下的棋子
        :param player_value: 当前玩家下棋标记value
        :return: 是否已经获胜 0: 未获胜 1: 获胜 2: 平局
        """
        row, col = board.shape
        # 1. 判断是否横向连续五子
        status = 0
        win_steps = []
        for x in range(row - self.goal_chess_num + 1):
            for y in range(col):
                board_status = board[x: x + self.goal_chess_num, y] == player
                if np.sum(board_status) >= goal_num:
                    if status <= 0:
                        status = 1
                    if goal_num != self.goal_chess_num:
                        ids, = np.where(board_status == 0)
                        if self.is_empty(x + ids[0], y):
                            win_steps.append((x + ids[0], y))

        # 2. 判断是否纵向连续五子
        for x in range(row):
            for y in range(col - self.goal_chess_num + 1):
                board_status = board[x, y: y + self.goal_chess_num] == player
                if np.sum(board_status) >= goal_num:
                    if status <= 0:
                        status = 1
                    if goal_num != self.goal_chess_num:
                        ids, = np.where(board_status == 0)
                        if self.is_empty(x, y + ids[0]):
                            win_steps.append((x, y + ids[0]))

        # 3. 判断有斜五子
        for x in range(row - self.goal_chess_num + 1):
            for y in range(col - self.goal_chess_num + 1):
                # 判断是否有左上 - 右下的连续五子
                index_x = [x + delta for delta in range(self.goal_chess_num)]
                index_y = [y + delta for delta in range(self.goal_chess_num)]
                board_status = board[index_x, index_y] == player
                if np.sum(board_status) >= goal_num:
                    if status <= 0:
                        status = 1
                    if goal_num != self.goal_chess_num:
                        ids, = np.where(board_status == 0)
                        if self.is_empty(x + ids[0], y + ids[0]):
                            win_steps.append((x + ids[0], y + ids[0]))

                # 判断是否有右上-左下的连续五子
                index_x = [x + delta for delta in range(self.goal_chess_num)]
                index_y = [y + delta for delta in range(self.goal_chess_num - 1, -1, -1)]
                board_status = board[index_x, index_y] == player
                if np.sum(board_status) >= goal_num:
                    if status <= 0:
                        status = 1
                    if goal_num != self.goal_chess_num:
                        ids, = np.where(board_status == 0)
                        if self.is_empty(x + ids[0], y + self.goal_chess_num - 1 - ids[0]):
                            win_steps.append((x + ids[0], y + self.goal_chess_num - 1 - ids[0]))

        # 4. 判断是否为平局
        if (board[...] != GAME_PLAYER.EMPTY).all():  # 棋盘中没有剩余的格子，判断为平局
            if status <= 0:
                status = 2

        return status, win_steps

    def reverse_player(self):
        if self.game_end:
            raise ValueError("game end is True!")
        if self.current_player == GAME_PLAYER.PLAYER_ONE:
            self.current_player = GAME_PLAYER.PLAYER_TWO
            self.oppo_player = GAME_PLAYER.PLAYER_ONE
        else:
            self.current_player = GAME_PLAYER.PLAYER_ONE
            self.oppo_player = GAME_PLAYER.PLAYER_TWO
        return self.current_player

    def shape(self):
        return self.board.shape

    def get_steps(self):
        return self.steps

    def get_chess_value(self, r, c):
        return self.board[r, c]

    def is_empty(self, r, c):
        return self.board[r, c] == GAME_PLAYER.EMPTY
