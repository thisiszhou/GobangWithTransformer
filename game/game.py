from game.base_board import ChessBoard
from game.cons import GAME_PLAYER
import numpy as np
from loguru import logger
from collections import defaultdict
import time


class Game(object):
    def __init__(self,
                 chess_size,
                 goal_chess_num=5,
                 collect_train_data=True,
                 first_random_alpha=10
                 ):
        self.player_method = dict()
        self.current_player = None
        self.is_play = False
        self.chessboard = ChessBoard(chess_size, chess_size, goal_chess_num, random_alpha=first_random_alpha)
        self.action = None
        self.winner = None
        self.epoch = -1
        self.debug_show = True
        self.collect_train_data = collect_train_data
        self.board = []
        self.me = []
        self.oppo = []
        self.step = []
        self.label = []
        self.weight = []

    def start(self):
        self.is_play = True
        self.epoch += 1
        self.winner = None
        self.chessboard.reset()
        self.current_player = self.chessboard.current_player

        # collect data
        self.board = []
        self.me = []
        self.oppo = []
        self.step = []
        self.label = []

    def end(self):
        self.is_play = False

    def change_player_agent(self):
        self.player_method = {
            GAME_PLAYER.PLAYER_ONE: self.player_method[GAME_PLAYER.PLAYER_TWO],
            GAME_PLAYER.PLAYER_TWO: self.player_method[GAME_PLAYER.PLAYER_ONE]
        }

    def set_player(self, player1_method, player2_method):
        self.player_method = {GAME_PLAYER.PLAYER_ONE: player1_method, GAME_PLAYER.PLAYER_TWO: player2_method}

    def play(self, job_id, wait=0, displayer=None):
        self.start()
        while not self.is_over():
            if displayer is not None:
                displayer.show_fps(self.chessboard.steps, self.chessboard.board, self.chessboard.is_empty)
            if self.is_playing() and not self.is_over():
                if wait > 0:
                    time.sleep(wait)
                current_player = self.chessboard.current_player
                action = self.player_method[current_player](self.chessboard)
                if action is None:
                    break
                self.move(*action)
                if self.collect_train_data:
                    self.update_train_data()
        self.end()
        return self.get_train_data(), self.winner

    def is_empty(self, r, c):
        return self.chessboard.is_empty(r, c)

    def update_train_data(self):
        board, me, oppo, step, winner = self.chessboard.get_last_train_window_output()
        self.board.append(board)
        self.me.append(me)
        self.oppo.append(oppo)
        self.step.append(step)
        if winner is not None:
            length = len(self.board)
            win_length = int(length / 2 + length % 2)
            lose_length = int(length / 2)
            self.label = np.zeros((length, 1), dtype=np.float)
            self.weight = np.zeros((length, 1), dtype=np.float)
            self.label[1 - length % 2::2, 0] = 1
            self.weight[1 - length % 2::2, 0] = np.array([0.5 + x / (win_length * 2) for x in range(1, win_length + 1)])
            self.weight[length % 2::2, 0] = 1 / lose_length

    def get_train_data(self):
        return self.board, self.me, self.oppo, self.step, self.label, self.weight

    def move(self, r, c):
        status, winner = self.chessboard.move(r, c)
        if status == -1:
            raise ValueError("status is -1")
        if status == 0:
            self.current_player = self.chessboard.current_player
            self.debug_show = True
        elif status == 2:
            self.winner = GAME_PLAYER.PLAYER_TWO
        else:
            self.winner = winner

    def is_over(self):
        return self.winner is not None

    def is_playing(self):
        return self.is_play



