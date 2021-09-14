import pygame
import numpy as np
from chessboard import cons
from chessboard.utils import if_win

REC_SIZE = 50  # 棋子移动空间的大小
CHESS_RADIUS = REC_SIZE // 2 - 2
CHESS_LEN = 15  # 棋子长度
MAP_WIDTH = CHESS_LEN * REC_SIZE  # 地图宽度
MAP_HEIGHT = CHESS_LEN * REC_SIZE  # 地图高度
INFO_WIDTH = 200  # 信息宽度
BUTTON_WIDTH = 140  # 按钮宽度
BUTTON_HEIGHT = 50  # 按钮高度
SCREEN_WIDTH = MAP_WIDTH + INFO_WIDTH  # 屏幕宽度
SCREEN_HEIGHT = MAP_HEIGHT  # 屏幕高度
SEARCH_DEPTH = 5  # 搜索深度5
LIMITED_MOVE_NUM = 10  # 限制步数10


class ChessBoard(object):
    def __init__(self, width: int, height: int):
        """
        初始化，设置
        :param width: 棋盘宽
        :param height:  棋盘高
        """
        self.width = width
        self.height = height
        self.map = np.zeros((height, width), dtype=int)
        self.steps = []

    def reset(self):
        """
        复位函数
        """
        self.map[...] = 0
        self.steps = []

    def isEmpty(self, x: int, y: int):
        """
        判断是否为空
        :param x:
        :param y:
        :return:
        """
        return self.map[y][x] == 0
