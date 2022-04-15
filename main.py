from chess_board.game import Game
import pygame
from loguru import logger
from user.random.rand import Rand


game = Game("human", "human", 5, "bg_picture/bgmain.jpg", "bg_picture/bgbian.jpg", "bg_picture/manwin.jpg", "bg_picture/comwin.jpg")
# game = Game(Rand, Rand, 10, "bg_picture/bgmain.jpg", "bg_picture/bgbian.jpg", "bg_picture/manwin.jpg", "bg_picture/comwin.jpg")
game.play()
