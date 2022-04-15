from chess_board.game import Game
import pygame
from loguru import logger
from user.random.rand import Rand
from user.transformer.agent import Transformer_Gobang
from user.beta.agent import ChessAI

# # 5
# agent = Transformer_Gobang(15, 15, train=False, model_file="data/model.pkl")
# game = Game(agent.predict_step, agent.predict_step,   5, "bg_picture/bgmain.jpg", "bg_picture/bgbian.jpg",
# # game = Game(agent.predict_step,  "human",  4, "bg_picture/bgmain.jpg", "bg_picture/bgbian.jpg",
#             "bg_picture/manwin.jpg", "bg_picture/comwin.jpg", show_board=True, collect_train_data=False)
# game.play(wait=0.8)

# 4

agent = ChessAI(15)
game = Game(agent.predict_step, "human",   5, "bg_picture/bgmain.jpg", "bg_picture/bgbian.jpg",
# game = Game(agent.predict_step,  "human",  4, "bg_picture/bgmain.jpg", "bg_picture/bgbian.jpg",
            "bg_picture/manwin.jpg", "bg_picture/comwin.jpg", show_board=True, collect_train_data=False)
game.play(wait=0.5)
