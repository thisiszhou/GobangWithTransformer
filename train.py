from chess_board.game import Game
import pygame
from loguru import logger
from user.random.rand import Rand
from user.transformer.agent import Transformer_Gobang

epoch = 10000000
# model_file = "data/model.pkl"
model_file = None

# 4
agent = Transformer_Gobang(8, 8, train=True, model_file=model_file)
game = Game(agent.predict_step, agent.predict_step, 4, "bg_picture/bgmain.jpg", "bg_picture/bgbian.jpg", "bg_picture/manwin.jpg", "bg_picture/comwin.jpg",
            show_board=False, collect_train_data=True)
train_data = game.play(agent, epoch, 10, 100, "data", wait=0.)
agent.save_model("data")

# # 5
# agent = Transformer_Gobang(15, 15, train=True, model_file=model_file)
# game = Game(agent.predict_step, agent.predict_step, 5, "bg_picture/bgmain.jpg", "bg_picture/bgbian.jpg", "bg_picture/manwin.jpg", "bg_picture/comwin.jpg",
#             show_board=False, collect_train_data=True)
# train_data = game.play(agent, epoch, 10, 100, "data", wait=0.)
# agent.save_model("data")

# agent = Transformer_Gobang(15, 15, train=True, model_file=model_file)
# game = Game(agent.predict_step, agent.predict_step, 5, "bg_picture/bgmain.jpg", "bg_picture/bgbian.jpg", "bg_picture/manwin.jpg", "bg_picture/comwin.jpg",
#             show_board=True, collect_train_data=True)
# train_data = game.play(agent, epoch, 1, 500, "data", wait=0.8)
# agent.save_model("data")


