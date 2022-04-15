from chess_board.game import Game
import pygame
from loguru import logger
from user.random.rand import Rand
from user.transformer.agent import Transformer_Gobang
import pickle

model_file = "data/model8.pkl"
# model_file = None

agent = Transformer_Gobang(8, 8, train=True, model_file=model_file)
game = Game(agent.predict_step, agent.predict_step, 4, "bg_picture/bgmain.jpg", "bg_picture/bgbian.jpg", "bg_picture/manwin.jpg", "bg_picture/comwin.jpg", show_board=False, collect_train_data=True)
train_data = game.play()
# print([x[-1] for x in train_data])
# print("------")
# print([x[-2] for x in train_data])

# train_data = [x[:2] for x in train_data]
# train_data1 = [x[:1] for x in train_data]
# train_data2 = [x[1:2] for x in train_data]
print()
for i in range(1000):
    loss = agent.train(train_data)
    print(loss)
# print(agent.predict(*train_data1[:3]))
# print(train_data1[3:])
# print(agent.predict(*train_data2[:3]))
# print(train_data2[3:])