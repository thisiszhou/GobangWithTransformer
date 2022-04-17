from chess_board.base_board import ChessBoard
from user.transformer.net import Transformer
from torch import Tensor
from loguru import logger
import torch
import numpy as np
import os
from user.random.rand import Rand
import random


class Transformer_Gobang(object):
    def __init__(self, chess_row, chess_col, model_file, train=True):
        self.chess_row = chess_row
        self.chess_col = chess_col
        if os.path.exists(model_file):
            self.model = torch.load(model_file)
            logger.info(f"Load model from {model_file}!")
        else:
            self.model = Transformer(5, 10, chess_row)
            logger.info("Random init model!")
            if not os.path.exists(os.path.dirname(model_file)):
                os.mkdir(os.path.dirname(model_file))
        self.training = train
        if not train:
            self.model.eval()
        if train:
            self.model.train()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # test multiprocess
        self.process_num = 0

    def add_process_num(self):
        self.process_num += 1
        print("process_num:", self.process_num)

    def predict(self, board, me, oppo):
        return self.model(Tensor(board), Tensor(me), Tensor(oppo))

    def predict_step(self, chessboard: ChessBoard, rand_p=0.00, rule=True, first_random=True):
        if len(chessboard.steps) == 0 and first_random:
            return chessboard.get_random_first_step()
        if self.training and random.random() < rand_p:
            return Rand(chessboard)

        board, me, oppo = chessboard.get_last_pred_window_output()
        # print(f"----{len(chessboard.steps)}-----")
        # print("board:", board.shape)
        # print("me:", me.shape)
        # print("oppo:", oppo.shape)
        self.model.eval()
        map = self.predict(board, me, oppo).detach().numpy()
        assert map.shape[0] == 1, "predict batch size is not 1!"
        map += chessboard.get_valid_board()
        win_step = chessboard.search_current_player_certain_step()
        if rule and len(win_step) > 0:
            map[[0 for _ in win_step], [x[0] for x in win_step], [x[1] for x in win_step]] += 10
            # print("map:", map)
            # print("-"*20)
            # for step in win_step:
            #     print("step:", step, map[0, step[0], step[1]])
        output = np.unravel_index(map.argmax(), map.shape)
        return output[1:]

    def train(self, train_data, batch_size=256):
        board, me, oppo, step, label, w = train_data
        start = 0
        loss_ = -1
        self.model.train()
        while start < len(board):
            self.optimizer.zero_grad()
            map = self.predict(board[start: start + batch_size],
                               me[start: start + batch_size],
                               oppo[start: start + batch_size])
            index1 = [x[0] for x in step[start: start + batch_size]]
            index2 = [x[1] for x in step[start: start + batch_size]]
            index0 = [x for x in range(len(index1))]
            pred: Tensor = map[index0, index1, index2].unsqueeze(-1)
            loss_: Tensor = self.loss(pred, label[start: start + batch_size], w)
            loss_.backward()
            self.optimizer.step()
            start += batch_size
        return loss_.detach().numpy()

    def loss(self, pred: Tensor, label, w):
        label = torch.tensor(label)
        w = torch.tensor(w)
        pred = torch.clip(pred, min=0.00001, max=0.99999)
        return (- label * torch.log(pred) * w
                - (1 - label) * torch.log(1 - pred) * w
                ).mean()

    def save_model(self, save_file="data/model.pkl"):
        torch.save(self.model, save_file)
        print(f"model saved file: {save_file}")
