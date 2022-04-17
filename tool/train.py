import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game.game import Game
from user.transformer.agent import Transformer_Gobang
from user.beta.agent import ChessAI
from loguru import logger
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import time

MODEL_FILE_LOAD = "data/model_teacher_train.pkl"
MODEL_FILE_SAVE = "data/model_teacher_train.pkl"
CHESSBOARD_SIZE = 15
SHOW_LOSS_STEP = 4
SAVE_MODEL_STEP = 20
PROCESS = 4
EPOCH = 100000


def main():
    teacher = ChessAI(CHESSBOARD_SIZE)
    agent = Transformer_Gobang(CHESSBOARD_SIZE, CHESSBOARD_SIZE, train=True, model_file=MODEL_FILE_LOAD)
    game = Game(CHESSBOARD_SIZE)
    total_loss = -1.
    winner_num = [0, 0]
    player_num = defaultdict(int)
    cepoch = 0
    start_time = time.time()
    win_model = [0, 0]
    for i in range(EPOCH):
        if i % 2 == 0:
            player = {1: "Transformer", -1: "Teacher"}
            game.set_player(agent.predict_step, teacher.predict_step)
        else:
            game.set_player(teacher.predict_step, agent.predict_step)
            player = {1: "Teacher", -1: "Transformer"}
        train_datas = []
        if PROCESS > 1:
            with ProcessPoolExecutor(max_workers=PROCESS) as executor:
                output = executor.map(game.play, [x for x in range(PROCESS)])
                train_datas = output
        else:
            train_datas.append(game.play(0))
        for data, winner in train_datas:
            loss = agent.train(data)
            cepoch += 1
            if total_loss < 0:
                total_loss = loss
            else:
                total_loss = total_loss * 0.95 + loss * 0.05
            if winner == 1:
                winner_num[0] += 1
            else:
                winner_num[1] += 1
            player_num[player[winner]] += 1
            if player[winner] == "Transformer":
                if winner == 1:
                    win_model[0] += 1
                else:
                    win_model[1] += 1
            if cepoch % SHOW_LOSS_STEP == 0:
                spend_time = time.time() - start_time
                start_time = time.time()
                logger.info(f"Train epoch: {cepoch}, loss: {total_loss}, spend time: {round(spend_time, 2)}s.")
                tw = player_num["Teacher"]
                sw = player_num["Transformer"]
                win_rate = sw / (tw + sw)
                logger.info(f"first win: {winner_num[0]}, second win: {winner_num[1]}, Teacher win: {tw} "
                            f"Transformer win: {sw}, rate: {round(win_rate, 2) * 100}%, Transformer first win: "
                            f"{win_model[0]}, Transformer second win: {win_model[1]}.")
            if cepoch % SAVE_MODEL_STEP == 0:
                agent.save_model(MODEL_FILE_SAVE)


if __name__ == '__main__':
    main()





