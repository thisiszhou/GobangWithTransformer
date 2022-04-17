import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game.visual_game import VisualGame
from user.transformer.agent import Transformer_Gobang
from user.beta.agent import ChessAI

MODEL_FILE = "data/model_teacher_train.pkl"
CHESSBOARD_SIZE = 15
TRAIN = False


if __name__ == "__main__":
    teacher = ChessAI(CHESSBOARD_SIZE)
    agent = Transformer_Gobang(CHESSBOARD_SIZE, CHESSBOARD_SIZE, train=True, model_file=MODEL_FILE)
    game = VisualGame(play1={"name": "agent", "method": agent.predict_step})
    game.train = TRAIN
    game.play(agent=agent, wait=0.2, model_file=MODEL_FILE)
