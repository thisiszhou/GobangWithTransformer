from chess_board.game import Game
from user.transformer.agent import Transformer_Gobang
from user.beta.agent import ChessAI

epoch = 10000000
model_file_load = "data/model_teacher_train.pkl"
model_file_save = "data/model_teacher_train.pkl"

# 5
teacher = ChessAI(15)
agent = Transformer_Gobang(15, 15, train=True, model_file=model_file_load)


game = Game(agent.predict_step, teacher.predict_step, 5, "bg_picture/bgmain.jpg", "bg_picture/bgbian.jpg", "bg_picture/manwin.jpg", "bg_picture/comwin.jpg",
            show_board=False, collect_train_data=True)
train_data = game.play(agent, epoch, 2, 10, model_file_save, wait=0.)
agent.save_model(model_file_save)

# game = Game(teacher.predict_step, agent.predict_step, 5, "bg_picture/bgmain.jpg", "bg_picture/bgbian.jpg", "bg_picture/manwin.jpg", "bg_picture/comwin.jpg",
#             show_board=True, collect_train_data=True)
# train_data = game.play(agent, epoch, 1, 2, model_file_save, wait=0.5)
# agent.save_model("data")


