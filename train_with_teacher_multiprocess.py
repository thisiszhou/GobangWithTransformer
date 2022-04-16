from chess_board.game import Game
from user.transformer.agent import Transformer_Gobang
from user.beta.agent import ChessAI
from concurrent.futures import ProcessPoolExecutor


def main():
    epoch = 10000000
    model_file_load = "data/model_teacher_train.pkl"
    model_file_save = "data/model_teacher_train.pkl"

    # 5
    teacher = ChessAI(15)
    agent = Transformer_Gobang(15, 15, train=True, model_file=model_file_load)

    game = Game(agent.predict_step, teacher.predict_step, 5, "bg_picture/bgmain.jpg", "bg_picture/bgbian.jpg",
                "bg_picture/manwin.jpg", "bg_picture/comwin.jpg",
                show_board=False, collect_train_data=True)
    params = [(agent, epoch, 2, 10, model_file_save), (agent, epoch, 2, 10, model_file_save)]
    with ProcessPoolExecutor(max_workers=2) as executor:
        datas = executor.map(game.play, *params)
        print("datas", datas)


if __name__ == '__main__':
    main()





