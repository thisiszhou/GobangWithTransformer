from chess_board.base_board import ChessBoard
import random


def Rand(chessboard: ChessBoard):
    r = random.randint(0, chessboard.row - 1)
    c = random.randint(0, chessboard.col - 1)
    while not chessboard.is_empty(r, c):
        r = random.randint(0, chessboard.row - 1)
        c = random.randint(0, chessboard.row - 1)
    return r, c

