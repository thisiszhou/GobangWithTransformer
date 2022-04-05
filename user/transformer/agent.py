from chess_board.base_board import ChessBoard


class Transformer_Gobang(object):
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def predict(self, chessboard: ChessBoard):

