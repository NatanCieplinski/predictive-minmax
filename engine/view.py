from chessboard import display
class View:
    def __init__(self, board):
        self.fen = board.fen()
        display.start(self.fen)
    def update_board(self, board):
        display.update(board.fen())
