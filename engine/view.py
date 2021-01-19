from chessboard import display

class View:
    def __init__(self, board):
        self.fen = board.fen()
    def create_board(self):
        display.start(self.fen)
    def update_board(self, board):
        display.update(board.fen())
