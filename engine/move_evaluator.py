import numpy as np

from engine.minimax import Minimax

class MoveEvaluator:
    @staticmethod
    def find_best_move(board, heuristic_depth, color):
        best_move = None
        max_eval = -np.inf

        for move in board.legal_moves:
            board.push(move)
            current_move_evaluation = Minimax.minimax(board, heuristic_depth - 1, -
                        np.inf, np.inf, False, color)
            board.pop()

            if current_move_evaluation > max_eval:
                max_eval = current_move_evaluation
                best_move = move

        return best_move