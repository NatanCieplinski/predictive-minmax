import numpy as np

from engine.minimax import Minimax
from engine.heuristics import Heuristics
from engine.datasets import Datasets

class MoveEvaluator:
    @staticmethod
    def find_best_move(board, heuristic_depth, color):
        best_move = None
        max_eval = -np.inf

        for move in board.legal_moves:
            board.push(move)

            current_move_evaluation = Minimax.minimax(board, heuristic_depth - 1, -
                        np.inf, np.inf, False, color)

            MoveEvaluator.update_heuristics_dataset(board, current_move_evaluation, color)
            board.pop()

            if current_move_evaluation > max_eval:
                max_eval = current_move_evaluation
                best_move = move

        print(move)
        return best_move

    @staticmethod
    def update_heuristics_dataset(board, current_move_evaluation, color):
        h1 = Heuristics.material_heuristic(board) if color == chess.WHITE else - Heuristics.material_heuristic(board)
        h2 = Heuristics.piece_square_table_heuristic(board) if color == chess.WHITE else - Heuristics.piece_square_table_heuristic(board)
        h3 = Heuristics.attack_heuristic(board) if color == chess.WHITE else - Heuristics.attack_heuristic(board)
        
        instance = [
            h1,
            h2 * 0.5,
            h3 * 0.3,
            current_move_evaluation
        ]
        Datasets.HEURISTICS_DATA.append(instance)
