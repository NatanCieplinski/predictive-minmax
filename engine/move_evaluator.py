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

            MoveEvaluator.update_heuristics_dataset(board, current_move_evaluation)
            board.pop()

            if current_move_evaluation > max_eval:
                max_eval = current_move_evaluation
                best_move = move

        print(move)
        return best_move

    @staticmethod
    def update_heuristics_dataset(board, current_move_evaluation):
        instance = [
            Heuristics.material_heuristic(board),
            Heuristics.piece_square_table_heuristic(board),
            Heuristics.attack_heuristic(board),
            current_move_evaluation
        ]

        Datasets.HEURISTICS_DATA.append(instance)

