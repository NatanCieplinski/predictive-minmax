import numpy as np
import chess

from engine.minimax import Minimax
from engine.heuristics import Heuristics
from engine.datasets import Datasets

class MoveEvaluator:
    @staticmethod
    def find_best_move(board, heuristic_depth, is_white_turn, predictor):
        best_move = None
        max_eval = -np.inf

        for move in board.legal_moves:
            board.push(move)

            current_move_evaluation = Minimax.minimax(board, heuristic_depth - 1, -np.inf, np.inf, False, is_white_turn, predictor)

            if not predictor:
                MoveEvaluator.save_heuristics_into_dataset(board, current_move_evaluation, is_white_turn)

            board.pop()

            if current_move_evaluation > max_eval:
                max_eval = current_move_evaluation
                best_move = move

        return best_move

    @staticmethod
    def predict_best_move(board, predictor, is_white_turn):
        best_move = None
        max_eval = -np.inf

        for move in board.legal_moves:
            board.push(move)
            
            h1 = Heuristics.material_heuristic(board) if is_white_turn == chess.WHITE else - Heuristics.material_heuristic(board)
            h2 = Heuristics.piece_square_table_heuristic(board) if is_white_turn == chess.WHITE else - Heuristics.piece_square_table_heuristic(board)
            h3 = Heuristics.attack_heuristic(board) if is_white_turn == chess.WHITE else - Heuristics.attack_heuristic(board)
            features = predictor.scale([[h1, h2, h3]])

            current_move_evaluation = predictor.model.predict(features)[0][0]

            board.pop()

            if current_move_evaluation > max_eval:
                max_eval = current_move_evaluation
                best_move = move

        return best_move



    @staticmethod
    def save_heuristics_into_dataset(board, current_move_evaluation, is_white_turn):
        h1 = Heuristics.material_heuristic(board) if is_white_turn == chess.WHITE else - Heuristics.material_heuristic(board)
        h2 = Heuristics.piece_square_table_heuristic(board) if is_white_turn == chess.WHITE else - Heuristics.piece_square_table_heuristic(board)
        h3 = Heuristics.attack_heuristic(board) if is_white_turn == chess.WHITE else - Heuristics.attack_heuristic(board)
        
        instance = [
            h1,
            h2 * 0.5,
            h3 * 0.3,
            current_move_evaluation
        ]
        Datasets.HEURISTICS_DATA.append(instance)
