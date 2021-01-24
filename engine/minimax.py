import chess
import numpy as np

from engine.datasets import Datasets
from engine.heuristics import Heuristics

class Minimax:
    @staticmethod
    def minimax(board, depth, alpha, beta, is_maximizing_player , is_white_turn, predictor):
        if depth == 0 or board.is_game_over():

            if board.is_checkmate():
                final_move_score = -10000 if is_white_turn == chess.BLACK else 10000
            else:
                final_move_score = 10000 if is_white_turn == chess.BLACK else -10000
            
            if not predictor:
                evaluation = Heuristics.evaluate_board(board)
                if board.is_game_over():
                    if is_maximizing_player :
                        evaluation += -final_move_score
                    else:
                        evaluation += final_move_score

                return evaluation if is_white_turn == chess.WHITE else - evaluation

            else:
                h1 = Heuristics.material_heuristic(board) if is_white_turn == chess.WHITE else - Heuristics.material_heuristic(board)
                h2 = Heuristics.piece_square_table_heuristic(board) if is_white_turn == chess.WHITE else - Heuristics.piece_square_table_heuristic(board)
                h3 = Heuristics.attack_heuristic(board) if is_white_turn == chess.WHITE else - Heuristics.attack_heuristic(board)
                features = predictor.scale([[h1, h2, h3]])

                return predictor.model.predict(features)[0][0]


        if is_maximizing_player :

            max_evaluation = -np.inf

            for move in board.legal_moves:
                board.push(move)
                evaluation = Minimax.minimax(
                    board, depth - 1, alpha, beta, False, is_white_turn, predictor)
                board.pop()
                max_evaluation = max(max_evaluation, evaluation)
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break

            return max_evaluation

        else:

            min_evaluation = np.inf

            for move in board.legal_moves:
                board.push(move)
                evaluation = Minimax.minimax(
                    board, depth - 1, alpha, beta, True, is_white_turn, predictor)
                board.pop()
                min_evaluation = min(min_evaluation, evaluation)
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break

            return min_evaluation