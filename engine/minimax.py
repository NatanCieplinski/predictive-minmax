import chess
import numpy as np

from engine.datasets import Datasets
from engine.heuristics import Heuristics

class Minimax:
    @staticmethod
    def minimax(board, depth, alpha, beta, is_maximizing_player , player_color, predictor = False):
        if depth == 0 or board.is_game_over():
            fen_description = board.fen()

            final_move_score = -10000 if player_color == chess.BLACK else 10000

            if not board.is_checkmate():
                final_move_score = 10000 if player_color == chess.BLACK else -10000
            
            if not predictor:
                if not (fen_description in Datasets.EVALUATED_BOARDS):
                    Datasets.EVALUATED_BOARDS[fen_description] = Heuristics.evaluate_board(board)
                    if board.is_game_over():
                        if is_maximizing_player :
                            Datasets.EVALUATED_BOARDS[fen_description] += -final_move_score
                        else:
                            Datasets.EVALUATED_BOARDS[fen_description] += final_move_score

                return Datasets.EVALUATED_BOARDS[fen_description] if player_color == chess.WHITE else - Datasets.EVALUATED_BOARDS[fen_description]
            else:
                h1 = Heuristics.material_heuristic(board) if color == chess.WHITE else - Heuristics.material_heuristic(board)
                h2 = Heuristics.piece_square_table_heuristic(board) if color == chess.WHITE else - Heuristics.piece_square_table_heuristic(board)
                h3 = Heuristics.attack_heuristic(board) if color == chess.WHITE else - Heuristics.attack_heuristic(board)
                features = predictor.scale([[h1, h2, h3]])

                return predictor.model.predict(features)[0][0]


        if is_maximizing_player :

            max_evaluation = -np.inf

            for move in board.legal_moves:
                board.push(move)
                evaluation = Minimax.minimax(
                    board, depth - 1, alpha, beta, False, player_color)
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
                    board, depth - 1, alpha, beta, True, player_color)
                board.pop()
                min_evaluation = min(min_evaluation, evaluation)
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break

            return min_evaluation