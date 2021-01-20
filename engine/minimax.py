import chess
import numpy as np

from engine.datasets import Datasets
from engine.evaluator import Evaluator

class Minimax:
    @staticmethod
    def minimax(board, depth, alpha, beta, max_player, player_color):
        if depth == 0 or board.is_game_over():
            fen_description = board.fen()

            final_move_score = -10000 if player_color == chess.BLACK else 10000

            if not (fen_description in Datasets.EVALUATED_BOARDS):
                Datasets.EVALUATED_BOARDS[fen_description] = Evaluator.evaluate_board(board)
                if board.is_game_over():
                    if max_player:
                        Datasets.EVALUATED_BOARDS[fen_description] += -final_move_score
                    else:
                        Datasets.EVALUATED_BOARDS[fen_description] += final_move_score

            return - \
                Datasets.EVALUATED_BOARDS[fen_description] if player_color == chess.BLACK else Datasets.EVALUATED_BOARDS[fen_description]

        if max_player:
            max_evaluation = -np.inf

            for move in board.legal_moves:
                if move.promotion is not None and move.promotion != 5:
                    continue
                board.push(move)
                evaluation = Minimax.minimax(
                    board,
                    depth - 1,
                    alpha,
                    beta,
                    False,
                    player_color)
                board.pop()
                max_evaluation = max(max_evaluation, evaluation)
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break

            return max_evaluation
        else:
            min_evaluation = np.inf
            for move in board.legal_moves:
                if move.promotion is not None and move.promotion != 5:
                    continue
                board.push(move)
                evaluation = Minimax.minimax(
                    board, depth - 1, alpha, beta, True, player_color)
                board.pop()
                min_evaluation = min(min_evaluation, evaluation)
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
            return min_evaluation