import time

import numpy as np

import chess as ch

from engine.view import View
from engine.evaluator import Evaluator


max_depth = 10

evaluated_boards = {}


def minimax(board, depth, alpha, beta, max_player, player_color):
    if depth == 0 or board.is_game_over():
        fen_description = board.fen()

        final_move_score = -10000 if player_color == ch.BLACK else 10000

        if not (fen_description in evaluated_boards):
            evaluated_boards[fen_description] = Evaluator.evaluate_board(board)
            if board.is_game_over():
                if max_player:
                    evaluated_boards[fen_description] += -final_move_score
                else:
                    evaluated_boards[fen_description] += final_move_score

        return - \
            evaluated_boards[fen_description] if player_color == ch.BLACK else evaluated_boards[fen_description]

    if max_player:
        max_evaluation = -np.inf

        for move in board.legal_moves:
            if move.promotion is not None and move.promotion != 5:
                continue
            board.push(move)
            evaluation = minimax(
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
            evaluation = minimax(
                board, depth - 1, alpha, beta, True, player_color)
            board.pop()
            min_evaluation = min(min_evaluation, evaluation)
            beta = min(beta, evaluation)
            if beta <= alpha:
                break
        return min_evaluation


def get_heuristic_move(board, heuristic_depth, color):
    max_move = None
    max_eval = -np.inf

    for move in board.legal_moves:
        board.push(move)

        eval = minimax(board, heuristic_depth - 1, -
                       np.inf, np.inf, False, color)

        board.pop()

        if eval > max_eval:
            max_eval = eval
            max_move = move

    return max_move


def play_game(
        heuristic_depth,
        player_white="ai",
        player_black="heuristic",
        show_display=True,
        delayed=False):
    board = ch.Board()

    players = [player_white, player_black]
    white_turn = True
    game_over = False

    display = View(board)
    display.create_board()

    while not game_over:
        for player in players:
            if player == "heuristic":
                move = get_heuristic_move(board, heuristic_depth, white_turn)
                board.push(move)

            if player == "human":
                is_valid = False
                while not is_valid:
                    print('Black move: ')
                    move = input()
                    try:
                        board.push_uci(move)
                        is_valid = True
                    except BaseException:
                        is_valid = False

            if show_display:
                display.update_board(board)
            if delayed:
                time.sleep(0.5)

            if board.is_game_over():
                game_over = True
                break

            white_turn = not white_turn

    return board.result()


play_game(3, "human", "heuristic")