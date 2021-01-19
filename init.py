import pathlib
import pickle
import random
import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import chess as ch
from IPython.display import display, HTML, clear_output

# https://python-chess.readthedocs.io/en/latest/core.html
# chess.PAWN:   chess.PieceType = 1
# chess.KNIGHT: chess.PieceType = 2
# chess.BISHOP: chess.PieceType = 3
# chess.ROOK:   chess.PieceType = 4
# chess.QUEEN:  chess.PieceType = 5
# chess.KING:   chess.PieceType = 6

material_values = [100, 305, 333, 563, 950, 10000]


def material_heuristic(board):
    material_advantage = 0

    for piece_type in ch.PIECE_TYPES:
        white_pieces = board.pieces(piece_type, ch.WHITE)
        black_pieces = board.pieces(piece_type, ch.BLACK)
        material_advantage += material_values[piece_type - 1] * len(
            white_pieces) - material_values[piece_type - 1] * len(black_pieces)

    return material_advantage


piece_square_tables = []

# PAWN
piece_square_tables.append([0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 20, 30, 30, 20, 10, 10, 5, 5, 10, 25,
                            25, 10, 5, 5, 0, 0, 0, 20, 20, 0, 0, 0, 5, -5, -10, 0, 0, -10, -5, 5, 5, 10, 10, -20, -20, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0])
# KNIGHT
piece_square_tables.append([-50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 0, 0, 0, -20, -40, -30, 0, 10, 15, 15, 10, 0, -30, -30, 5, 15, 20, 20,
                            15, 5, -30, -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 10, 15, 15, 10, 5, -30, -40, -20, 0, 5, 5, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50, ])
# BISHOP
piece_square_tables.append([-20, -10, -10, -10, -10, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 10, 10, 5, 0, -10, -10, 5, 5, 10, 10, 5,
                            5, -10, -10, 0, 10, 10, 10, 10, 0, -10, -10, 10, 10, 10, 10, 10, 10, -10, -10, 5, 0, 0, 0, 0, 5, -10, -20, -10, -10, -10, -10, -10, -10, -20, ])
# ROOK
piece_square_tables.append([0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 10, 10, 10, 10, 5, -
                            5, 0, 0, 0, 0, 0, 0, -
                            5, -
                            5, 0, 0, 0, 0, 0, 0, -
                            5, -
                            5, 0, 0, 0, 0, 0, 0, -
                            5, -
                            5, 0, 0, 0, 0, 0, 0, -
                            5, -
                            5, 0, 0, 0, 0, 0, 0, -
                            5, 0, 0, 0, 5, 5, 0, 0, 0])
# QUEEN
piece_square_tables.append([-20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 5, 5, 5, 0, -10, -5, 0, 5, 5,
                            5, 5, 0, -5, 0, 0, 5, 5, 5, 5, 0, -5, -10, 5, 5, 5, 5, 5, 0, -10, -10, 0, 5, 0, 0, 0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20])
# KING
piece_square_tables.append([-30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -
                            50, -50, -40, -40, -30, -20, -30, -30, -40, -40, -30, -30, -20, -10, -20, -20, -20, -20, -20, -20, -10, 20, 20, 0, 0, 0, 0, 20, 20, 20, 30, 10, 0, 0, 10, 30, 20])


def piece_square_table_heuristic(board):
    positional_advantage = 0

    for piece_type in ch.PIECE_TYPES:

        for square in board.pieces(piece_type, ch.WHITE):
            position = 7 * (7 - int((square - 1) / 8)) + (square - 1) % 8
            positional_advantage += piece_square_tables[piece_type - 1][position]

        for square in ch.flip_vertical(board.pieces(piece_type, ch.BLACK)):
            position = 7 * (7 - int((square - 1) / 8)) + (square - 1) % 8
            positional_advantage -= piece_square_tables[piece_type - 1][position]

    return positional_advantage


max_depth = 10

evaluated_boards = {}


def evaluate_board(board):
    return material_heuristic(board) + 0.5 * \
        piece_square_table_heuristic(board)


def minimax(board, depth, alpha, beta, max_player, player_color):
    if depth == 0 or board.is_game_over():
        fen_description = board.fen()

        final_move_score = -10000 if player_color == ch.BLACK else 10000

        if not (fen_description in evaluated_boards):
            evaluated_boards[fen_description] = evaluate_board(board)
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


def display_board(board, use_svg=True):
    if use_svg:
        html = board._repr_svg_()
    else:
        html = "<pre>" + str(board) + "</pre>"

    clear_output(wait=True)
    display(HTML(html))


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

    while not game_over:
        for player in players:
            if player == "heuristic":
                move = get_heuristic_move(board, heuristic_depth, white_turn)
                board.push(move)

            if player == "human":
                isValid = False
                while not isValid:
                    print('Black move: ')
                    move = input()
                    try:
                        board.push_uci(move)
                        isValid = True
                    except BaseException:
                        isValid = False

            if show_display:
                display_board(board)
            if delayed:
                time.sleep(0.5)

            if board.is_game_over():
                game_over = True
                break

            white_turn = not white_turn

    return board.result()


play_game(5, "human", "heuristic")