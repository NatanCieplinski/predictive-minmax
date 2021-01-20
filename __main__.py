import time
import numpy as np
import chess

from engine.view import View
from engine.minimax import Minimax
from config import Config

def get_heuristic_move(board, heuristic_depth, color):
    max_move = None
    max_eval = -np.inf

    for move in board.legal_moves:
        board.push(move)

        evaluation = Minimax.minimax(board, heuristic_depth - 1, -
                       np.inf, np.inf, False, color)

        board.pop()

        if evaluation > max_eval:
            max_eval = evaluation
            max_move = move

    return max_move


def main():
    board = chess.Board()

    players = [Config.WHITE_PLAYER, Config.BLACK_PLAYER]
    white_turn = True
    game_over = False

    display = View(board)

    while not game_over:
        for player in players:
            if player == "heuristic":
                move = get_heuristic_move(board, Config.DEPTH, white_turn)
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

            if Config.SHOW_BOARD:
                display.update_board(board)
            if Config.MOVE_DELAY:
                time.sleep(0.5)

            if board.is_game_over():
                game_over = True
                break

            white_turn = not white_turn

    return board.result()

if __name__ == "__main__":
    main()
