import time
import chess

from engine.view import View
from engine.move_evaluator import MoveEvaluator
from config import Config
from engine.datasets import Datasets

def main():
    players = [Config.WHITE_PLAYER, Config.BLACK_PLAYER]
    human_color = None
    if "human" in players:
        human_color = "White" if players[0] == "human" else "Black"

    board = chess.Board()
    display = View(board)

    white_turn = True
    game_over = False

    Datasets.load()
    while not game_over:
        for player in players:
            if player == "heuristic":
                move = MoveEvaluator.find_best_move(board, Config.DEPTH, white_turn)
                board.push(move)

            if player == "human":
                move_is_valid = True
                while move_is_valid:
                    print(f'{human_color} move: ')
                    move = input()
                    try:
                        board.push_uci(move)
                    except BaseException:
                        move_is_valid = False

            if Config.SHOW_BOARD:
                display.update_board(board)
            if Config.MOVE_DELAY:
                time.sleep(0.5)

            if board.is_game_over():
                game_over = True
                break

            white_turn = not white_turn

    Datasets.dump()
    return board.result()

if __name__ == "__main__":
    main()
