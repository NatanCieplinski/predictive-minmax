import time
import chess

from engine.view import View
from engine.move_evaluator import MoveEvaluator
from config import Config
from engine.datasets import Datasets
from engine.predictor import Predictor

def main():
    players = [Config.WHITE_PLAYER, Config.BLACK_PLAYER]
    human_color = None
    if "human" in players:
        human_color = "White" if players[0] == "human" else "Black"

    Datasets.load()
    predictor = Predictor()
    predictor.load_model()

    for counter in range(1): 
        board = chess.Board()
        if Config.SHOW_BOARD:
            display = View(board)

        white_turn = True
        game_over = False


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

                if player == "ai":
                    move = MoveEvaluator.predict_best_move(board, predictor, white_turn)
                    board.push(move)

                if Config.SHOW_BOARD:
                    display.update_board(board)

                if board.is_game_over():
                    game_over = True
                    break

                white_turn = not white_turn
        print(board.result())

    Datasets.dump()
    predictor.save_model()

if __name__ == "__main__":
    main()
