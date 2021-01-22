import time
import chess
import logging, sys

from engine.view import View
from engine.move_evaluator import MoveEvaluator
from config import Config
from engine.datasets import Datasets
from engine.predictor import Predictor

def play_match(players, predictor = None):
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

            white_turn = not white_turn

    print('Match result: '+str(board.result()))

def main():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    players = [Config.WHITE_PLAYER, Config.BLACK_PLAYER]
    human_color = None
    if "human" in players:
        human_color = "White" if players[0] == "human" else "Black"

    Datasets.load()

    if "ai" in players:
        predictor = Predictor()
        predictor.load_model()

    for counter in range(1000): 
        logging.info('Playing match number %d', counter)
        start = time.time()
        if "ai" in players:
            play_match(players, predictor)
            end = time.time()
            logging.info('The match lasted %d seconds', end - start)
            predictor.update_dataset()
            predictor.train_model()
            predictor.save_model()
        else:
            play_match(players)
            end = time.time()
            logging.info('The match lasted %d seconds', end - start)
    
        Datasets.dump()


if __name__ == "__main__":
    main()
