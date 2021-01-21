import json
import os

class Datasets:
    EVALUATED_BOARDS = {}

    @staticmethod
    def dump():
        with open('./datasets/evaluated_boards.json', 'w') as file:
	        json.dump(Datasets.EVALUATED_BOARDS, file)
    
    @staticmethod
    def load():
        if os.path.exists('./datasets/evaluated_boards.json'):
            with open('./datasets/evaluated_boards.json', 'r') as file:
                Datasets.EVALUATED_BOARDS = json.load(file)

