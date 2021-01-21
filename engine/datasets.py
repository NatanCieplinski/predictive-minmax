import json

class Datasets:
    EVALUATED_BOARDS = {}

    @staticmethod
    def dump():
        with open('./datasets/evaluated_boards.json', 'w') as evaluated_boards:
	        json.dump(Datasets.EVALUATED_BOARDS, evaluated_boards)
    
    @staticmethod
    def load():
        with open('./datasets/evaluated_boards.json', 'r') as evaluated_boards:
            Datasets.EVALUATED_BOARDS = json.load(evaluated_boards)

