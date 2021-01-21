import json
import os
import numpy as np
import pandas as pd
import csv

class Datasets:
    EVALUATED_BOARDS = {}
    HEURISTICS_DATA = []

    @staticmethod
    def dump():
        with open('./datasets/evaluated_boards.json', 'w') as file:
	        json.dump(Datasets.EVALUATED_BOARDS, file)
        pd.DataFrame(Datasets.HEURISTICS_DATA).drop_duplicates().to_csv("./datasets/heuristics.csv", index=False, header=None)
    
    @staticmethod
    def load():
        if os.path.exists('./datasets/evaluated_boards.json'):
            with open('./datasets/evaluated_boards.json', 'r') as file:
                Datasets.EVALUATED_BOARDS = json.load(file)

        if os.path.exists('./datasets/heuristics.csv'):
            with open('heuristics.csv', newline='') as file:
                reader = csv.reader(file)
                data = list(reader)

