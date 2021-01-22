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
            print('Saving boards...')
            json.dump(Datasets.EVALUATED_BOARDS, file)
        pd.DataFrame(Datasets.HEURISTICS_DATA).drop_duplicates().to_csv("./datasets/heuristics.csv", index=False, header=None)
        print('Saving heauristics values...')
    
    @staticmethod
    def load():
        if os.path.exists('./datasets/evaluated_boards.json'):
            with open('./datasets/evaluated_boards.json', 'r') as file:
                print('Loading boards...')
                Datasets.EVALUATED_BOARDS = json.load(file)

        if os.path.exists('./datasets/heuristics.csv'):
            with open('./datasets/heuristics.csv', newline='') as file:
                print('Loading heuristics values...')
                reader = csv.reader(file)
                Datasets.HEURISTICS_DATA = list(reader)

