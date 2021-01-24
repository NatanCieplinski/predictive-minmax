import json
import os
import numpy as np
import pandas as pd
import csv
import logging

class Datasets:
    HEURISTICS_DATA = []

    @staticmethod
    def dump():
        pd.DataFrame(Datasets.HEURISTICS_DATA).drop_duplicates().to_csv("./datasets/heuristics.csv", index=False, header=None)
        logging.info('Saving heuristics values...')
    
    @staticmethod
    def load():
        if os.path.exists('./datasets/heuristics.csv'):
            with open('./datasets/heuristics.csv', newline='') as file:
                logging.info('Loading heuristics values...')
                reader = csv.reader(file)
                Datasets.HEURISTICS_DATA = list(reader)
                print("Dataset entries: "+str(len(Datasets.HEURISTICS_DATA)))

