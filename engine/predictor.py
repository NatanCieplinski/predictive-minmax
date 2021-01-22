import keras
import pandas as pd
import os
import logging

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers

from engine.datasets import Datasets
from config import Config

class Predictor:
    def __init__(self):
        self.dataset = self.prepare_dataset(Datasets.HEURISTICS_DATA)
        self.y = self.dataset.pop('H')
        self.X = self.dataset
        self.scaler = MinMaxScaler().fit(self.dataset)
        self.model = self.build_model([self.dataset.shape[1]])

    def build_model(self, input_shape):
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=input_shape),
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['mae', 'mse'])
        return model

    def train_model(self, patience=20, verbose=True):

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)

        early_history = self.model.fit(
            self.X,
            self.y,
            epochs=1000,
            validation_split=0.2,
            verbose=verbose,
            callbacks=[early_stop]
        )

        return early_history

    def prepare_dataset(self, dataset):
        logging.info('Preparing dataframe...')
        dataframe = pd.DataFrame(dataset, columns = Config.DATASET_COLUMNS)
        return dataframe.drop_duplicates().apply(pd.to_numeric, errors='coerce', axis=1)

    def scale(self, instance):
        return self.scaler.transform(instance)

    def update_dataset(self):
        logging.info('Updating dataframe...')
        self.dataset = self.prepare_dataset(Datasets.HEURISTICS_DATA)
        self.y = self.dataset.pop('H')
        self.X = self.dataset
        self.scaler = MinMaxScaler().fit(self.dataset)

    def save_model(self):
        logging.info('Saving model...')
        self.model.save('./models/depth'+str(Config.DEPTH)+'/model')

    def load_model(self):
        logging.info('Loading model...')
        if os.path.exists('./models/depth'+str(Config.DEPTH)+'/model'):
            self.model = keras.models.load_model('./models/depth'+str(Config.DEPTH)+'/model')