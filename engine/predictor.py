import keras
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from engine.datasets import Datasets
from config import Config

class Predictor:
    def __init__(self):
        self.dataset = self.prepare_dataset(Datasets.HEURISTICS_DATA)
        self.scaler = MinMaxScaler().fit(self.dataset)
        self.model = self.build_model([self.dataset.shape[1]-1])

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
        y = self.dataset.pop('H')
        X = self.dataset()

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)

        early_history = self.model.fit(
            X,
            y,
            epochs=1000,
            validation_split=0.2,
            verbose=verbose,
            callbacks=[early_stop]
        )

        return early_history

    def prepare_dataset(self, dataset):
        columns_names = []
        for counter in range(len(dataset[0])):
            columns_names.append('h'+str(counter))
        columns_names.append('H')
        dataframe = pd.DataFrame(dataset, columns = columns_names)
        return dataframe.drop_duplicates().apply(pd.to_numeric, errors='coerce', axis=1)

    def scale(self, instance):
        return self.scaler.transform(instance)

    def save_model(self):
        self.model.save('./models/depth'+str(Config.DEPTH)+'/model')

    def load_model(self):
        if os.path.exists('./models/depth'+str(Config.DEPTH)+'/model'):
            self.model = keras.models.load_model('./models/depth'+str(Config.DEPTH)+'/model')