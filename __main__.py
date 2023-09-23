# -*- coding: utf-8 -*-
#==============================================================================
# INTELIGÊNCIA ARTIFICIAL APLICADA
# REDES NEURAIS - SEMANA 6
# REDE NEURAL TEMPORAL (RNT)
# PROF. EDSON RUSCHEL
#==============================================================================

#==============================================================================
# IMPORTAÇÃO DE BIBLIOTECAS
#==============================================================================
from typing import List
import pandas as pd

from dtos import ConfigParams, Results
from test import TemporalNeuralNetworkTest


training_data_path = './data/learning-data.csv'
test_data_path = './data/test-data.csv'

params = ConfigParams()

train_df = pd.read_csv(training_data_path)
test_df = pd.read_csv(test_data_path)

results: List[Results] = []
possible_combinations: List[ConfigParams] = [
    ConfigParams(
        preview_window_size=100,
        lose_function='mean_squared_error',
        optimizator='adam',
        neurons_qty_on_temporal_layer=10,
        neurons_qty_on_dense_layer=50,
        epochs=50,
        lote=50,
        test_id='test-1'
    ),
    ConfigParams(
        preview_window_size=100,
        lose_function='mean_absolute_error',
        optimizator='adam',
        neurons_qty_on_temporal_layer=20,
        neurons_qty_on_dense_layer=50,
        epochs=50,
        lote=50,
        test_id='test-2'
    ),
    ConfigParams(
        preview_window_size=100,
        lose_function='binary_crossentropy',
        optimizator='adam',
        neurons_qty_on_temporal_layer=30,
        neurons_qty_on_dense_layer=50,
        epochs=50,
        lote=50,
        test_id='test-3'
    ),
]

for param_combination in possible_combinations:
    results.append(
        TemporalNeuralNetworkTest()
            .setup(param_combination)
            .run(train_df, test_df)
    )

TemporalNeuralNetworkTest.save_executions_as_csv(results, possible_combinations)