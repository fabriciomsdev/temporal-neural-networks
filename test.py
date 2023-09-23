from typing import List, Dict
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import secrets

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from dtos import ConfigParams, Results


class TemporalNeuralNetworkTest:
    def __init__(self):
        self.params = ConfigParams()
        self.results = Results()

    def setup(self, params):
        self.params = params
        return self 

    def run(self, train_df, test_df):
        print(f'id: {self.params.test_id} | Running test with params: {self.params.__dict__}')
        results = Results()
        epochs = self.params.epochs
        lote = self.params.lote

        train_data_scaled, test_data_scaled, scaler = self._normalize_data(train_df, test_df)
        
        # Criar os conjuntos de treinamento e teste
        X_train, y_train = self._create_dataset(train_data_scaled, self.params.preview_window_size)
        X_test, y_test = self._create_dataset(test_data_scaled, self.params.preview_window_size)

        # Reshape dos dados para o formato esperado pela LSTM [amostras, janela de tempo, características]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_train.shape[1], 1))

        # Criar o modelo da rede neural
        model = self._create_model()

        #==============================================================================
        # TREINAMENTO DA REDE NEURAL TEMPORAL
        #==============================================================================
        # Treinar o modelo
        model.fit(X_train, y_train, epochs=epochs, batch_size=lote, verbose=1)

        #==============================================================================
        # REALIZAR PREVISÕES DO MODELO
        #==============================================================================
        # Fazer previsões
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        #==============================================================================
        # POSPROCESSAMENTO DOS DADOS
        #==============================================================================
        # Desfazer a normalização dos dados
        train_predict = scaler.inverse_transform(train_predict)
        y_train = scaler.inverse_transform([y_train])
        test_predict = scaler.inverse_transform(test_predict)
        y_test = scaler.inverse_transform([y_test])

        #==============================================================================
        # APRESENTAÇÃO GRÁFICA DOS RESULTADOS
        #==============================================================================
        
        # Plotar os resultados em dois gráficos separados
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        # Gráfico 1: Dados de treinamento
        ax1.plot(y_train[0], label='Dados de treinamento reais')
        ax1.plot(train_predict[:, 0], label='Previsões de treinamento')
        ax1.legend()
        ax1.set_title('Dados de Treinamento')
        ax1.set_xlabel('Período')
        ax1.set_ylabel('Valor (U$)')
        ax1.grid(True)

        # Gráfico 2: Dados de teste
        ax2.plot(y_test[0], label='Dados de teste reais')
        ax2.plot(test_predict[:, 0], label='Previsões de teste')
        ax2.legend()
        ax2.set_title('Dados de Teste')
        ax2.set_xlabel('Período')
        ax2.set_ylabel('Valor (U$)')
        ax2.grid(True)

        # Ajustar espaçamento entre subplots
        plt.tight_layout()

        # Exibir os gráficos
        if self.params.show_results:
            plt.show()
        
        image_path = f'./results/images/test-{self.params.test_id}.png'
        results.graph = image_path

        plt.savefig(image_path)

        results.mean_squared_error_train = mean_squared_error(y_train[0], train_predict[:, 0])
        results.mean_squared_error_test = mean_squared_error(y_test[0], test_predict[:, 0])

        print('\n' + '=' * 70)
        print("Erro Médio Quadrático (MSE) - Treinamento: {:.4f}".format(results.mean_squared_error_train))
        print("Erro Médio Quadrático (MSE) - Teste: {:.4f}".format(results.mean_squared_error_test))
        print('\n')

        results.mean_absolute_error_train = mean_absolute_error(y_train[0], train_predict[:, 0])
        results.mean_absolute_error_test = mean_absolute_error(y_test[0], test_predict[:, 0])

        print('=' * 70)
        print("Erro Médio Absoluto (MAE) - Treinamento: {:.4f}".format(results.mean_absolute_error_train))
        print("Erro Médio Absoluto (MAE) - Teste: {:.4f}".format(results.mean_absolute_error_test))
        print('\n')

        results.r2_train = r2_score(y_train[0], train_predict[:, 0])
        results.r2_test = r2_score(y_test[0], test_predict[:, 0])

        print('=' * 70)
        print("Coeficiente de Determinação (R²) - Treinamento: {:.4f}".format(results.r2_train))
        print("Coeficiente de Determinação (R²) - Teste: {:.4f}".format(results.r2_test))
        print('\n' + '=' * 70)

        return results

    def _create_model(self):
        model = Sequential()
        model.add(LSTM(self.params.neurons_qty_on_temporal_layer, input_shape=(self.params.preview_window_size, 1)))
        model.add(Dense(self.params.neurons_qty_on_dense_layer))
        model.compile(loss=self.params.lose_function, optimizer=self.params.optimizator)
        return model

    def _create_dataset(self, dataset, window_size=1):
        X, Y = [], []
        for i in range(len(dataset) - window_size):
            window = dataset[i:(i + window_size), 0]
            X.append(window)
            Y.append(dataset[i + window_size, 0])
        return np.array(X), np.array(Y)

    def _normalize_data(self, df_training, df_test, scaler=MinMaxScaler(feature_range=(0, 1))):
        # Filtrar apenas a coluna 'Valor' nos dados de treinamento e teste
        train_data = df_training[['Valor']].values
        test_data = df_test[['Valor']].values

        # Normalizar os dados de treinamento entre 0 e 1
        train_data_scaled = scaler.fit_transform(train_data)
        test_data_scaled = scaler.transform(test_data)

        return train_data_scaled, test_data_scaled, scaler


    @staticmethod
    def save_executions_as_csv(results: List[Results], possible_combinations: List[ConfigParams]):
        rows_of_result = []

        for index, result in enumerate(results):
            data = result.__dict__
            data.update(possible_combinations[index].__dict__)
            rows_of_result.append(data)

        df = pd.DataFrame(rows_of_result)
        now = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')

        df.to_csv(f'./results/tables/results-{now}.csv', index=False)

