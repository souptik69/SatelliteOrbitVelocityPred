import zipfile
import io
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tcn_sequence_models.data_processing.preprocessor import Preprocessor
from tcn_sequence_models.models import TCN_TCN_Attention
from data_preprocessing.DataPrepClass import SatelliteDataProcessor
from data_preprocessing.PreprocessingClass import SatelliteDataPreProcess

class ModelTrainer:
    def __init__(self, config_path, input_file):
        self.config_path = config_path
        self.input_file = input_file
        self.processor = None
        self.preprocess = None
        self.model = None

    def load_data(self, input_file):
        with zipfile.ZipFile(input_file, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            self.processed_df = pd.DataFrame()
            for file_name in file_list:
                with zip_ref.open(file_name) as file:
                    data = pd.read_csv(io.TextIOWrapper(file))
                    self.processed_df = self.processed_df.append(data)
        self.processor = SatelliteDataProcessor(self.processed_df)
        self.processor.load_data()
        self.processed_df = self.processor.process_data()

    def preprocess_data(self):
        self.preprocess = SatelliteDataPreProcess(self.processed_df)
        self.df = self.preprocess.preprocess_data()

    def combine_data(self):
        with zipfile.ZipFile(self.input_file, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            self.processor_list = []
            self.preprocess_list = []
            self.df_list = []
            for file_name in file_list:
                with zip_ref.open(file_name) as file:
                    # data = pd.read_csv(io.TextIOWrapper(file))
                    processor = SatelliteDataProcessor(file)
                    processor.load_data()
                    processed_df = processor.process_data()
                    preprocess = SatelliteDataPreProcess(processed_df)
                    df = preprocess.preprocess_data()
                    self.processor_list.append(processor)
                    self.preprocess_list.append(preprocess)
                    self.df_list.append(df)

    def test_train_dataset(self):
        split_ratio = 0.7
        input_seq_len = 120
        output_seq_len = 30
        features_input_encoder = ['position_x_sine_wave', 'position_x_cosine_wave', 'position_y_sine_wave', 'position_y_cosine_wave', 'position_z_sine_wave', 'position_z_cosine_wave', 'velocity_x_sine_wave', 'velocity_x_cosine_wave', 'velocity_y_sine_wave', 'velocity_y_cosine_wave', 'velocity_z_sine_wave', 'velocity_z_cosine_wave', 'Position Vector(X)', 'Position Vector(Y)', 'Position Vector(Z)', 'Velocity Vector(X)', 'Velocity Vector(Y)', 'Velocity Vector(Z)']
        # features_input_decoder = ['Position Vector(X)', 'Position Vector(Y)', 'Position Vector(Z)', 'Velocity Vector(X)', 'Velocity Vector(Y)', 'Velocity Vector(Z)']
        features_input_decoder = []
        feature_target = ['Position Vector(X)', 'Position Vector(Y)', 'Position Vector(Z)', 'Velocity Vector(X)', 'Velocity Vector(Y)', 'Velocity Vector(Z)']

        self.preprocessor_loaded_list = []
        X_train_list = []
        y_train_list = []
        X_val_list = []
        y_val_list = []

        for i in range(len(self.df_list)):
            preprocessor = Preprocessor(self.df_list[i])
            preprocessor.process(
                features_input_encoder,
                features_input_decoder,
                feature_target,
                input_seq_len,
                output_seq_len,
                model_type="tcn_tcn",
                time_col=self.df_list[i].index,
                split_ratio=split_ratio,
                split_date=None,
                temporal_encoding_modes=None,
                autoregressive=True
            )
            self.preprocessor_loaded_list.append(preprocessor)

            X_train, y_train, X_val, y_val = preprocessor.train_test_split(split_ratio)
            X_train_list.append(X_train)
            y_train_list.append(y_train)
            X_val_list.append(X_val)
            y_val_list.append(y_val)

        self.x_train = [np.concatenate([X_train_list[j][i] for j in range(len(X_train_list))], axis=0) for i in range(len(X_train_list[0]))]
        self.x_val = [np.concatenate([X_val_list[j][i] for j in range(len(X_val_list))], axis=0) for i in range(len(X_val_list[0]))]
        self.y_train = np.concatenate(y_train_list, axis=0)
        self.y_val = np.concatenate(y_val_list, axis=0)

    
    def train(self):
        
        self.model = TCN_TCN_Attention()
        self.model.build(
            num_layers_tcn = None,
            num_filters = 12,
            kernel_size = 9,
            dilation_base = 2,
            dropout_rate = 0.3,
            key_size = 6,
            value_size = 6,
            num_attention_heads = 1,
            neurons_output = [16],
            activation = "relu",
            kernel_initializer = "he_normal",
            batch_norm_tcn = False,
            layer_norm_tcn = True,
            autoregressive=True,
            padding_encoder='causal',
            padding_decoder='causal')

        self.model.compile(optimizer= tf.keras.optimizers.legacy.Adam(learning_rate=0.01,decay=1e-3),run_eagerly = True)

        cb_early_stopping = EarlyStopping(patience=7, restore_best_weights=True)
        self.model.fit(self.x_train,
                        self.y_train,
                        (self.x_val, self.y_val),
                        epochs=1,
                        batch_size=32,
                        callbacks = cb_early_stopping
                        )

        self.model.save_model(self.config_path)

    def parameter_check(self):
        self.model = TCN_TCN_Attention()
        self.model.parameter_search(self.x_train,
                                    self.y_train,
                                    self.x_val,
                                    self.y_val,
                                    batch_size = 32,
                                    results_path = "./search_final1",
                                    patience=3,
                                    loss = "mse",
                                    max_trials = 50,
                                    executions_per_trial = 1,
                                    num_filters = [12, 16],
                                    neurons_output = [16],
                                    kernel_size = [3,5, 9, 13],
                                    dilation_base = [2],
                                    dropout_rate = [0.3],
                                    key_value_size = [6, 8],
                                    num_attention_heads = [1, 2],
                                    activation = ["relu"],
                                    kernel_initializer = ["he_normal"],
                                    batch_norm_tcn = [False],
                                    layer_norm_tcn = [True],
                                    padding_encoder = ['same', 'causal'],
                                    padding_decoder = ['causal']
                                )

config_path = "C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\EncoderDecoder\\ModelConfigs\\Test"
input_file = "C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\SatelliteData\\TESTDATA.zip"

trainer = ModelTrainer(config_path,input_file)
trainer.combine_data()
trainer.test_train_dataset()
trainer.train()