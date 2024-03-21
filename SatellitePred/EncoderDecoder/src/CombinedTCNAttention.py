

import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tcn_sequence_models.data_processing.preprocessor import Preprocessor
from tcn_sequence_models.models import TCN_TCN_Attention
from data_preprocessing.DataPrepClass import SatelliteDataProcessor
from data_preprocessing.PreprocessingClass import SatelliteDataPreProcess

class ModelTrainer:
    def __init__(self, config_path, input_file1, input_file2, config_path1, config_path2):
        self.config_path = config_path
        self.input_file1 = input_file1
        self.input_file2 = input_file2
        self.config_path1 = config_path1
        self.config_path2 = config_path2
        self.processor = None
        self.processor1 = None
        self.preprocess = None
        self.preprocess1 = None
        self.preprocessor_loaded = None
        self.preprocessor_loaded1 = None
        self.model = None

    def load_data(self, input_file):
        self.processor = SatelliteDataProcessor(input_file)
        self.processor.load_data()
        self.processed_df = self.processor.process_data()

    def preprocess_data(self):
        self.preprocess = SatelliteDataPreProcess(self.processed_df)
        self.df = self.preprocess.preprocess_data()
    
    def combine_data(self):
        self.processor1 = SatelliteDataProcessor(self.input_file1)
        self.processor1.load_data()
        processed_df1 = self.processor1.process_data()

        self.preprocess1 = SatelliteDataPreProcess(processed_df1)
        self.df1 = self.preprocess1.preprocess_data()

        
        self.processor2 = SatelliteDataProcessor(self.input_file2)
        self.processor2.load_data()
        processed_df2 = self.processor2.process_data()

        self.preprocess2 = SatelliteDataPreProcess(processed_df2)
        self.df2 = self.preprocess2.preprocess_data()

    def test_train_dataset(self):

        split_ratio = 0.7
        input_seq_len = 120
        output_seq_len = 30
        features_input_encoder = ['position_x_sine_wave','position_x_cosine_wave','position_y_sine_wave','position_y_cosine_wave','position_z_sine_wave','position_z_cosine_wave','velocity_x_sine_wave','velocity_x_cosine_wave','velocity_y_sine_wave','velocity_y_cosine_wave','velocity_z_sine_wave','velocity_z_cosine_wave','Position Vector(X)', 'Position Vector(Y)','Position Vector(Z)','Velocity Vector(X)','Velocity Vector(Y)', 'Velocity Vector(Z)']
        features_input_decoder = ['Position Vector(X)', 'Position Vector(Y)','Position Vector(Z)','Velocity Vector(X)','Velocity Vector(Y)', 'Velocity Vector(Z)']
        feature_target = ['Position Vector(X)','Position Vector(Y)','Position Vector(Z)','Velocity Vector(X)','Velocity Vector(Y)', 'Velocity Vector(Z)']
        self.preprocessor_loaded = Preprocessor(self.df1)

        self.preprocessor_loaded.process(
            features_input_encoder,
            features_input_decoder,
            feature_target,
            input_seq_len,
            output_seq_len,
            model_type="tcn_tcn",
            time_col=self.df1.index,
            split_ratio = split_ratio,
            split_date = None,
            temporal_encoding_modes=None,
            autoregressive=True,
            )

        X_train1, y_train1, X_val1, y_val1 = self.preprocessor_loaded.train_test_split(split_ratio)

        self.preprocessor_loaded1 = Preprocessor(self.df2)

        self.preprocessor_loaded1.process(
            features_input_encoder,
            features_input_decoder,
            feature_target,
            input_seq_len,
            output_seq_len,
            model_type="tcn_tcn",
            time_col=self.df2.index,
            split_ratio = split_ratio,
            split_date = None,
            temporal_encoding_modes=None,
            autoregressive=True,
            )

        X_train2, y_train2, X_val2, y_val2 = self.preprocessor_loaded1.train_test_split(split_ratio)
       
        self.x_train = [np.concatenate((X_train1[i], X_train2[i]), axis=0)
                        for i in range(len(X_train1))]
        self.x_val =  [np.concatenate((X_val1[i], X_val2[i]), axis=0)
                        for i in range(len(X_val1))]
                        
        # self.x_train = X_train1 + X_train2
        # self.x_val = np.concatenate((X_val1,X_val2), axis=0)
        # self.x_val = X_val1 + X_val2
        self.y_train = np.concatenate((y_train1,y_train2), axis=0)
        self.y_val = np.concatenate((y_val1,y_val2), axis=0)

    def train(self):
        
        self.model = TCN_TCN_Attention()
        self.model.build(
            num_layers_tcn = None,
            num_filters = 12,
            kernel_size = 3,
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
            padding_encoder='same',
            padding_decoder='causal')

        self.model.compile(optimizer= tf.keras.optimizers.legacy.Adam(learning_rate=0.005,decay=1e-3))

        cb_early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
        self.model.fit(self.x_train,
                        self.y_train,
                        (self.x_val, self.y_val),
                        epochs=50,
                        batch_size=16,
                        callbacks = cb_early_stopping
                        )

        # self.preprocessor_loaded.save_preprocessor_config(save_path=self.config_path2)
        # self.preprocessor_loaded1.save_preprocessor_config(save_path=self.config_path1)
        self.model.save_model(self.config_path)

config_path = "C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\EncoderDecoder\\CombineRetraining"
input_file1 = "C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\SatelliteData\\Vanguard3.csv"
input_file2 = "C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\SatelliteData\\Explorer7.csv"
config_path1 = "C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\EncoderDecoder\\Explorer_preprocessor"
config_path2 = "C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\EncoderDecoder\\Vanguard_preprocessor"

trainer = ModelTrainer(config_path, input_file1, input_file2,config_path1,config_path2)
trainer.combine_data()
trainer.test_train_dataset()
trainer.train()