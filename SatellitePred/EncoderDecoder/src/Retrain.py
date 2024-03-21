import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tcn_sequence_models.data_processing.preprocessor import Preprocessor
from tcn_sequence_models.models import TCN_TCN_attention_transfer
from data_preprocessing.DataPrepClass import SatelliteDataProcessor
from data_preprocessing.PreprocessingClass import SatelliteDataPreProcess

class SatelliteModelTrainer:
    def __init__(self, config_path, input_file1, input_file2, config_path2,config_path3):
        self.config_path = config_path
        self.input_file1 = input_file1
        self.input_file2 = input_file2
        self.config_path2 = config_path2
        self.config_path3 = config_path3
        self.processor = None
        self.processor1 = None
        self.processor2 = None
        self.preprocess = None
        self.preprocess1 = None
        self.preprocess2 = None
        self.preprocessor_loaded = None
        self.preprocessor_loaded1 = None
        self.model_loaded = None
        self.model_loaded1 = None

    def load_data(self, input_file):
        self.processor = SatelliteDataProcessor(input_file)
        self.processor.load_data()
        self.processed_df = self.processor.process_data()

    def preprocess_data(self):
        self.preprocess = SatelliteDataPreProcess(self.processed_df)
        self.df = self.preprocess.preprocess_data()

    def train_model(self):
    

        split_ratio = 0.7
        input_seq_len = 120
        output_seq_len = 30

        self.preprocessor_loaded = Preprocessor(self.df)
        self.preprocessor_loaded.load_preprocessor_config(load_path=self.config_path)

        self.preprocessor_loaded.process_from_config_training(input_seq_len, output_seq_len)
        X_train, y_train, X_val, y_val = self.preprocessor_loaded.train_test_split(split_ratio)

        self.model_loaded = TCN_TCN_attention_transfer()
        self.model_loaded.load_model_Transfer(self.config_path, self.preprocessor_loaded.X[:3], is_training_data=True)
        self.model_loaded.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.02, decay=1e-3))
        self.model_loaded.summary()
        cb_early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
        self.model_loaded.fit(X_train,
                              y_train,
                              (X_val, y_val),
                              epochs=20,
                              batch_size=16,
                              callbacks=cb_early_stopping
                              )

        self.preprocessor_loaded.save_preprocessor_config(save_path=self.config_path2)
        self.model_loaded.save_model(self.config_path2)

    def preprocess_and_combine(self):
        
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

        
        self.combined_df = pd.concat([self.df1, self.df2], ignore_index=True)


    def train_combined_dataset(self):
        split_ratio = 0.7
        input_seq_len = 120
        output_seq_len = 30

        self.preprocessor_loaded1 = Preprocessor(self.combined_df)
        self.preprocessor_loaded1.load_preprocessor_config(load_path=self.config_path2)
        self.preprocessor_loaded1.process_from_config_training(input_seq_len, output_seq_len)
        X_train, y_train, X_val, y_val = self.preprocessor_loaded1.train_test_split(split_ratio)

        self.model_loaded1 = TCN_TCN_attention_transfer()
        self.model_loaded1.load_model_Transfer(self.config_path2, self.preprocessor_loaded1.X[:3], is_training_data=True)
        self.model_loaded1.trainable()
        self.model_loaded1.summary()

        self.model_loaded1.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.004, decay=1e-3))

        cb_early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
        self.model_loaded1.fit(X_train,
                              y_train,
                              (X_val, y_val),
                              epochs=50,
                              batch_size=16,
                              callbacks=cb_early_stopping
                              )
        self.preprocessor_loaded1.save_preprocessor_config(save_path=self.config_path3)
        self.model_loaded1.save_model(self.config_path3)

config_path = "C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\EncoderDecoder\\RetrainModelWeights"
config_path2 = "C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\EncoderDecoder\\Retraining"
input_file1 = "C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\SatelliteData\\Vanguard3.csv"
input_file2 = "C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\SatelliteData\\Explorer7.csv"
config_path3 = "C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\EncoderDecoder\\CombineRetraining"
trainer = SatelliteModelTrainer(config_path, input_file1, input_file2, config_path2, config_path3)

trainer.load_data(input_file1)
trainer.preprocess_data()

trainer.train_model()

trainer.preprocess_and_combine()
trainer.train_combined_dataset()