import os
import zipfile
import pandas as pd
from tcn_sequence_models.data_processing.preprocessor import Preprocessor
from data_preprocessing.DataPrepClass import SatelliteDataProcessor
from data_preprocessing.PreprocessingClass import SatelliteDataPreProcess

class DataProcessor:
    def __init__(self, zip_file_path, save_folder):
        self.zip_file_path = zip_file_path
        self.save_folder = save_folder

    def process_data_from_zip(self):
        with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
            zip_file_names = zip_ref.namelist()
            for file_name in zip_file_names:
                if file_name.endswith('.csv'):
                    csv_file_name = os.path.splitext(os.path.basename(file_name))[0]
                    folder_path = os.path.join(self.save_folder, csv_file_name)
                    os.makedirs(folder_path, exist_ok=True)
                    csv_file_path = zip_ref.extract(file_name, folder_path)
                    self.process_and_save(csv_file_path, folder_path)

    def process_and_save(self, csv_file_path, save_folder):
        processor = SatelliteDataProcessor(csv_file_path)
        processor.load_data()
        processed_df = processor.process_data()
        
        preprocess = SatelliteDataPreProcess(processed_df)
        df = preprocess.preprocess_data()
        
        time_col = 'epoch'
        df[time_col] = pd.to_datetime(df[time_col])
        
        split_ratio = 0.7
        input_seq_len = 120
        output_seq_len = 30
        features_input_encoder = ['position_x_sine_wave', 'position_x_cosine_wave', 'position_y_sine_wave', 'position_y_cosine_wave', 'position_z_sine_wave', 'position_z_cosine_wave', 'velocity_x_sine_wave', 'velocity_x_cosine_wave', 'velocity_y_sine_wave', 'velocity_y_cosine_wave', 'velocity_z_sine_wave', 'velocity_z_cosine_wave', 'Position Vector(X)', 'Position Vector(Y)', 'Position Vector(Z)', 'Velocity Vector(X)', 'Velocity Vector(Y)', 'Velocity Vector(Z)']
        features_input_decoder = ['Position Vector(X)', 'Position Vector(Y)', 'Position Vector(Z)', 'Velocity Vector(X)', 'Velocity Vector(Y)', 'Velocity Vector(Z)']
        feature_target = ['Position Vector(X)', 'Position Vector(Y)', 'Position Vector(Z)', 'Velocity Vector(X)', 'Velocity Vector(Y)', 'Velocity Vector(Z)']
        
        preprocessor = Preprocessor(df)
        preprocessor.process(
            features_input_encoder,
            features_input_decoder,
            feature_target,
            input_seq_len,
            output_seq_len,
            model_type="tcn_tcn",
            time_col=time_col,
            split_ratio=split_ratio,
            split_date=None,
            temporal_encoding_modes=None,
            autoregressive=True
        )
        
        save_path = save_folder
        preprocessor.save_preprocessor_config(save_path)

# Example usage
zip_file_path = 'C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\SatelliteData\\T1.zip'
save_folder = 'C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\EncoderDecoder\\PreProcessorConfigs'
processor = DataProcessor(zip_file_path, save_folder)
processor.process_data_from_zip()
