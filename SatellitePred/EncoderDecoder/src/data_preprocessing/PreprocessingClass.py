import numpy as np
import pandas as pd
import tensorflow as tf
# from DataPrepClass import SatelliteDataProcessor

class SatelliteDataPreProcess:
    def __init__(self, input_data_frame):
        self.df = input_data_frame
        self.df['epoch'] = pd.to_datetime(self.df['epoch'])
        self.df.set_index('epoch', inplace=True)


    def preprocess_data(self):
        df_resampled = self.df.resample('D').mean()
        df_resampled = df_resampled.interpolate(method='time')
        df_resampled['epoch'] = df_resampled.index
        date_time = pd.to_datetime(df_resampled['epoch'])
        timestamp_s = date_time.map(pd.Timestamp.timestamp)
        
        fft = tf.signal.rfft(df_resampled['Position Vector(X)'])
        f_per_dataset = np.arange(0, len(fft))
        n_samples_d = len(df_resampled['Position Vector(X)'])
        days_per_year = 365.2524
        years_per_dataset = n_samples_d / days_per_year
        f_per_year = f_per_dataset / years_per_dataset
        f_per_month = f_per_year / 12  # divide by 12 to get per month cycle
        max_index = np.argmax(np.abs(fft))
        f_peak_posx = f_per_month[max_index]

        fft = tf.signal.rfft(df_resampled['Position Vector(Y)'])
        f_per_dataset = np.arange(0, len(fft))
        n_samples_d = len(df_resampled['Position Vector(Y)'])
        years_per_dataset = n_samples_d / days_per_year
        f_per_year = f_per_dataset / years_per_dataset
        f_per_month = f_per_year / 12  # divide by 12 to get per month cycle
        max_index = np.argmax(np.abs(fft))
        f_peak_posy = f_per_month[max_index]

        fft = tf.signal.rfft(df_resampled['Position Vector(Z)'])
        f_per_dataset = np.arange(0, len(fft))
        n_samples_d = len(df_resampled['Position Vector(Z)'])
        years_per_dataset = n_samples_d / days_per_year
        f_per_year = f_per_dataset / years_per_dataset
        f_per_month = f_per_year / 12  # divide by 12 to get per month cycle
        max_index = np.argmax(np.abs(fft))
        f_peak_posz = f_per_month[max_index]

        fft = tf.signal.rfft(df_resampled['Velocity Vector(X)'])
        f_per_dataset = np.arange(0, len(fft))
        n_samples_d = len(df_resampled['Velocity Vector(X)'])
        years_per_dataset = n_samples_d / days_per_year
        f_per_year = f_per_dataset / years_per_dataset
        f_per_month = f_per_year / 12  # divide by 12 to get per month cycle
        max_index = np.argmax(np.abs(fft))
        f_peak_velx = f_per_month[max_index]

        fft = tf.signal.rfft(df_resampled['Velocity Vector(Y)'])
        f_per_dataset = np.arange(0, len(fft))
        n_samples_d = len(df_resampled['Velocity Vector(Y)'])
        years_per_dataset = n_samples_d / days_per_year
        f_per_year = f_per_dataset / years_per_dataset
        f_per_month = f_per_year / 12  # divide by 12 to get per month cycle
        max_index = np.argmax(np.abs(fft))
        f_peak_vely = f_per_month[max_index]

        fft = tf.signal.rfft(df_resampled['Velocity Vector(Z)'])
        f_per_dataset = np.arange(0, len(fft))
        n_samples_d = len(df_resampled['Velocity Vector(Z)'])
        years_per_dataset = n_samples_d / days_per_year
        f_per_year = f_per_dataset / years_per_dataset
        f_per_month = f_per_year / 12  # divide by 12 to get per month cycle
        max_index = np.argmax(np.abs(fft))
        f_peak_velz = f_per_month[max_index]

        freq_per_sec = f_peak_posx / (30.44 * 24 * 60 * 60)
        df_resampled['position_x_sine_wave'] = np.sin(2 * np.pi * freq_per_sec * timestamp_s)
        df_resampled['position_x_cosine_wave'] = np.cos(2 * np.pi * freq_per_sec * timestamp_s)

        freq_per_sec1 = f_peak_posy / (30.44 * 24 * 60 * 60)
        df_resampled['position_y_sine_wave'] = np.sin(2 * np.pi * freq_per_sec1 * timestamp_s)
        df_resampled['position_y_cosine_wave'] = np.cos(2 * np.pi * freq_per_sec1 * timestamp_s)

        freq_per_sec2 = f_peak_posz / (30.44 * 24 * 60 * 60)
        df_resampled['position_z_sine_wave'] = np.sin(2 * np.pi * freq_per_sec2 * timestamp_s)
        df_resampled['position_z_cosine_wave'] = np.cos(2 * np.pi * freq_per_sec2 * timestamp_s)

        freq_per_sec3 = f_peak_velx / (30.44 * 24 * 60 * 60)
        df_resampled['velocity_x_sine_wave'] = np.sin(2 * np.pi * freq_per_sec3 * timestamp_s)
        df_resampled['velocity_x_cosine_wave'] = np.cos(2 * np.pi * freq_per_sec3 * timestamp_s)

        freq_per_sec4 = f_peak_vely / (30.44 * 24 * 60 * 60)
        df_resampled['velocity_y_sine_wave'] = np.sin(2 * np.pi * freq_per_sec4 * timestamp_s)
        df_resampled['velocity_y_cosine_wave'] = np.cos(2 * np.pi * freq_per_sec4 * timestamp_s)

        freq_per_sec5 = f_peak_velz / (30.44 * 24 * 60 * 60)
        df_resampled['velocity_z_sine_wave'] = np.sin(2 * np.pi * freq_per_sec5 * timestamp_s)
        df_resampled['velocity_z_cosine_wave'] = np.cos(2 * np.pi * freq_per_sec5 * timestamp_s)

        return df_resampled
    

# Test the code
# input_file = 'C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\SatelliteData\\Vanguard3.csv'
# processor = SatelliteDataProcessor(input_file)
# processor.load_data()
# processed_df = processor.process_data()

# preprocess = SatelliteDataPreProcess(processed_df)
# final_df = preprocess.preprocess_data()

# print(final_df.index)









            
