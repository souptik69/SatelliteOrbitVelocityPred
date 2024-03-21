from tcn_sequence_models.data_processing.preprocessor import Preprocessor
from tcn_sequence_models.models import TCN_TCN_Attention
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
from tcn_sequence_models.utils.scaling import inverse_scale_sequences
import numpy as np

config_path = "C:\\Users\\ssen\\Documents\\COOPERANTS\\SatellitePred\\EncoderDecoder\\ModelWeights"
df = pd.read_csv('C:\\Users\\ssen\\Downloads\\Satellite\\SatelliteData\\Model\\TCN\\vanguard.csv')
time_col = 'epoch'
df[time_col]= pd.to_datetime(df[time_col])
# features_input_encoder = ['position_x_sine_wave','position_x_cosine_wave','position_y_sine_wave','position_y_cosine_wave','position_z_sine_wave','position_z_cosine_wave',
#                           'velocity_z_sine_wave','velocity_z_cosine_wave','Position Vector(X)', 'Position Vector(Y)','Position Vector(Z)','Velocity Vector(X)','Velocity Vector(Y)', 'Velocity Vector(Z)']

# features_input_decoder = ['Position Vector(X)', 'Position Vector(Y)','Position Vector(Z)','Velocity Vector(X)','Velocity Vector(Y)', 'Velocity Vector(Z)']

# feature_target = ['Position Vector(X)','Position Vector(Y)','Position Vector(Z)','Velocity Vector(X)','Velocity Vector(Y)', 'Velocity Vector(Z)']
split_ratio = 0.7
input_seq_len = 120
output_seq_len = 30
preprocessor_loaded = Preprocessor(df)
preprocessor_loaded.load_preprocessor_config(load_path=config_path)
preprocessor_loaded.process_from_config_inference()
# X_train, y_train, X_val, y_val = preprocessor_loaded.train_test_split(split_ratio=0.7)
model_loaded = TCN_TCN_Attention()
# model_loaded.load_model(config_path, preprocessor_loaded.X, is_training_data=False)
model_loaded.load_model(config_path, preprocessor_loaded.X,is_training_data=False)
# y_pred = model_loaded.predict(preprocessor_loaded.X)
# y_pred_unscaled = inverse_scale_sequences(y_pred, preprocessor_loaded.scaler_y)
y_val_unscaled = inverse_scale_sequences(np.expand_dims(preprocessor_loaded.y, axis=-1),
                                                        preprocessor_loaded.scaler_y)
# plt.plot(y_pred_unscaled[10000])
# plt.plot(y_val_unscaled[10000])
# plt.show()
# print(preprocessor_loaded.X)
# y_pred = model_loaded.eval(X_val,y_val)
# y_pred_unscaled = inverse_scale_sequences(y_pred, preprocessor_loaded.scaler_y)
# y_true_unscaled = inverse_scale_sequences(np.expand_dims(preprocessor_loaded.y, axis=-1),
#                                                         preprocessor_loaded.scaler_y)
# plt.plot(y_pred_unscaled[1000])
# # plt.plot(y_true_unscaled[1000])

# plt.show()

