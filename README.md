# SatelliteOrbitPrediction
This project utilizes a new encoder-decoder based architecture using temporal CNNs to predict the position and velocity of several satellites

## Environment

- python 3.10.8
- pandas
- numpy
- scipy
- tensorflow
- scikit-learn


## WorkFlow


- [SatelliteOrbitVelocityPred/SatellitePred/EncoderDecoder/src/data_preprocessing] -- Contains helper classes for preprocessesing the raw training and testing Satellite input files . Satellite input files have Kepler's parameters, which are processed to generate satellite position and velocity vectors by the DataPrepClass.py file. These vectors are then preprocessed using different feature engineering techniques in PreprocessingClass.py.
- [SatelliteOrbitVelocityPred/SatellitePred/EncoderDecoder/src/AttentionTCN.py] -- Contains all the functions to process the training data in machine inter-pretable format ,train the final model with processed data and to do a Bayesian Hyper-Parameter Check with processed training data. Uses helper functions for preprocessing and model trainingg.
- [src/tcn_sequence_models/utils] - Contains various helper functions which are used in preprocessing abd traing / validating different models
- [src/tcn_sequence_models/data_processing] - Contains the various training/testing data processing helper programs which are used to sequence the data using sliding window to generate sequences and to do necessary processing steps like normalization, scaling etc. Also has optional functions to apply seasonal temporal encoding, one-hot encoding etc. to data.
- [src/tcn_sequence_models/tf_models] - Contains diffrent model architecture folders. Each model folder uses a TCN based architecture and has their own encoder and decoder class, and a program which ties together these classes as a feed forward layer class. The tcn.py contains the Temporal Convolutional Network Block. We have used the TCN_TCN_Attention model for our experiments based on results.
- [src/tcn_sequence_models/models.py] - Ties together all the mdoels as different classes and also provides model training/testing and hyper-parameter checking functions.
- [src/Testing.py] - Contains different functions to test and plot the results.
- [SatelliteOrbitVelocityPred/SatellitePred/EncoderDecoder/PreProcessorConfigs] - Contains the preprocess configs for the different satellies .
- [SatelliteOrbitVelocityPred/SatellitePred/EncoderDecoder/ModelConfigs] - Contains saved models .
- [SatelliteOrbitVelocityPred/SatellitePred/Plots]- Contains different training and testing results plots for different satellites.
- [SatelliteOrbitVelocityPred/SatellitePred/SatellieData] - Contains raw satellite data files.
