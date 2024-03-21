from tcn_sequence_models.data_processing.preprocessor import Preprocessor
from tcn_sequence_models.models import TCN_TCN_Attention
from tcn_sequence_models.utils.scaling import inverse_scale_sequences
from data_preprocessing.DataPrepClass import SatelliteDataProcessor
from data_preprocessing.PreprocessingClass import SatelliteDataPreProcess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class ModelTester:
    def __init__(self, config_path,config_path1, input_file):
        self.config_path = config_path
        self.config_path1 = config_path1
        self.input_file = input_file
        self.preprocessor = None
        self.model = None

    def load_data(self):
        processor = SatelliteDataProcessor(self.input_file)
        processor.load_data()
        processed_df = processor.process_data()

        #Remove this process and check
        preprocess = SatelliteDataPreProcess(processed_df)
        df = preprocess.preprocess_data()
       
        self.preprocessor = Preprocessor(df)
      
        self.preprocessor.load_preprocessor_config(load_path=self.config_path1)
        self.preprocessor.process_from_config_inference()

    def load_model(self):
        self.model = TCN_TCN_Attention()
        self.model.load_model(self.config_path, self.preprocessor.X, is_training_data=False)

    def evaluate(self):
        # mse = self.model.eval(self.preprocessor.X, self.preprocessor.y)
        mse = self.model.predict(self.preprocessor.X)
        return mse
    
    def plot(self):
        y_pred = self.model.predict(self.preprocessor.X)
        y_pred_unscaled = inverse_scale_sequences(y_pred, self.preprocessor.scaler_y)
        y_true_unscaled = inverse_scale_sequences(self.preprocessor.y, self.preprocessor.scaler_y)
        num_outputs = y_pred_unscaled.shape[2]  # Number of output values
        output_labels = ['Position Vector(X)', 'Position Vector(Y)', 'Position Vector(Z)', 'Velocity Vector(X)', 'Velocity Vector(Y)', 'Velocity Vector(Z)']
        plt.figure(figsize=(10, 6))  # Adjust figure size as needed

        for i in range(num_outputs):
               plt.subplot(num_outputs, 1, i + 1)
               plt.plot(y_pred_unscaled[9000, :, i], label='Predicted')
               plt.plot(y_true_unscaled[9000, :, i], label='True')
               plt.xlabel('Time')
               plt.ylabel(output_labels[i])
               plt.legend()

        plt.show()
    
    def plot_API(self):
        y_pred = self.model.predict(self.preprocessor.X)
        y_pred_unscaled = inverse_scale_sequences(y_pred, self.preprocessor.scaler_y)
        y_true_unscaled = inverse_scale_sequences(self.preprocessor.y, self.preprocessor.scaler_y)
        num_outputs = y_pred_unscaled.shape[2]
        output_labels = ['Position Vector(X)', 'Position Vector(Y)', 'Position Vector(Z)', 'Velocity Vector(X)', 'Velocity Vector(Y)', 'Velocity Vector(Z)']
        fig = plt.figure(figsize=(10, 6 * num_outputs))
        gs = gridspec.GridSpec(num_outputs, 1, figure=fig)

        for i in range(num_outputs):
            plt.subplot(gs[i])
            plt.plot(y_pred_unscaled[5000, :, i], label='Predicted')
            plt.plot(y_true_unscaled[5000, :, i], label='True')
            plt.xlabel('Time')
            plt.ylabel(output_labels[i])
            plt.legend()

        plt.savefig('plots.png')
        plt.close(fig)

         # Return the HTML response with the plots
        return '''
        <h1>Predictions and Plots</h1>
        <img src="plots.png" alt="Plots">
        '''


    def plot_scatter(self):
        y_pred = self.model.predict(self.preprocessor.X)
        y_pred_unscaled = inverse_scale_sequences(y_pred, self.preprocessor.scaler_y)
        y_true_unscaled = inverse_scale_sequences(self.preprocessor.y, self.preprocessor.scaler_y)            
        num_outputs = y_pred_unscaled.shape[2]  # Number of output values

        plt.figure(figsize=(10, 6))  # Adjust figure size as needed

        for i in range(num_outputs):
            plt.subplot(num_outputs, 1, i + 1)
            plt.scatter(y_true_unscaled[9000, :, i], y_pred_unscaled[9000, :, i])
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Scatter Plot - Output {i+1}')

        plt.tight_layout()
        plt.show()
    
    
    def plot_density(self):
        y_pred = self.model.predict(self.preprocessor.X)
        y_pred_unscaled = inverse_scale_sequences(y_pred, self.preprocessor.scaler_y)
        y_true_unscaled = inverse_scale_sequences(self.preprocessor.y, self.preprocessor.scaler_y)
        num_outputs = y_pred_unscaled.shape[2]  # Number of output values

        plt.figure(figsize=(10, 6))  # Adjust figure size as needed

        for i in range(num_outputs):
            plt.subplot(num_outputs, 1, i + 1)
            plt.hist(y_true_unscaled[9000, :, i], bins=30, density=True, alpha=0.5, label='True Values')
            plt.hist(y_pred_unscaled[9000, :, i], bins=30, density=True, alpha=0.5, label='Predicted Values')
            plt.xlabel('Values')
            plt.ylabel('Density')
            plt.title(f'Density Plot - Output {i+1}')
            plt.legend()

        plt.tight_layout()
        plt.show()
    
    def plot_error_histogram(self):
        y_pred = self.model.predict(self.preprocessor.X)
        y_pred_unscaled = inverse_scale_sequences(y_pred, self.preprocessor.scaler_y)
        y_true_unscaled = inverse_scale_sequences(self.preprocessor.y, self.preprocessor.scaler_y)
        num_outputs = y_pred_unscaled.shape[2]  # Number of output values
        
        plt.figure(figsize=(10, 6))  # Adjust figure size as needed

        for i in range(num_outputs):
            errors = y_pred_unscaled[9000, :, i] - y_true_unscaled[9000, :, i]
            plt.subplot(num_outputs, 1, i + 1)
            plt.hist(errors, bins=30, density=True, alpha=0.5)
            plt.xlabel('Errors')
            plt.ylabel('Density')
            plt.title(f'Error Histogram - Output {i+1}')

        plt.tight_layout()
        plt.show()

    def plot_all(self):
        y_pred = self.model.predict(self.preprocessor.X)
        y_pred_unscaled = inverse_scale_sequences(y_pred, self.preprocessor.scaler_y)
        y_true_unscaled = inverse_scale_sequences(self.preprocessor.y, self.preprocessor.scaler_y)
        num_outputs = y_pred_unscaled.shape[2]  # Number of output values

        fig = plt.figure(figsize=(10, 8))  # Adjust figure size as needed
        gs = gridspec.GridSpec(num_outputs + 2, 1, figure=fig)

        # Scatter plot
        ax_scatter = fig.add_subplot(gs[0])
        for i in range(num_outputs):
            ax_scatter.scatter(y_true_unscaled[9000, :, i], y_pred_unscaled[9000, :, i])
            ax_scatter.set_xlabel('True Values',fontsize=8)
            ax_scatter.set_ylabel('Predicted Values',fontsize=8)
            ax_scatter.set_title('Scatter Plot',fontsize=10)

        # Density plot
        ax_density = fig.add_subplot(gs[1])
        for i in range(num_outputs):
            ax_density.hist(y_true_unscaled[9000, :, i], bins=30, density=True, alpha=0.5, label='True Values')
            ax_density.hist(y_pred_unscaled[9000, :, i], bins=30, density=True, alpha=0.5, label='Predicted Values')
            ax_density.set_xlabel('Values',fontsize=8)
            ax_density.set_ylabel('Density',fontsize=8)
            ax_density.set_title('Density Plot',fontsize=10)
            ax_density.legend()

        # Error histogram
        ax_error = fig.add_subplot(gs[2])
        for i in range(num_outputs):
            errors = y_pred_unscaled[9000, :, i] - y_true_unscaled[9000, :, i]
            ax_error.hist(errors, bins=30, density=True, alpha=0.5)
            ax_error.set_xlabel('Errors',fontsize=8)
            ax_error.set_ylabel('Density',fontsize=8)
            ax_error.set_title('Error Histogram',fontsize=10)

        # # Compute and display MSE
        # mse = self.evaluate()
        # plt.annotate(f"MSE: {mse:.4f}", xy=(0.5, 0), xycoords='figure fraction', ha='center',
        #             fontsize=12, color='red')

        # plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)
        plt.show()


# Usage example
config_path = "C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\EncoderDecoder\\ModelConfigs\\FinalModel4"
input_file = "C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\SatelliteData\\TESTDATA.csv"
config_path1 = "C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\EncoderDecoder\\PreProcessorConfigs\\TESTDATA"

model_loader = ModelTester(config_path,config_path1, input_file)
model_loader.load_data()
model_loader.load_model()
# # model_loader.plot()
mse = model_loader.evaluate()
print(mse)
# model_loader.plot()
# model_loader.plot_all()
# model_loader.plot_scatter()
# model_loader.plot_density()
# model_loader.plot_error_histogram()


