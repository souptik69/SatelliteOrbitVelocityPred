# # from fastapi import FastAPI, UploadFile, File
# # from tcn_sequence_models.data_processing.preprocessor import Preprocessor
# # from tcn_sequence_models.models import TCN_TCN_Attention
# # import pandas as pd
# # import tensorflow as tf
# # import numpy as np
# # app = FastAPI()

# # config_path = "C:\\Users\\ssen\\Documents\\COOPERANTS\\SatellitePred\\EncoderDecoder\\RetrainModelWeights"
# # df = pd.read_csv('C:\\Users\\ssen\\Documents\\COOPERANTS\\SatellitePred\\SatelliteData\\explorer.csv')
# # time_col = 'epoch'
# # df[time_col]= pd.to_datetime(df[time_col])

# # split_ratio = 0.7
# # input_seq_len = 120
# # output_seq_len = 30

# # preprocessor_loaded = Preprocessor(df)
# # preprocessor_loaded.load_preprocessor_config(load_path=config_path)

# # preprocessor_loaded.process_from_config_inference()
# # # X_train, y_train, X_val, y_val = preprocessor_loaded.train_test_split(split_ratio)

# # # Load the pre-trained model
# # model_loaded = TCN_TCN_Attention()
# # model_loaded.load_model(config_path, preprocessor_loaded.X, is_training_data=False)


# # # Define the prediction endpoint
# # @app.post('/predict')
# # async def predict(file: UploadFile):
# #     # Save the uploaded file temporarily
# #     with open(file.filename, 'wb') as buffer:
# #         buffer.write(await file.read())

# #     # Load the saved file
# #     preprocessor_loaded = Preprocessor(file.filename)
# #     preprocessor_loaded.load_preprocessor_config(load_path=config_path)
# #     preprocessor_loaded.process_from_config_inference()
    
# #     # Make predictions using the loaded model
# #     y_pred = model_loaded.predict(preprocessor_loaded.X)
    
# #     # Process the predictions as per your requirements
# #     # ...
    
# #     # Return the predicted results
# #     return {'predictions': np.array(y_pred).tolist()}

# # # Run the FastAPI server
# # if __name__ == '__main__':
# #     import uvicorn
# #     uvicorn.run(app, host='0.0.0.0', port=8000)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import json
import uvicorn
from io import StringIO
import numpy as np
from Testing import ModelTester

app = FastAPI()


class PredictionRequest(BaseModel):
    test_dataset: bytes

@app.get("/favicon.ico")
async def get_favicon():
    raise HTTPException(status_code=404)

@app.get("/")
async def root():
    return "FastAPI Server is running"

@app.get("/upload")
async def upload_page():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload CSV Dataset</title>
    </head>
    <body>
        <h1>Upload JSON Dataset</h1>
        <form action="/predict" method="POST" enctype="multipart/form-data">
            <input type="file" name="test_dataset">
            <br><br>
            <input type="submit" value="Upload and Predict">
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/predict")
async def predict_and_plot(request: PredictionRequest):
    # Read the uploaded JSON test dataset file
    # test_data_json = await request.test_dataset.read()
    # test_data = json.loads(test_data_json)

    # # Convert JSON to DataFrame and save as CSV
    # test_df = pd.DataFrame(test_data)
    # test_csv = test_df.to_csv(index=False)
    # test_csv = await request.test_dataset.read()
    test_csv = request.test_dataset.decode()
    model_loader = ModelTester(config_path,config_path1, test_csv)

    model_loader.load_data()
    model_loader.load_model()

    model_loader.plot_API()

    return "Prediction and plotting completed."





if __name__ == "__main__":
    config_path = "C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\EncoderDecoder\\ModelConfigs\\FinalModel"
    config_path1 = "C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\EncoderDecoder\\PreProcessorConfigs\\IRIDIUM4"
    uvicorn.run(app, host="localhost", port=8000)


# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import HTMLResponse
# from pydantic import BaseModel
# import pandas as pd
# import json
# import uvicorn
# from io import StringIO
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from tcn_sequence_models.data_processing.preprocessor import Preprocessor
# from tcn_sequence_models.models import TCN_TCN_Attention
# from tcn_sequence_models.utils.scaling import inverse_scale_sequences
# from data_preprocessing.DataPrepClass import SatelliteDataProcessor
# from data_preprocessing.PreprocessingClass import SatelliteDataPreProcess

# app = FastAPI()

# class PredictionRequest(BaseModel):
#     test_dataset: UploadFile = File(...)

# @app.post("/predict", response_class=HTMLResponse)
# async def predict_and_plot(request: PredictionRequest):
#     # Read the uploaded JSON test dataset file
#     test_data_json = await request.test_dataset.read()
#     test_data = json.loads(test_data_json)

#     # Convert JSON to DataFrame and save as CSV
#     test_df = pd.DataFrame(test_data)
#     test_csv = test_df.to_csv(index=False)

#     # Perform preprocessing on the test dataset
#     processor = SatelliteDataProcessor(test_csv)
#     processor.load_data()
#     processed_df = processor.process_data()

#     preprocess = SatelliteDataPreProcess(processed_df)
#     df = preprocess.preprocess_data()

#     # Load the preprocessor
#     preprocessor = Preprocessor(df)
#     preprocessor.load_preprocessor_config(load_path=config_path1)
#     preprocessor.process_from_config_inference()

#     # Load the saved model
#     model = TCN_TCN_Attention()
#     model.load_model(config_path, preprocessor.X, is_training_data=False)

#     # Make predictions
#     y_pred = model.predict(preprocessor.X)
#     y_pred_unscaled = inverse_scale_sequences(y_pred, preprocessor.scaler_y)
#     y_true_unscaled = inverse_scale_sequences(preprocessor.y, preprocessor.scaler_y)

#     num_outputs = y_pred_unscaled.shape[2]  # Number of output values
#     output_labels = ['Position Vector(X)', 'Position Vector(Y)', 'Position Vector(Z)', 'Velocity Vector(X)', 'Velocity Vector(Y)', 'Velocity Vector(Z)']

#     # Create plots
#     fig = plt.figure(figsize=(10, 6 * num_outputs))
#     gs = gridspec.GridSpec(num_outputs, 1, figure=fig)

#     for i in range(num_outputs):
#         plt.subplot(gs[i])
#         plt.plot(y_pred_unscaled[9000, :, i], label='Predicted')
#         plt.plot(y_true_unscaled[9000, :, i], label='True')
#         plt.xlabel('Time')
#         plt.ylabel(output_labels[i])
#         plt.legend()

#     # Save the plots to a file
#     plt.savefig('plots.png')
#     plt.close(fig)

#     # Return the HTML response with the plots
#     return '''
#     <h1>Predictions and Plots</h1>
#     <img src="plots.png" alt="Plots">
#     '''


# @app.get("/")
# async def root():
#     return {"message": "Welcome to the prediction and plotting API!"}

# if __name__ == "__main__":
#     config_path = "C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\EncoderDecoder\\ModelConfigs\\FinalModel"
#     config_path1 = "C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\EncoderDecoder\\PreProcessorConfigs\\IRIDIUM4"
#     uvicorn.run(app, host="localhost", port=8000)


# import os
# import json
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from io import BytesIO
# from fastapi import FastAPI, Form, File, UploadFile, Request
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# import uvicorn
# from Testing import ModelTester
# # from tcn_sequence_models.data_processing.preprocessor import Preprocessor
# # from tcn_sequence_models.models import TCN_TCN_Attention
# from tcn_sequence_models.utils.scaling import inverse_scale_sequences
# # from data_preprocessing.DataPrepClass import SatelliteDataProcessor
# # from data_preprocessing.PreprocessingClass import SatelliteDataPreProcess


# app = FastAPI()
# app.mount("/static", StaticFiles(directory="C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\EncoderDecoder\\src\\static"), name="static")
# templates = Jinja2Templates(directory="C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\EncoderDecoder\\src\\templates")


# def process_dataset(file):
#     # Convert JSON to CSV and perform necessary preprocessing
#     data = json.load(file)
#     df = pd.DataFrame(data)
#     test_csv = df.to_csv(index=False)

#     return test_csv


# def plot_results(y_pred_unscaled, y_true_unscaled):
#     num_outputs = y_pred_unscaled.shape[2]  # Number of output values
#     output_labels = ['Position Vector(X)', 'Position Vector(Y)', 'Position Vector(Z)', 'Velocity Vector(X)', 'Velocity Vector(Y)', 'Velocity Vector(Z)']

#     plt.figure(figsize=(10, 6))  # Adjust figure size as needed

#     for i in range(num_outputs):
#         plt.subplot(num_outputs, 1, i + 1)
#         plt.plot(y_pred_unscaled[9000, :, i], label='Predicted')
#         plt.plot(y_true_unscaled[9000, :, i], label='True')
#         plt.xlabel('Time')
#         plt.ylabel(output_labels[i])
#         plt.legend()

#     plt.show()


# @app.get("/")
# def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})


# @app.post("/upload")
# async def upload(dataset: UploadFile = File(...)):
#     if dataset:
#         file_content = await dataset.read()
#         file = BytesIO(file_content)
#         df = process_dataset(file)

#         # Load model and perform predictions
#         config_path = "C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\EncoderDecoder\\ModelConfigs\\FinalModel"
#         config_path1 = "C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\EncoderDecoder\\PreProcessorConfigs\\IRIDIUM4"
#         model_loader = ModelTester(config_path, config_path1, df)
#         model_loader.load_data()
#         model_loader.load_model()

#         y_pred = model_loader.model.predict(model_loader.preprocessor.X)
#         y_pred_unscaled = inverse_scale_sequences(y_pred, model_loader.preprocessor.scaler_y)
#         y_true_unscaled = inverse_scale_sequences(model_loader.preprocessor.y, model_loader.preprocessor.scaler_y)

#         # Plot the results
#         plot_results(y_pred_unscaled, y_true_unscaled)

#         return {"message": "Dataset uploaded, processed, and plotted successfully"}

#     return {"message": "Failed to process the dataset"}


# if __name__ == "__main__":
#     # Create the 'static' directory if it doesn't exist
#     os.makedirs("static", exist_ok=True)
#     # Run the FastAPI server
#     uvicorn.run(app, host="localhost", port=8000)
