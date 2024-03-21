import pandas as pd
df = pd.read_json (r'C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\SatelliteData\\T1.json')
df.to_csv (r'C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\SatelliteData\\T1.csv', index = None)
# print(df.head())