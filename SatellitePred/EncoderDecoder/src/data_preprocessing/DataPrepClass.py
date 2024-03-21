import numpy as np
import pandas as pd
import scipy.constants as sp
import json

class SatelliteDataProcessor:
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = None

    def load_data(self):
        # with open(self.input_file, 'r') as file:
        #     data = json.load(file)
        # self.df = pd.read_json(self.input_file)

        self.df = pd.read_csv(self.input_file)
        # self.df = pd.DataFrame.from_dict(data, orient='columns')


        self.df['epoch'] = pd.to_datetime(self.df['EPOCH'])

    def process_data(self):
        a = 7155  # Semi-Major axis
        G = sp.gravitational_constant  # Gravitational Constant

        M = self.df[['EPOCH', 'ECCENTRICITY', 'INCLINATION', 'RA_OF_ASC_NODE', 'ARG_OF_PERICENTER',
                     'MEAN_ANOMALY', 'MEAN_MOTION_DOT']].values
        processed_data = []

        for m in M:
            epoch = m[0]
            e = m[1]  # eccentricity
            i = m[2]  # inclination
            r = m[3]  # rate of ascending node
            M_val = m[5]  # mean_anomaly
            Mu = G * M_val  # standard gravitational parameter
            O = m[4]  # argument of perigee
            n = m[6]  # number of revs per day
            i1 = (np.tan(e / 2)) / (np.sqrt((1 - e) / (1 + e)))
            Theta = 2 * (np.arctan(i1))  # true anomaly
            h = np.sqrt(Mu * a * (1 - np.square(e)))  # angular momentum
            x = [np.cos(Theta), np.sin(Theta), 0]
            x1 = np.array(x)
            v = [-np.sin(Theta), e + np.cos(Theta), 0]
            v1 = np.array(v)
            F = ((np.square(h) / Mu) * (1 / (1 + (e * np.cos(Theta)))))
            k = [[(np.cos(O) * np.cos(r)) - (np.sin(O) * np.sin(r) * np.cos(i)),
                  ((-1) * np.sin(O) * np.cos(r)) - (np.cos(O) * np.sin(r) * np.cos(i)), np.sin(r) * np.sin(i)],
                 [(np.cos(O) * np.sin(r)) + (np.sin(O) * np.cos(r) * np.cos(i)),
                  ((-1) * np.sin(O) * np.sin(r)) + (np.cos(O) * np.cos(r) * np.cos(i)), (-1) * np.cos(r) * np.sin(i)],
                 [np.sin(O) * np.sin(i), np.cos(O) * np.sin(i), np.cos(i)]]
            k1 = np.array(k)
            L1 = F * x1
            r1 = np.dot(L1, k1)  # position vector
            F1 = Mu / h
            L = F1 * v1
            vel = np.dot(L, k1)  # velocity vector

            processed_data.append({
                'epoch': epoch,
                'Position Vector(X)': r1[0],
                'Position Vector(Y)': r1[1],
                'Position Vector(Z)': r1[2],
                'Velocity Vector(X)': vel[0],
                'Velocity Vector(Y)': vel[1],
                'Velocity Vector(Z)': vel[2]
            })

        processed_df = pd.DataFrame(processed_data)
        return processed_df

# # Test the code
# input_file = 'C:\\Users\\ssen\\Documents\\COOPERANTS\\satelliteprediction-playground\\SatellitePred\\SatelliteData\\IRIDIUM4.csv'
# processor = SatelliteDataProcessor(input_file)
# processor.load_data()
# processed_df = processor.process_data()

# # Use the processed DataFrame as desired
# print(processed_df.head())
# print(processed_df.index)
