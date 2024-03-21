# Importing Liraries

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
import scipy.constants as sp

# Uploading Dataset


df = pd.read_csv('C:\\Users\\ssen\\Downloads\\Satellite\\SatelliteData\\Satellite.csv')
df['epoch'] = pd.to_datetime(df['EPOCH'])

# Collecting hyperparameters


X = df[['EPOCH', 'ECCENTRICITY', 'INCLINATION', 'RA_OF_ASC_NODE', 'ARG_OF_PERICENTER', 'MEAN_ANOMALY',
        'MEAN_MOTION_DOT']].copy()

# Calculating position vector and velocity vector


a = 7155  # Semi-Major axis
G = sp.gravitational_constant  # Gravitational Constant

M = X.to_numpy()
# df1 = pd.DataFrame()
df2 = pd.DataFrame()
# J3 = []
# J4 = np.array(J3, dtype=object)
# J6 = np.array(J5, dtype=object)
for m in M:
    m1 = np.array(m)
    epoch = m1[0]
    e = m1[1]  # eccentricity
    i = m1[2]  # inclination
    r = m1[3]  # rate of ascending node
    M = m1[5]  # mean_anomaly
    Mu = G * M  # standard gravitational parameter
    O = m1[4]  # argument of perigree
    n = m1[6]  # numer of revs per day
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
    r = np.dot(L1, k1)  # position vector
    # r1 = normalize(r.reshape(-1,1))
    # r1 = np.array(r)
    # J4 = np.append(J4,r1)
    F1 = Mu / h
    L = F1 * v1
    vel = np.dot(L, k1)  # velocity vector
    # vel1 = normalize(vel.reshape(-1,1))
    # J7 = [[epoch], [r], [vel]]
    # df2 = pd.DataFrame({'epoch' : epoch, 'Position Vector': r, 'Velocity Vector': vel})
    # df1 = pd.concat([df1,df2])
    # df1 = df1.append({'epoch': epoch, 'Position Vector': r, 'Velocity Vector': vel}, ignore_index=True)
    df2 = df2.append(
        {'epoch': epoch, 'Position Vector(X)': r[0], 'Position Vector(Y)': r[1], 'Position Vector(Z)': r[2],
         'Velocity Vector(X)': vel[0], 'Velocity Vector(Y)': vel[1], 'Velocity Vector(Z)': vel[2]}, ignore_index=True)

# Saving to csv File
# df2.to_csv('Final1.csv')

# Normalizing data

# min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
# df3 = df2[['epoch', 'Position Vector(X)', 'Position Vector(Y)', 'Position Vector(Z)', 'Velocity Vector(X)',
#            'Velocity Vector(Y)', 'Velocity Vector(Z)']].copy()
# X1 = np.array(df3)
# X_scale = min_max_scaler.fit_transform(X1[:, [1, 2, 3, 4, 5, 6]])
# X_scale1 = np.concatenate((X1[:, [0]], X_scale), axis=1)
# df1 = pd.DataFrame(X_scale1, columns=['epoch', 'Position Vector(X)', 'Position Vector(Y)', 'Position Vector(Z)',
#                                       'Velocity Vector(X)',
#                                       'Velocity Vector(Y)', 'Velocity Vector(Z)'])

#Saving Normalized data in csv file

# df2.to_csv('Final2.csv')
df2 = pd.read_csv('Final2.csv', parse_dates=['epoch'], index_col='Date')
missing_values = df2.isna()
print(missing_values)