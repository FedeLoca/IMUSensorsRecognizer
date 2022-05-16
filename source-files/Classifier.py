import collections
import math

import numpy as np
import pandas as pd
import scipy.stats as sps
# from keras import Sequential
# from keras.layers import LSTM, Dropout, Dense
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
from boruta import BorutaPy
import Features as f
from TrainingSample import TrainingSample


class Classifier:

    # [3 (axes) * 2 (sensors) * 11 (features) + 1 (magnitude) * 2 (sensors)] * 3 (devices)
    max_features_n = 204

    invalid_windows = list()
    valid_windows = list()

    def __init__(self, training_data, test_data, num_actions_dict, actions_num_dict):
        self.y_list = None
        self.x_list = None
        self.feature_names = list()
        self.training_data = training_data
        self.test_data = test_data
        self.num_actions_dict = num_actions_dict
        self.actions_num_dict = actions_num_dict
        self.n_classes = len(self.num_actions_dict.keys())

    def classify(self, model_type, test_size):
        print("\nPreparing data for training:")
        x_train, y_train, x_test, y_test = list(), list(), list(), list()
        for i in range(0, self.x_list.__len__()):
            # random_state=21
            x_train_el, x_test_el, y_train_el, y_test_el = train_test_split(self.x_list[i], self.y_list[i],
                                                                            test_size=test_size,
                                                                            shuffle=True)
            x_train.extend(x_train_el)
            y_train.extend(y_train_el)
            x_test.extend(x_test_el)
            y_test.extend(y_test_el)
        x_train = np.array(x_train)
        x_train = np.delete(x_train, np.s_[len(self.feature_names):], 1)  # remove data lengths as they are not feat.
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        print("### before columns: " + str(len(x_test[0])))
        x_test_lengths = x_test[:, len(self.feature_names):]
        x_test = np.delete(x_test, np.s_[len(self.feature_names):], 1)
        print("### after columns: " + str(len(x_test[0])))
        print("### length columns: " + str(len(x_test_lengths[0])))
        y_test = np.array(y_test)
        print("### features names length: " + str(len(self.feature_names)))
        print("### Features number: " + str(len(x_test[0])))
        print("### Total samples: " + str(len(y_train) + len(y_test)))
        print("### Train samples: " + str(len(y_train)))
        print("### Test samples: " + str(len(y_test)))
        print("### Train samples percentage: " + str((1 - test_size) * 100) + "%")
        valid_windows_n = len(self.valid_windows)
        invalid_windows_n = len(self.invalid_windows)
        print("### Total windows: " + str(valid_windows_n + invalid_windows_n))
        print("### Valid windows: " + str(valid_windows_n))
        print("### Invalid windows: " + str(invalid_windows_n))
        invalid_windows_perc = (invalid_windows_n/(valid_windows_n + invalid_windows_n)) * 100
        print("### Invalid windows percentage: " + str(invalid_windows_perc) + "%")

        model = None
        if model_type == 'k-nn':
            # n_neighbors must be equal or lower than the number of train samples
            model = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                         metric='minkowski', metric_params=None)
        elif model_type == 'random forest':
            model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                                           min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                           max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True,
                                           oob_score=False, n_jobs=None, random_state=None, warm_start=False,
                                           class_weight=None, ccp_alpha=0.0, max_samples=None)

        print("Are all values valid? " + str(not np.any(np.isnan(x_train)) and np.all(np.isfinite(x_train))))

        print("\nTraining...")
        model.fit(x_train, y_train)

        print("\nTest...")
        prediction = model.predict(x_test)
        confusion_matrix = skm.confusion_matrix(y_test, prediction)
        for i in range(0, len(y_test)):
            print('Result: Real: {},  Predicted: {}, Lengths: {}'.format(y_test[i], prediction[i], x_test_lengths[i]))
        # print("Test samples: " + str(y_test))
        # print("Prediction: " + str(prediction))
        score = model.score(x_test, y_test)
        print("Score: " + str(score))

        '''
        print("\nFeature ranking...")
        feature_selector = BorutaPy(model, n_estimators='auto', random_state=1)
        feature_selector.fit(x_train, y_train)  # it changes number of features in the model???? model.n_features_
        feature_ranks = list(zip(self.feature_names, feature_selector.ranking_, feature_selector.support_))
        for feat in feature_ranks:
            print('Feature: {:<25} Rank: {},  Keep: {}'.format(feat[0], feat[1], feat[2]))
        # filter input down to selected features
        x_train_filtered = feature_selector.transform(x_train)
        print("\nNew number of features: " + str(len(x_train_filtered)))
        '''

        return score, confusion_matrix

    # returns a list containing, for every action, of an x matrix in which each row contains all the features for a
    # sample and an array y of the row labels
    def compute_features(self):
        print("Computing features...")
        x_list = list()
        y_list = list()
        row_number = 0
        for action in self.training_data.keys():
            x = list()
            y = list()
            for sample in self.training_data[action]:
                y.append(self.actions_num_dict[action])
                row = self.compute_row(sample, row_number)
                x.append(row)
                row_number += 1
            x_list.append(x)
            y_list.append(y)
        self.x_list = x_list
        self.y_list = y_list

    # returns a list containing, for every action, of an x matrix in which each row contains all the features for a
    # window and an array y of the row labels
    def compute_features_on_windows(self, window_dim):
        print("Computing features...")
        x_list = list()
        y_list = list()
        row_number = 0
        for action in self.training_data.keys():
            x = list()
            y = list()
            for sample in self.training_data[action]:
                windows = self.split_in_windows(sample, window_dim)
                # print("Windows number: " + str(len(windows)))
                # print("Window example: " + windows[0].__str__())
                for window in windows:
                    y.append(self.actions_num_dict[action])
                    row = self.compute_row(window, row_number)
                    x.append(row)
                    row_number += 1
            x_list.append(x)
            y_list.append(y)
        self.x_list = x_list
        self.y_list = y_list

    def compute_row(self, sample, row_number):
        row = list()
        for sensor_i in range(0, sample.sensors_num):
            acc_x, acc_y, acc_z = sample.get_acc_axes(sensor_i)
            gyro_x, gyro_y, gyro_z = sample.get_gyro_axes(sensor_i)
            axes = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
            sensor_names = list(sample.acc_data.keys())
            axes_names = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']

            # FEATURES
            # mean
            i = 0
            for a in axes:
                row.append(np.mean(a))
                if row_number == 0:
                    self.feature_names.append("mean " + "-" + axes_names[i] + "-" + sensor_names[sensor_i])
                    i += 1
            # variance
            i = 0
            for a in axes:
                row.append(np.var(a))
                if row_number == 0:
                    self.feature_names.append("variance " + "-" + axes_names[i] + "-" + sensor_names[sensor_i])
                    i += 1
            # standard deviation
            i = 0
            for a in axes:
                row.append(np.std(a))
                if row_number == 0:
                    self.feature_names.append("standard deviation " + "-" + axes_names[i] + "-" + sensor_names[sensor_i])
                    i += 1
            # mean squared error
            # i = 0
            # for a in axes:
            #    row.append(skm.mean_squared_error(observed, predicted))
            #    if sample_num == 0:
            #       self.feature_names.append("mean squared error " + "-" + axes_names[i] + "-" + sensor_names[sensor_num])
            #       i += 1
            # kurtosis
            i = 0
            for a in axes:
                row.append(sps.kurtosis(a))
                if row_number == 0:
                    self.feature_names.append("kurtosis " + "-" + axes_names[i] + "-" + sensor_names[sensor_i])
                    i += 1
            # symmetry
            i = 0
            for a in axes:
                row.append(f.symmetry(a))
                if row_number == 0:
                    self.feature_names.append("symmetry " + "-" + axes_names[i] + "-" + sensor_names[sensor_i])
                    i += 1
            # zero-crossing rate
            i = 0
            for a in axes:
                row.append(f.zero_crossing(a))
                if row_number == 0:
                    self.feature_names.append("zero-crossing rate " + "-" + axes_names[i] + "-" + sensor_names[sensor_i])
                    i += 1
            # difference between max and min
            i = 0
            for a in axes:
                row.append(f.min_max_diff(a))
                if row_number == 0:
                    self.feature_names.append("max-min " + "-" + axes_names[i] + "-" + sensor_names[sensor_i])
                    i += 1
            # number of peaks
            i = 0
            for a in axes:
                row.append(f.peaks_number(a))
                if row_number == 0:
                    self.feature_names.append("peaks " + "-" + axes_names[i] + "-" + sensor_names[sensor_i])
                    i += 1
            # energy
            i = 0
            for a in axes:
                row.append(f.energy(a))
                if row_number == 0:
                    self.feature_names.append("energy " + "-" + axes_names[i] + "-" + sensor_names[sensor_i])
                    i += 1
            # pearson correlation
            row.append(f.correlation(acc_x, acc_y))
            row.append(f.correlation(acc_y, acc_z))
            row.append(f.correlation(acc_x, acc_z))
            row.append(f.correlation(gyro_x, gyro_y))
            row.append(f.correlation(gyro_y, gyro_z))
            row.append(f.correlation(gyro_x, gyro_z))
            if row_number == 0:
                self.feature_names.append("acc correlation xy-" + sensor_names[sensor_i])
                self.feature_names.append("acc correlation yz-" + sensor_names[sensor_i])
                self.feature_names.append("acc correlation xz-" + sensor_names[sensor_i])
                self.feature_names.append("gyro correlation xy-" + sensor_names[sensor_i])
                self.feature_names.append("acc correlation yz-" + sensor_names[sensor_i])
                self.feature_names.append("acc correlation xz-" + sensor_names[sensor_i])
            # magnitude
            row.append(f.magnitude(acc_x, acc_y, acc_z))
            row.append(f.magnitude(gyro_x, gyro_y, gyro_z))
            if row_number == 0:
                self.feature_names.append("acc magnitude-" + sensor_names[sensor_i])
                self.feature_names.append("gyro magnitude-" + sensor_names[sensor_i])

        for sensor_i in range(0, sample.sensors_num):
            acc_x, _, _ = sample.get_acc_axes(sensor_i)
            gyro_x, _, _ = sample.get_gyro_axes(sensor_i)
            # append data length to keep track of it and be able to associate it with prediction success
            row.append(len(acc_x))
            row.append(len(gyro_x))

        for i in range(0, len(row)):
            if math.isnan(row[i]) or math.isinf(row[i]):
                print(self.feature_names[i])

        return row

    def split_in_windows(self, sample, window_dim):
        windows = list()
        acc_data = sample.acc_data
        gyro_data = sample.acc_data
        sensors_num = len(acc_data.values())

        # start splitting when all sensors started collecting data and finish when any sensor stopped collecting data
        first_epoch = 0
        last_epoch = 0
        if sensors_num > 0:
            first_epoch = list(acc_data.values())[0]['epoch'].iloc[0]
            last_epoch = list(acc_data.values())[0]['epoch'].iloc[-1]
        for df in acc_data.values():
            epochs = df['epoch']
            if epochs.iloc[0] > first_epoch:
                first_epoch = epochs.iloc[0]
            if epochs.iloc[-1] < last_epoch:
                last_epoch = epochs.iloc[-1]

        start_epoch = first_epoch
        end_epoch = start_epoch + window_dim

        last_acc_splits = dict()
        for (s, df) in acc_data.items():
            epochs = df['epoch']
            for i in range(0, len(epochs)):
                if epochs.iloc[i] >= start_epoch:
                    last_acc_splits[s] = i
                    break

        last_gyro_splits = dict()
        for (s, df) in gyro_data.items():
            epochs = df['epoch']
            for i in range(0, len(epochs)):
                if epochs.iloc[i] >= start_epoch:
                    last_gyro_splits[s] = i
                    break

        #print("first-last " + str(last_epoch-first_epoch))
        #print("first tot " + str(first_epoch))
        #print("last tot " + str(last_epoch))
        while start_epoch < last_epoch:
            new_acc_data = dict()
            new_gyro_data = dict()
            valid = True
            for (s, df) in acc_data.items():
                epochs = df['epoch']
                window_data = list()
                j = 0
                for i in range(last_acc_splits[s], len(epochs)):
                    j = i
                    row = list()
                    # print(str(start_epoch) + "<=" + str(epochs.iloc[i]) + "<" + str(end_epoch))
                    if start_epoch <= epochs.iloc[i] < end_epoch:
                        for col in df.columns:
                            row.append(df[col].iloc[i])
                        window_data.append(row)
                        # print(str(len(window_data)))
                    else:
                        break
                if len(window_data) > 0:
                    window_df = pd.DataFrame(window_data, columns=df.columns)
                else:
                    valid = False
                    window_df = pd.DataFrame(columns=df.columns)
                new_acc_data[s] = window_df
                last_acc_splits[s] = j

            for (s, df) in gyro_data.items():
                epochs = df['epoch']
                window_data = list()
                j = 0
                for i in range(last_gyro_splits[s], len(epochs)):
                    j = i
                    row = list()
                    if start_epoch <= epochs.iloc[i] < end_epoch:
                        for col in df.columns:
                            row.append(df[col].iloc[i])
                        window_data.append(row)
                    else:
                        break
                if len(window_data) > 0:
                    window_df = pd.DataFrame(window_data, columns=df.columns)
                else:
                    valid = False
                    window_df = pd.DataFrame(columns=df.columns)
                new_gyro_data[s] = window_df
                last_gyro_splits[s] = j

            window = TrainingSample(sample.action_name, new_acc_data, new_gyro_data)
            if valid:
                self.valid_windows.append(window)
                windows.append(window)
            else:
                self.invalid_windows.append(window)
            start_epoch = end_epoch
            end_epoch = start_epoch + window_dim

        return windows

    '''
    def lstm(self, x_train, y_train, x_test, y_test):
        verbose, epochs, batch_size = 0, 15, 64
        n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
        model = Sequential()
        model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
        # evaluate model
        _, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
        return accuracy
    '''
