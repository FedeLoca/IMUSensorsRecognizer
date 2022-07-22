import functools
import math
from collections import defaultdict
from operator import add

import numpy as np
import pandas as pd
import time
import scipy.stats as sps
import fastdtw
from scipy.spatial import distance
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from hmmlearn import hmm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as skm
from boruta import BorutaPy
from sklearn.tree import DecisionTreeClassifier

import Features as f
from TrainingSample import TrainingSample


class Classifier:
    # [3 (axes) * 2 (sensors) * 11 (features) + 1 (magnitude) * 2 (sensors)] * 3 (devices)
    max_features_n = 204

    invalid_windows = list()
    valid_windows = list()

    def __init__(self, training_data, test_data, num_actions_dict, actions_num_dict, mobile_average_window_dim, tuning):
        self.y_list = None
        self.x_list = None
        self.feature_names = list()
        self.training_data = training_data
        self.test_data = test_data
        self.num_actions_dict = num_actions_dict
        self.actions_num_dict = actions_num_dict
        self.n_classes = len(self.num_actions_dict.keys())
        self.mobile_average_window_dim = mobile_average_window_dim
        self.tuning = tuning

    def classify(self, model_type, test_size):
        print("\nPreparing data for training:")
        x_train, y_train, x_test, y_test = list(), list(), list(), list()
        for (a, l) in self.x_list.items():
            # random_state=21
            x_train_el, x_test_el, y_train_el, y_test_el = train_test_split(l, self.y_list[a],
                                                                            test_size=test_size,
                                                                            shuffle=True)
            x_train.extend(x_train_el)
            y_train.extend(y_train_el)
            x_test.extend(x_test_el)
            y_test.extend(y_test_el)

        x_test_lengths = 0
        if model_type != 'lstm':
            x_train = np.array(x_train)
            # x_train_lengths = x_train[:, len(self.feature_names):]
            # remove data lengths as they are not features
            x_train = np.delete(x_train, np.s_[len(self.feature_names):], 1)
            y_train = np.array(y_train)
            x_test = np.array(x_test)
            x_test_lengths = x_test[:, len(self.feature_names):]
            x_test = np.delete(x_test, np.s_[len(self.feature_names):], 1)
            y_test = np.array(y_test)
        else:
            x_train = np.array(x_train, dtype=object)
            y_train = np.array(y_train)
            x_test = np.array(x_test, dtype=object)
            y_test = np.array(y_test)

        print("### Features number: " + str(len(x_test[0])))
        print("### Total samples: " + str(len(y_train) + len(y_test)))
        print("### Train samples: " + str(len(y_train)))
        print("### Test samples: " + str(len(y_test)))
        print("### Train samples percentage: " + str((1 - test_size) * 100) + "%")
        valid_windows_n = len(self.valid_windows)
        invalid_windows_n = len(self.invalid_windows)
        if valid_windows_n + invalid_windows_n > 0:
            print("### Total windows: " + str(valid_windows_n + invalid_windows_n))
            print("### Valid windows: " + str(valid_windows_n))
            print("### Invalid windows: " + str(invalid_windows_n))
            invalid_windows_perc = (invalid_windows_n / (valid_windows_n + invalid_windows_n)) * 100
            print("### Invalid windows percentage: " + str(invalid_windows_perc) + "%")

        model = None
        parameters = dict()
        if model_type == 'k-nn':
            # n_neighbors must be equal or lower than the number of train samples
            parameters = {'n_neighbors': [1, 2, 4, 8]}
            model = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                         metric='minkowski', metric_params=None)
        elif model_type == 'rf':
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
            # Criterion
            criterion = ["gini", "entropy", "log_loss"]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]
            # Create the random grid
            parameters = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap,
                           'criterion': criterion}
            model = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=None, min_samples_split=2,
                                           min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt',
                                           max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True,
                                           oob_score=False, n_jobs=-1, random_state=None, warm_start=False,
                                           class_weight=None, ccp_alpha=0.0, max_samples=None)
        elif model_type == 'k-nn-dtw':
            # n_neighbors must be equal or lower than the number of train samples
            parameters = {'n_neighbors': [1, 2, 4, 8]}
            model = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                         metric=Classifier.fast_dtw, metric_params=None)
        elif model_type == 'svm':
            parameters = {'C': [10, 20, 40, 80]}
            model = SVC(C=20.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False,
                        tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1,
                        decision_function_shape='ovr', break_ties=False, random_state=None)
        elif model_type == 'nb':
            model = GaussianNB(priors=None, var_smoothing=1e-09)
        elif model_type == 'lr':
            parameters = {'C': [0.5, 1, 5]}
            model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                                       intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs',
                                       max_iter=10000, multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                                       l1_ratio=None)
        elif model_type == 'dt':
            parameters = {'min_samples_split': [10, 20, 30], 'min_samples_leaf': [1, 2, 5, 10]}
            model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2,
                                           min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                           max_features=None, random_state=None, max_leaf_nodes=None,
                                           min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0)
        elif model_type == 'hmm':
            parameters = {'n_components': [1, 2, 5, 10]}
            # model = hmm.BaseHMM(n_components=1, startprob_prior=1.0, transmat_prior=1.0, algorithm="viterbi",
            #                     random_state=None, n_iter=10, tol=1e-2, verbose=False, params=string.ascii_letters,
            #                     init_params=string.ascii_letters, implementation="log")
            model = hmm.GaussianHMM(n_components=1, covariance_type='diag', min_covar=1e-3, startprob_prior=1.0,
                                    transmat_prior=1.0, means_prior=0, means_weight=0, covars_prior=1e-2,
                                    covars_weight=1, algorithm="viterbi", random_state=None, n_iter=10, tol=1e-2,
                                    verbose=False, params="stmc", init_params="stmc", implementation="log")

        if model_type == 'lstm':
            epochs, batch_size = 15, 64
            model = Classifier.lstm(x_train, y_train)
            print("\nTraining...")
            start_time = time.time()
            model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
            train_seconds = time.time() - start_time
            print("\n\n--- %s train seconds ---\n\n" % train_seconds)

            print("\nTest...")
            start_time = time.time()
            prediction = model.predict(x_test, y_test, batch_size=batch_size)
            predict_seconds = time.time() - start_time
            print("\n\n--- %s predict seconds ---\n\n" % predict_seconds)
            y_pred = (prediction > 0.5)
            confusion_matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
            for i in range(0, len(y_test)):
                print('Result: Real: {},  Predicted: {}'.format(y_test[i], prediction[i]))
            score = model.score(x_test, y_test)
            print("Score: " + str(score))
        else:
            print("Are all values valid? " + str(not np.any(np.isnan(x_train)) and np.all(np.isfinite(x_train))))

            if self.tuning:
                model = RandomizedSearchCV(estimator=model, param_distributions=parameters, n_iter=150, cv=3,
                                           verbose=2, random_state=42, n_jobs=-1)
                # model = GridSearchCV(estimator=model, param_grid=parameters, cv=5, n_jobs=-1, verbose=2)

            print("\nTraining...")
            start_time = time.time()
            model.fit(x_train, y_train)
            train_seconds = time.time() - start_time
            print("\n\n--- %s train seconds ---\n\n" % train_seconds)

            if self.tuning:
                print("Best parameters set found on development set:")
                print(model.best_params_)
                # print("Grid scores on development set:")
                # means = model.cv_results_["mean_test_score"]
                # stds = model.cv_results_["std_test_score"]
                # for mean, std, params in zip(means, stds, model.cv_results_["params"]):
                #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
                # print()

            print("\nTest...")
            start_time = time.time()
            prediction = model.predict(x_test)
            predict_seconds = time.time() - start_time
            print("\n\n--- %s predict seconds ---\n\n" % predict_seconds)
            confusion_matrix = skm.confusion_matrix(y_test, prediction)
            for i in range(0, len(y_test)):
                print('Result: Real: {},  Predicted: {}, Lengths: {}'.format(y_test[i], prediction[i], x_test_lengths[i]))
                #  print("Test samples: " + str(y_test))
                #  print("Prediction: " + str(prediction))
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

        return score, confusion_matrix, train_seconds, predict_seconds

    # returns a list containing, for every action, an x matrix in which each row contains all the features for a
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

    # returns a list containing, for every action, an x matrix in which each row contains all the features for a
    # window and an array y of the row labels
    def compute_features_on_windows(self, window_dim, overlap, from_session):
        print("Computing features...")
        x_list = defaultdict(list)
        y_list = defaultdict(list)
        row_number = 0
        for action in self.training_data.keys():
            for sample in self.training_data[action]:
                windows = self.split_in_windows(sample, window_dim, overlap, from_session)
                # print("Windows number: " + str(len(windows)))
                # print("Window example: " + windows[0].__str__())
                for window in windows:
                    y_list[window.action_name].append(self.actions_num_dict[window.action_name])
                    row = self.compute_row(window, row_number)
                    x_list[window.action_name].append(row)
                    row_number += 1
        self.x_list = x_list
        self.y_list = y_list

    def compute_row(self, sample, row_number):
        row = list()
        acc_x, acc_y, acc_z = sample.get_acc_axes()
        gyro_x, gyro_y, gyro_z = sample.get_gyro_axes()
        axes = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        axes_names = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']

        # print("Computing mobile averages...")
        axes = [f.moving_average_conv(a, self.mobile_average_window_dim) for a in axes]

        # FEATURES
        # mean
        i = 0
        for a in axes:
            row.append(np.mean(a))
            if row_number == 0:
                self.feature_names.append("mean " + "-" + axes_names[i])
                i += 1
        # variance
        i = 0
        for a in axes:
            row.append(np.var(a))
            if row_number == 0:
                self.feature_names.append("variance " + "-" + axes_names[i])
                i += 1
        # standard deviation
        i = 0
        for a in axes:
            row.append(np.std(a))
            if row_number == 0:
                self.feature_names.append("standard deviation " + "-" + axes_names[i])
                i += 1
        # mean squared error
        # i = 0
        # for a in axes:
        #    row.append(skm.mean_squared_error(observed, predicted))
        #    if sample_num == 0:
        #       self.feature_names.append("mean squared error " + "-" + axes_names[i])
        #       i += 1
        # kurtosis
        i = 0
        for a in axes:
            row.append(sps.kurtosis(a))
            if row_number == 0:
                self.feature_names.append("kurtosis " + "-" + axes_names[i])
                i += 1
        # symmetry
        i = 0
        for a in axes:
            row.append(f.symmetry(a))
            if row_number == 0:
                self.feature_names.append("symmetry " + "-" + axes_names[i])
                i += 1
        # zero-crossing rate
        i = 0
        for a in axes:
            row.append(f.zero_crossing(a))
            if row_number == 0:
                self.feature_names.append("zero-crossing rate " + "-" + axes_names[i])
                i += 1
        # difference between max and min
        i = 0
        for a in axes:
            row.append(f.min_max_diff(a))
            if row_number == 0:
                self.feature_names.append("max-min " + "-" + axes_names[i])
                i += 1
        # number of peaks
        i = 0
        for a in axes:
            row.append(f.peaks_number(a))
            if row_number == 0:
                self.feature_names.append("peaks " + "-" + axes_names[i])
                i += 1
        # energy
        i = 0
        for a in axes:
            row.append(f.energy(a))
            if row_number == 0:
                self.feature_names.append("energy " + "-" + axes_names[i])
                i += 1
        # pearson correlation
        row.append(f.correlation(acc_x, acc_y))
        row.append(f.correlation(acc_y, acc_z))
        row.append(f.correlation(acc_x, acc_z))
        row.append(f.correlation(gyro_x, gyro_y))
        row.append(f.correlation(gyro_y, gyro_z))
        row.append(f.correlation(gyro_x, gyro_z))
        if row_number == 0:
            self.feature_names.append("acc correlation xy")
            self.feature_names.append("acc correlation yz")
            self.feature_names.append("acc correlation xz")
            self.feature_names.append("gyro correlation xy")
            self.feature_names.append("acc correlation yz")
            self.feature_names.append("acc correlation xz")
        # magnitude
        row.append(f.magnitude(acc_x, acc_y, acc_z))
        row.append(f.magnitude(gyro_x, gyro_y, gyro_z))
        if row_number == 0:
            self.feature_names.append("acc magnitude")
            self.feature_names.append("gyro magnitude")

        acc_x, _, _ = sample.get_acc_axes()
        gyro_x, _, _ = sample.get_gyro_axes()
        # append data length to keep track of it and be able to associate it with prediction success
        row.append(len(acc_x))
        row.append(len(gyro_x))

        for i in range(0, len(row)):
            if math.isnan(row[i]) or math.isinf(row[i]):
                print("Invalid: " + self.feature_names[i])

        return row

    def split_in_windows(self, sample, window_dim, overlap, from_session):
        print("Start splitting sample " + str(sample.sample_num))

        windows = list()
        acc_data = sample.acc_data
        gyro_data = sample.acc_data

        # start splitting when all sensors started collecting data and finish when any sensor stopped collecting data
        # print(acc_data.columns.tolist())
        first_epoch = max(acc_data['epoch'].iloc[0], gyro_data['epoch'].iloc[0])
        last_epoch = min(acc_data['epoch'].iloc[-1], gyro_data['epoch'].iloc[-1])

        start_epoch = first_epoch
        end_epoch = start_epoch + window_dim
        start_next_epoch = end_epoch - int(window_dim * overlap)

        last_acc_split = 0
        acc_epochs = acc_data['epoch']
        for i in range(0, len(acc_epochs)):
            if acc_epochs.iloc[i] >= start_epoch:
                last_acc_split = i
                break

        last_gyro_split = 0
        gyro_epochs = gyro_data['epoch']
        for i in range(0, len(gyro_epochs)):
            if gyro_epochs.iloc[i] >= start_epoch:
                last_gyro_split = i
                break

        # print("first-last " + str(last_epoch-first_epoch))
        # print("first " + str(first_epoch))
        # print("last " + str(last_epoch))
        current_action = "other"
        while start_epoch < last_epoch:
            valid = True

            # print("start " + str(start_epoch))
            # print("start next " + str(start_next_epoch))
            # print("end " + str(end_epoch))

            if from_session:
                # print("Acc window extraction...")
                new_acc_data, valid, last_acc_split, action_name, current_action = \
                    Classifier.extract_session_window(last_acc_split, acc_epochs, acc_data, start_epoch, end_epoch,
                                                      start_next_epoch, valid, current_action, True)
                # print("Gyro window extraction...")
                new_gyro_data, valid, last_gyro_split, _, _ = \
                    Classifier.extract_session_window(last_gyro_split, gyro_epochs, gyro_data, start_epoch, end_epoch,
                                                      start_next_epoch, valid, "", False)

                window = TrainingSample(action_name, new_acc_data, new_gyro_data, sample.sample_num)
            else:
                # print("Acc window extraction...")
                new_acc_data, valid, last_acc_split = Classifier.extract_window(last_acc_split, acc_epochs, acc_data,
                                                                                start_epoch, end_epoch, start_next_epoch,
                                                                                valid)
                # print("Gyro window extraction...")
                new_gyro_data, valid, last_gyro_split = Classifier.extract_window(last_gyro_split, gyro_epochs, gyro_data,
                                                                                  start_epoch, end_epoch, start_next_epoch,
                                                                                  valid)

                window = TrainingSample(sample.action_name, new_acc_data, new_gyro_data, sample.sample_num)

            if valid:
                self.valid_windows.append(window)
                windows.append(window)
            else:
                self.invalid_windows.append(window)
            start_epoch = start_next_epoch
            end_epoch = start_epoch + window_dim
            start_next_epoch = end_epoch - int(window_dim * overlap)

        return windows

    @staticmethod
    def extract_window(last_split, epochs, data, start_epoch, end_epoch, start_next_epoch, valid):
        window_data = list()
        count = -1
        start_next_split = -1
        for i in range(last_split, len(epochs)):
            count = count + 1
            row = list()
            epoch = epochs.iloc[i]
            if start_next_split < 0 and epoch >= start_next_epoch:
                start_next_split = i

            if start_epoch <= epoch < end_epoch:
                for col in data.columns:
                    row.append(data[col].iloc[i])
                window_data.append(row)
            else:
                break

        print(str(len(window_data)))
        if len(window_data) > 0:
            window_df = pd.DataFrame(window_data, columns=data.columns)
        else:
            valid = False
            window_df = pd.DataFrame(columns=data.columns)

        return window_df, valid, start_next_split

    @staticmethod
    def extract_session_window(last_split, epochs, data, start_epoch, end_epoch, start_next_epoch, valid,
                               current_action, determine_action):
        window_data = list()
        labels = data["label"]
        actions = defaultdict(list)
        actions[current_action].append(start_epoch)
        start_next_split = -1
        next_current_action = current_action
        for i in range(last_split, len(epochs)):
            row = list()
            label = labels.iloc[i]
            epoch = epochs.iloc[i]

            if start_next_split < 0 and epoch >= start_next_epoch:
                next_current_action = current_action
                start_next_split = i

            if start_epoch <= epoch < end_epoch:
                if determine_action and not pd.isnull(label) and label != "ciak":
                    split_extra_column = label.split(" ")
                    # print(actions)
                    if split_extra_column[0] == "start":
                        current_action = split_extra_column[2]
                        actions["other"][-1] = epoch - actions["other"][-1]
                        actions[current_action].append(epoch)
                        # print("START " + split_extra_column[2])
                    elif split_extra_column[0] == "end":
                        # print("END " + split_extra_column[2])
                        current_action = "other"
                        actions[split_extra_column[2]][-1] = epoch - actions[split_extra_column[2]][-1]
                        actions["other"].append(epoch)

                for col in data.columns[:-1]:
                    row.append(data[col].iloc[i])
                window_data.append(row)
            else:
                actions[current_action][-1] = epoch - actions[current_action][-1]
                break

        action_name = None
        if determine_action:
            duration_max = -1
            action_name = ""
            for (k, v) in actions.items():
                if len(v) == 0:
                    duration = 0
                else:
                    duration = functools.reduce(add, v)

                if duration > duration_max:
                    if k != "other" or (k == "other" and action_name == ""):
                        duration_max = duration
                        action_name = k
            # print(action_name + ": " + str(len(window_data)))

        if len(window_data) > 0:
            window_df = pd.DataFrame(window_data, columns=data.columns[:-1])
        else:
            valid = False
            window_df = pd.DataFrame(columns=data.columns[:-1])

        return window_df, valid, start_next_split, action_name, next_current_action

    @staticmethod
    def dtw(a, b):
        an = a.size
        bn = b.size
        pointwise_distance = distance.cdist(a.reshape(-1, 1), b.reshape(-1, 1))
        cumdist = np.matrix(np.ones((an+1, bn+1)) * np.inf)
        cumdist[0, 0] = 0

        for ai in range(an):
            for bi in range(bn):
                minimum_cost = np.min([cumdist[ai, bi+1],
                                       cumdist[ai+1, bi],
                                       cumdist[ai, bi]])
                cumdist[ai+1, bi+1] = pointwise_distance[ai, bi] + minimum_cost

        return cumdist[an, bn]

    @staticmethod
    def fast_dtw(a, b):
        return fastdtw.fastdtw(a, b)[0]

    # returns a list containing, for every action, an x matrix in which each row contains all the accelerometer and
    # gyroscope axes (6) for all windows and an array y of the row labels
    def compute_lstm_data(self, window_dim):
        print("Computing features...")
        x_list = list()
        y_list = list()
        row_number = 0
        for action in self.training_data.keys():
            x = list()
            y = list()
            for sample in self.training_data[action]:
                windows = self.split_in_windows(sample, window_dim)
                print("Windows number: " + str(len(windows)))
                print("Window example: " + windows[0].__str__())
                row = list()
                for window in windows:
                    acc_x, acc_y, acc_z = window.get_acc_axes()
                    gyro_x, gyro_y, gyro_z = window.get_gyro_axes()
                    row.append([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
                x.append(row)
                y.append(self.actions_num_dict[action])
                row_number += 1
            x_list.append(x)
            y_list.append(y)
        self.x_list = x_list
        self.y_list = y_list

    @staticmethod
    def lstm(x_train, y_train):
        n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
        model = Sequential()
        model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

