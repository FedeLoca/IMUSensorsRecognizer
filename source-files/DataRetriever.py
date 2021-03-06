import collections

import pandas as pd
import os
from TrainingSample import TrainingSample
from TestSample import TestSample
import MobileAVG


class DataRetriever:

    columns_types = {'epoch': 'long', 'timestamp': 'str', 'x': 'float', 'y': 'float', 'z': 'float', 'label': 'str'}
    header_labelled = ["epoch", "timestamp", "x", "y", "z", "label"]
    header = ["epoch", "timestamp", "x", "y", "z"]
    header_len = 5

    @staticmethod
    def get_folder_paths(outer_folder_name):
        os.chdir(outer_folder_name)
        folder_paths_list = [os.path.abspath(name) for name in os.listdir() if os.path.isdir(name)]
        paths_dict = dict([(el.split(os.sep)[-1], el) for el in folder_paths_list])
        [print(k + ": " + v) for k, v in paths_dict.items()]
        return paths_dict

    @staticmethod
    def retrieve_data(folder_name, folder_path):
        os.chdir(folder_name)
        # retrieve the paths of the files in it. Each file represent a set of collected data for a specific sensor
        # (accelerometer or gyroscope) for a specific MetaWear device
        file_paths_list = [os.path.abspath(name) for name in os.listdir(folder_path)]
        gyro_data = None
        acc_data = None
        file_name = None
        for i in range(0, file_paths_list.__len__()):  # for each file
            # parse the path to retrieve the file name
            file_name = file_paths_list[i].split(os.sep)[-1]
            if "ACC" in file_name:  # if the file name contains ACC it means it contains accelerometer data
                df = pd.read_csv(file_paths_list[i])
                df.dropna(inplace=True)
                df.reset_index(drop=True, inplace=True)
                acc_data = df
            else:  # otherwise, it contains gyroscope data
                df = pd.read_csv(file_paths_list[i])
                df.dropna(inplace=True)
                df.reset_index(drop=True, inplace=True)
                gyro_data = df

        if acc_data is None or gyro_data is None:
            print("none: " + file_name)
        if abs(len(acc_data.index) - len(gyro_data.index)) > 10:
            print(file_name)

        return acc_data, gyro_data

    @staticmethod
    def retrieve_training_data(folder):
        print("\nRetrieving training samples:")
        # go to data folder (each folder represents a SAMPLE for an ACTION) and create a dictionary
        # associating every action to its folder's path
        paths_dict = DataRetriever.get_folder_paths(folder)

        samples = collections.defaultdict(list)
        for (k, v) in paths_dict.items():  # for each folder
            # go inside it and retrieve the data using pandas
            acc_data, gyro_data = DataRetriever.retrieve_data(k, v)
            # remove digits from the folder name obtaining the ACTION name
            action_name = k.split("-")[1]
            sample_num = k.split("-")[2]
            # store the SAMPLE data (encapsulated in a Sample object) in a dictionary associating it to its ACTION name
            samples[action_name].append(TrainingSample(action_name, acc_data, gyro_data, sample_num))
            os.chdir("..")

        # now the dictionary samples contains the lists of SAMPLES, with each list associated to its ACTION name
        # print a recap of the retrieved data
        for (k, v) in samples.items():
            print("################## " + k + ": (Number of samples: " + str(v.__len__()) + ")")
            for s in v:
                print(s)

        return samples

    @staticmethod
    def retrieve_test_data(folder):
        print("\nRetrieving test samples:")
        # go to data folder (each folder in it represents a SESSION) and create a dictionary associating
        # every SESSION to its folder's path
        paths_dict = DataRetriever.get_folder_paths(folder)

        session_id = 0
        samples = dict()
        for (k, v) in paths_dict.items():  # for each folder
            # go inside it and retrieve the data using pandas
            acc_data, gyro_data = DataRetriever.retrieve_data(k, v)
            # store the SAMPLE data (encapsulated in a TestSample object) in a dictionary associating it
            # to its SESSION id
            samples[session_id] = TestSample(session_id, acc_data, gyro_data)
            os.chdir("..")
            session_id += 1

        # now the dictionary samples contains the SAMPLES associated to their SESSION name
        # print a recap of the retrieved data
        for (k, v) in samples.items():
            print("################## Session " + k + ":")
            print(v)

        return samples

    @staticmethod
    def retrieve_train_session_data(folder, mobile_average_window_dim):
        print("\nRetrieving training samples:")
        # go to data folder (each folder in it represents a SESSION) and create a dictionary associating
        # every SESSION to its folder's path
        paths_dict = DataRetriever.get_folder_paths(folder)

        samples = collections.defaultdict(list)
        for (k, v) in paths_dict.items():  # for each folder
            os.chdir(v)
            file_paths_list = [os.path.abspath(name) for name in os.listdir(v)]

            gyro_data = None
            acc_data = None
            for i in range(0, file_paths_list.__len__()):  # for each file
                # parse the path to retrieve the file name
                file_name_ext = file_paths_list[i].split(os.sep)[-1]
                file_name = file_name_ext[:-13]  # remove _labelled.csv
                split_file_name = file_name.split("-")
                print("Retrieving data from file " + file_name_ext)

                if "ACC" in file_name:
                    acc_data = pd.read_csv(file_paths_list[i], skiprows=[0], names=DataRetriever.header_labelled)
                    if mobile_average_window_dim > 1:
                        print("Computing mobile averages for sample " + str(k))
                        print("Acc data length before: " + str(len(acc_data.index)))
                        acc_data = MobileAVG.compute_mobile_average(acc_data, mobile_average_window_dim)
                        print("Acc data length after: " + str(len(acc_data.index)))

                else:
                    gyro_data = pd.read_csv(file_paths_list[i], skiprows=[0], names=DataRetriever.header_labelled)
                    if mobile_average_window_dim > 1:
                        print("Computing mobile averages for sample " + str(k))
                        print("Gyro data length before: " + str(len(gyro_data.index)))
                        gyro_data = MobileAVG.compute_mobile_average(gyro_data, mobile_average_window_dim)
                        print("Gyro data length after: " + str(len(gyro_data.index)))

            samples["SESSION"].append(TrainingSample("SESSION", acc_data, gyro_data, k))

        for ss in samples.values():
            print("################## SESSION: \n")
            for s in ss:
                print(s)

        os.chdir("..")
        return samples



