import collections
import pandas
import os
from TrainingSample import TrainingSample
from TestSample import TestSample


class DataRetriever:

    @staticmethod
    def get_folder_paths(outer_folder_name):
        os.chdir(outer_folder_name)
        folder_paths_list = [os.path.abspath(name) for name in os.listdir() if os.path.isdir(name)]
        paths_dict = dict([(el.split("\\")[-1], el) for el in folder_paths_list])
        [print(k + ": " + v) for k, v in paths_dict.items()]
        return paths_dict

    @staticmethod
    def retrieve_data(folder_name, folder_path):
        os.chdir(folder_name)
        # retrieve the paths of the files in it. Each file represent a set of collected data for a specific sensor
        # (accelerometer or gyroscope) for a specific MetaWear device
        file_paths_list = [os.path.abspath(name) for name in os.listdir(folder_path)]
        acc_data = dict()
        gyro_data = dict()
        for i in range(0, file_paths_list.__len__()):  # for each file
            # parse the path to retrieve the file name
            file_name = file_paths_list[i].split("\\")[-1]
            if "ACC" in file_name:  # if the file name contains ACC it means it contains accelerometer data
                # so read the data and store it in the acc_data dict associated with the device MAC address
                df = pandas.read_csv(file_paths_list[i])
                df.dropna(inplace=True)
                df.reset_index(drop=True, inplace=True)
                acc_data[file_name.split("-")[0]] = df
            else:  # otherwise, it contains gyroscope data
                # so read the data and store it in the gyro_data dict associated with the device MAC address
                df = pandas.read_csv(file_paths_list[i])
                df.dropna(inplace=True)
                df.reset_index(drop=True, inplace=True)
                gyro_data[file_name.split("-")[0]] = df
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
            action_name = ''.join([i for i in k if not i.isdigit()])
            # store the SAMPLE data (encapsulated in a Sample object) in a dictionary associating it to its ACTION name
            samples[action_name].append(TrainingSample(action_name, acc_data, gyro_data))
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
