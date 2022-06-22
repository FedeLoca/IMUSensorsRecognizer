import os
import csv

import pandas


class CSVLabeller:

    header = 'epoch,"timestamp","x","y","z"'
    header_len = 5

    @staticmethod
    def create_new_file(new_file_path):
        new_file = open(new_file_path, 'w', newline='')
        writer = csv.writer(new_file)
        writer.writerow([CSVLabeller.header])
        return new_file, writer


if __name__ == "__main__":
    to_label_path = ".." + os.sep + "to-label"
    os.chdir(to_label_path)

    bookmarks_path = ""
    acc_path = ""
    gyro_path = ""
    new_acc_path, new_acc_file, acc_writer = None, None, None
    new_gyro_path, new_gyro_file, gyro_writer = None, None, None
    file_paths_list = [os.path.abspath(name) for name in os.listdir(to_label_path)]
    for i in range(0, file_paths_list.__len__()):  # for each file
        # parse the path to retrieve the file name
        file_name_ext = file_paths_list[i].split(os.sep)[-1]
        if "ACC" in file_name_ext:
            print("Labelling ACC file " + file_name_ext)
            acc_path = file_paths_list[i]
            new_acc_path = acc_path.replace(".csv", "_labelled.csv")
            new_acc_file, acc_writer = CSVLabeller.create_new_file(new_acc_path)
        elif "GYRO" in file_name_ext:
            print("Labelling GYRO file " + file_name_ext)
            gyro_path = file_paths_list[i]
            new_gyro_path = gyro_path.replace(".csv", "_labelled.csv")
            new_gyro_file, gyro_writer = CSVLabeller.create_new_file(new_gyro_path)
        else:
            print("With bookmarks from file " + file_name_ext)
            bookmarks_path = file_paths_list[i]

    bookmarks_df = pandas.read_excel(bookmarks_path, header=0)
    labels = list()
    for i in range(0, len(bookmarks_df["Data"])):
        label = bookmarks_df.iloc[i]["Unnamed: 0"].lower()
        bookmark = bookmarks_df.iloc[i]["Data"]
        print("Label " + label + " at bookmark " + str(round(bookmark, 3)))
        labels.append((label, round(bookmark, 3)))
    #print(labels)

    with open(acc_path, 'r') as acc_file:
        acc_reader = csv.reader(acc_file)
        next(acc_reader)
        for (label, bookmark) in labels:
            print("Labelling acc with label " + label + " at bookmark " + str(bookmark))
            labelled = False
            while not labelled:
                acc_row = next(acc_reader)
                split_row = acc_row[0].split(",")
                timestamp = float(split_row[1][-7:-1].replace(":", "."))
                print("Timestamp " + str(timestamp))
                if "ciak" in label and timestamp == bookmark:
                    new_acc_row = acc_row[0] + ",\"" + label + "\""
                    labelled = True
                elif "start" in label and timestamp >= bookmark \
                        and (int(timestamp) == int(bookmark) or int(timestamp) == (int(bookmark) + 1) % 60):
                    new_acc_row = acc_row[0] + ",\"" + label + "\""
                    labelled = True
                elif "end" in label and timestamp >= bookmark \
                        and (int(timestamp) == int(bookmark) or int(timestamp) == (int(bookmark) + 1) % 60):
                    new_acc_row = acc_row[0] + ",\"" + label + "\""
                    labelled = True
                else:
                    new_acc_row = acc_row[0]
                acc_writer.writerow([new_acc_row])

        for acc_row in acc_reader:
            acc_writer.writerow(acc_row)
        new_acc_file.close()

    with open(gyro_path, 'r') as gyro_file:
        gyro_reader = csv.reader(gyro_file)
        next(gyro_reader)
        for (label, bookmark) in labels:
            print("Labelling gyro with label " + label + " at bookmark " + str(bookmark))
            labelled = False
            while not labelled:
                gyro_row = next(gyro_reader)
                split_row = gyro_row[0].split(",")
                timestamp = float(split_row[1][-7:-1].replace(":", "."))
                #print("Timestamp " + str(timestamp))
                if "ciak" in label and timestamp == bookmark:
                    new_gyro_row = gyro_row[0] + ",\"" + label + "\""
                    labelled = True
                elif "start" in label and timestamp >= bookmark \
                        and (int(timestamp) == int(bookmark) or int(timestamp) == (int(bookmark) + 1) % 60):
                    new_gyro_row = gyro_row[0] + ",\"" + label + "\""
                    labelled = True
                elif "end" in label and timestamp >= bookmark \
                        and (int(timestamp) == int(bookmark) or int(timestamp) == (int(bookmark) + 1) % 60):
                    new_gyro_row = gyro_row[0] + ",\"" + label + "\""
                    labelled = True
                else:
                    new_gyro_row = gyro_row[0]
                gyro_writer.writerow([new_gyro_row])

        for gyro_row in gyro_reader:
            gyro_writer.writerow(gyro_row)
        new_gyro_file.close()
