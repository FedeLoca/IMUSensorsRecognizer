import os

import csv


class CSVSplitter:

    header = 'epoch,"timestamp","x","y","z"'
    header_len = 5

    @staticmethod
    def create_new_file(folder_path_o, new_file_name_o):
        new_file_path = folder_path_o + os.sep + new_file_name_o + ".csv"
        if not os.path.exists(folder_path_o):
            os.makedirs(folder_path_o)
        new_file_o = open(new_file_path, 'w', newline='')
        writer_o = csv.writer(new_file_o)
        writer_o.writerow([CSVSplitter.header])
        return new_file_o, writer_o


if __name__ == "__main__":
    to_split_path = ".." + os.sep + "to-split"
    os.chdir(to_split_path)

    file_paths_list = [os.path.abspath(name) for name in os.listdir(to_split_path)]
    for i in range(0, file_paths_list.__len__()):  # for each file
        # parse the path to retrieve the file name
        file_name_ext = file_paths_list[i].split(os.sep)[-1]
        print("Splitting file " + file_name_ext)
        file_name = file_name_ext[:-13]  # remove _labelled.csv

        split_file_name = file_name.split("-")

        other_count = 1
        folder_path = "." + os.sep + split_file_name[2] + "-other-" + split_file_name[3] + "." + str(other_count)
        new_file_name = file_name + "-other-" + str(other_count)
        new_file, writer = CSVSplitter.create_new_file(folder_path, new_file_name)

        print("Start writing " + new_file_name)

        with open(file_name_ext, 'r') as csvfile:
            datareader = csv.reader(csvfile)
            isHeader = True
            for row in datareader:
                if not isHeader:
                    split_row = row[0].split(",")
                    if split_row.__len__() == (CSVSplitter.header_len + 1):
                        if split_row[CSVSplitter.header_len] == "ciak":
                            real_row = ",".join([split_row[i] for i in range(CSVSplitter.header_len)])
                            writer.writerow([real_row])

                        else:
                            extra_column = "".join([x for x in split_row[CSVSplitter.header_len] if x != '"'])
                            print("Encountered " + extra_column)
                            split_extra_column = extra_column.split(" ")
                            if split_extra_column[0] == "start":
                                new_file.close()
                                other_count += 1
                                print("Stop writing " + new_file_name)

                                gesture_number = split_extra_column[1]
                                gesture_name = split_extra_column[2]
                                if "NO" in split_row[CSVSplitter.header_len]:
                                    folder_path = "." + os.sep + split_file_name[2] + "-INVALID-" + gesture_name + "-" \
                                                  + split_file_name[3] + "." + gesture_number
                                else:
                                    folder_path = "." + os.sep + split_file_name[2] + "-" + gesture_name + "-" \
                                                  + split_file_name[3] + "." + gesture_number
                                new_file_name = file_name + "-" + gesture_name + "-" + gesture_number
                                new_file, writer = CSVSplitter.create_new_file(folder_path, new_file_name)
                                print("Start writing " + new_file_name)

                                real_row = ",".join([split_row[i] for i in range(CSVSplitter.header_len)])
                                writer.writerow([real_row])

                            elif split_extra_column[0] == "end":
                                real_row = ",".join([split_row[i] for i in range(CSVSplitter.header_len)])
                                writer.writerow([real_row])
                                new_file.close()
                                print("Stop writing " + new_file_name)

                                folder_path = "." + os.sep + split_file_name[2] + "-other-" + split_file_name[3] + "." \
                                              + str(other_count)
                                new_file_name = file_name + "-other-" + str(other_count)
                                new_file, writer = CSVSplitter.create_new_file(folder_path, new_file_name)
                                print("Start writing " + new_file_name)
                    else:
                        writer.writerow(row)
                else:
                    isHeader = False
        new_file.close()
        print("Stop writing " + new_file_name)
