import numpy as np
import pandas as pd


def compute_mobile_average(data, mobile_average_window_dim):
    filtered_data = list()
    considered_rows = list()
    new_row = list()
    for index, row in data.iterrows():
        considered_rows.append(row)

        if len(considered_rows) == mobile_average_window_dim:
            new_row.append(0)
            new_row.append("")
            new_row.append(0)
            new_row.append(0)
            new_row.append(0)
            new_row.append(np.NaN)
            for i in range(mobile_average_window_dim):
                new_row[0] += considered_rows[i]["epoch"]
                new_row[1] = considered_rows[i]["timestamp"]
                new_row[2] += considered_rows[i]["x"]
                new_row[3] += considered_rows[i]["y"]
                new_row[4] += considered_rows[i]["z"]
                label = considered_rows[i]["label"]
                if i == int(mobile_average_window_dim / 2) and not pd.isnull(label):
                    new_row[5] = label
            new_row[0] = int(new_row[0] / mobile_average_window_dim)
            new_row[2] /= mobile_average_window_dim
            new_row[3] /= mobile_average_window_dim
            new_row[4] /= mobile_average_window_dim
            filtered_data.append(new_row)
            new_row = list()
            del considered_rows[0]

    return pd.DataFrame(filtered_data, columns=data.columns)
