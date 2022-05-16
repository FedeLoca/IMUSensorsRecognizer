class Sample:
    def __init__(self, acc_data, gyro_data):
        self.acc_data = acc_data  # dictionary of accelerometer data as DataFrame associated to its device MAC address
        self.gyro_data = gyro_data  # dictionary of gyroscope data as DataFrame associated to its device MAC address
        self.sensors_num = len(acc_data)

    # get all axes of acceleration data for a single sensor
    def get_acc_axes(self, index):
        c = 0
        for k in self.acc_data.keys():
            if index == c:
                return self.acc_data[k]['x'], self.acc_data[k]['y'], self.acc_data[k]['z']
            c += 1

    # get all axes of acceleration data for a single sensor
    def get_gyro_axes(self, index):
        c = 0
        for k in self.gyro_data.keys():
            if index == c:
                return self.gyro_data[k]['x'], self.gyro_data[k]['y'], self.gyro_data[k]['z']
            c += 1

    # get a specific column of acceleration data for a specific sensor
    def get_acc_array(self, sensor_name, column_name):
        return self.acc_data[sensor_name][column_name]

    # get a specific column of angular velocity data for a specific sensor
    def get_gyro_array(self, sensor_name, column_name):
        return self.gyro_data[sensor_name][column_name]

    # get a specific column of acceleration data for all sensor
    def get_acc_arrays(self, column_name):
        res = dict()
        for (sensor_name, data) in self.acc_data.items():
            res[sensor_name] = data[column_name]
        return res

    # get a specific column of angular velocity data for all sensor
    def get_gyro_arrays(self, column_name):
        res = dict()
        for (sensor_name, data) in self.gyro_data.items():
            res[sensor_name] = data[column_name]
        return res

    def print_data(self):
        s = "ACC: "
        for (k, v) in self.acc_data.items():
            s += k + " (rows: " + str(len(v.index)) + "), "
        s += "\nGYRO: "
        for (k, v) in self.gyro_data.items():
            s += k + " (rows: " + str(len(v.index)) + "), "
        # s += "\n" + str(self.acc_data['DFDFC18A6D4A'])
        return s
