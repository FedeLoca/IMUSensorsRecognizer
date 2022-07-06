class Sample:
    def __init__(self, acc_data, gyro_data):
        self.acc_data = acc_data  # accelerometer data as DataFrame
        self.gyro_data = gyro_data  # gyroscope data as DataFrame

    # get all axes of acceleration data
    def get_acc_axes(self):
        return self.acc_data['x'], self.acc_data['y'], self.acc_data['z']

    # get all axes of acceleration data
    def get_gyro_axes(self):
        return self.gyro_data['x'], self.gyro_data['y'], self.gyro_data['z']

    def print_data(self):
        s = "ACC: (rows: " + str(len(self.acc_data.index)) + ")"
        s += "\nGYRO: (rows: " + str(len(self.gyro_data.index)) + ")"
        return s
