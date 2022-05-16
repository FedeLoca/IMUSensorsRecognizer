from Sample import Sample


class TrainingSample(Sample):
    def __init__(self, action_name, acc_data, gyro_data):
        super().__init__(acc_data, gyro_data)
        self.action_name = action_name

    def __str__(self):
        s = self.action_name + ": (Number of sensors: " + str(self.acc_data.__len__()) + ")\n"
        s += super().print_data()
        return s
