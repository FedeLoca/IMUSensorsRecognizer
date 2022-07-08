from Sample import Sample


class TrainingSample(Sample):
    def __init__(self, action_name, acc_data, gyro_data, sample_num):
        super().__init__(acc_data, gyro_data)
        self.action_name = action_name
        self.sample_num = sample_num

    def __str__(self):
        s = self.action_name + " " + self.sample_num + "\n"
        s += super().print_data()
        return s
