from Sample import Sample


class TestSample(Sample):
    def __init__(self, session_id, acc_data, gyro_data):
        super().__init__(acc_data, gyro_data)
        self.session_id = session_id

    def __str__(self):
        s = "Session " + self.session_id + ": (Number of sensors: " + str(self.acc_data.__len__()) + ")\nACC: "
        s += super().print_data()
        return s
