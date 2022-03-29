class Opt():
    def __init__(self):
        self.mode='seg'
        self.project="test"
        self.num_epochs=30
        self.num_workers=3
        self.batch_size=4
        self.classes=['false','true']
        self.device = "cuda:2"
        self.learning_rate = 0.0001
        self.log_dir = "logs"


class TrainOpt(Opt):
    def __init__(self):
        super(TrainOpt, self).__init__()
        # self.mode = 'cls'

        if self.mode == 'seg':
            self.segmentation()
        elif self.mode == 'cls':
            self.classification()


    def segmentation(self):
        self.img_size=200

    def classification(self):
        pass


class PredictOpt(Opt):
    def __init__(self):
        super(PredictOpt, self).__init__()
        self.restore_model = 'logs/unetplus_16.pkl'
        # self.mode = 'cls'
        if self.mode == 'seg':
            self.segmentation()
        elif self.mode == 'cls':
            self.classification()

    def segmentation(self):
        self.is_mask = False



    def classification(self):
        pass
