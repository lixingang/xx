class Opt():
    def __init__(self):
        # basic setting
        self.project="test"
        self.num_epochs=100
        self.num_workers=8
        self.batch_size=20
        self.device = "cuda:2"
        self.learning_rate = 0.001
        self.log_dir = "Logs"
        # self.data_dir = "/data/lxg/workplace/dhs_tfrecords/npy"
        self.data_dir = "/home/lxg/dhs_npy"
        # self.ratio = [0.7,0.15,0.15]

        # cross-validation
        self.train_list = []
        self.valid_list = []
        self.test_list = []

        # train
        self.images_dir = {
            "train":"",
            "valid":"",
            "test":"",
        }
        self.masks_dir = {
            "train": "",
            "valid": "",
            "test": "",
        }

        # test
        # self.restore_model = None
        self.restore_model = 'Logs/best_model.pkl'

