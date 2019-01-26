class Config():
    """
    Configuration for training MobileNetV2-YOLOv3 model
    """

    def __init__(self):
        # MobileNetV2 parameters
        # ----------------------
        self.pretrained_weights = './MobileNetV2-Pretrained-Weights.pth.tar'
        # Conv and Inverted Residual Parameters: Table-2 (https://arxiv.org/pdf/1801.04381.pdf)
        self.t = [1, 1, 6, 6, 6, 6, 6, 6]  # t: expansion factor
        self.c = [32, 16, 24, 32, 64, 96, 160, 320]  # c: Output channels
        self.n = [1, 1, 2, 3, 4, 3, 3, 1]  # n: Number of times layer is repeated
        self.s = [2, 1, 2, 2, 2, 1, 2, 1]  # s: Stride
        # Width multiplier: Controls the width of the network
        self.alpha = 1.0

        # YOLOv3 parameters
        # -----------------
        self.n_classes = 5  # Udacity Self-driving car dataset
        self.class_map = {0: 'bike', 1: 'car', 2: 'pedestrian', 3: 'signal', 4: 'truck'}
        self.class_names = ['bike', 'car', 'pedestrian', 'signal', 'truck']
        self.final_channels = 3 * (5 + self.n_classes)
        self.input_shape = (416, 416)
        self.anchors = [[10, 13], [16, 30], [33, 23],
                        [30, 61], [62, 45], [59, 119],
                        [116, 90], [156, 198], [373, 326]]

        # Training parameters
        # -------------------
        self.use_gpu = True
        self.device_id = 0  # select 1 or 2 (0 is used for cpu)
        self.device = 'cuda:' + str(self.device_id) if self.device_id > 0 else 'cpu'
        self.optimizer = 'adam'  # 'adam' or 'sgd' or 'nesterov'
        self.weight_decay = 0
        self.max_epochs = 200
        self.base_lr = 0.0001  # Learning rate for MobileNetV2
        self.lr = 0.001  # Learning rate for the model
        self.best_loss = 1e6
        # NOTE: When loss=NaN --> Reduce learning rate.
        # SGD with LR: 0.001, 0.0001, 0.00001, 0.000001, 0.0000001 => NaN loss
        # ADAM with LR: 0.0000001 works

        # Dataset parameters
        # ------------------
        self.val_split = 0.1
        # Horizontal flip, scale and HSV.
        self.augment = True  # NOTE: Random augmentations might cause NaN becaue wh_loss=NaN
        self.batch_size = 24  # 24 works without memory error
        self.annotation_file = './annotations.csv'

        # Terminal display
        # ----------------
        self.display_interval = 10

        # Checkpoint config
        # -----------------
        self.start_epoch = 0
        self.start_from = None  # None if starting from scratch
        self.checkpoint_path = './checkpoints'
        self.load_best_model = False

