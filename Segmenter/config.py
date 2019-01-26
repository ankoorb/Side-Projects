from utils.helpers import _make_divisible

class Config():
    """
    Configuration for training DeepLabV3+
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
        self.r = [1, 1, 1, 1, 1, 1, 2, 2]  # r: Dilation (added to take care of dilation)
        # Width multiplier: Controls the width of the network
        self.alpha = 1  # Use multiples of 0.25, min=0.25, max=1.0

        # Data Augmentations
        # ------------------
        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]
        self.base_size = 640  # Scale
        self.image_size = 512  # Crop size

        # ASPP Parameters
        # ---------------
        self.aspp_inch = int(self.alpha * self.c[-1])  # Width multiplier * 320
        self.aspp_outch = int(self.alpha * 256)  # Width multiplier * 256

        # Decoder Parameters
        # ------------------
        self.n_classes = 19
        self.low_level_inCh = _make_divisible(self.alpha * self.c[2], 8)  # Width multiplier * 32
        self.low_level_outCh = int(2 * self.low_level_inCh)  # 2 * low level features channels
        self.in_channels = _make_divisible(self.alpha * 256, 8)  # Width multiplier * 256
        self.out_channels = _make_divisible(self.alpha * 256, 8)  # Width multiplier * 256

        # Data
        # ----
        self.dataset_root = './cityscapes'

        # Training config
        # ---------------
        self.use_gpu = True
        self.batch_size = 16
        self.start_epoch = 0
        self.num_epochs = 250
        self.power = 0.9  # Learning rate policy multiplier
        self.lr = 0.0001  # Learning rate
        self.lr_multiplier = 0.9  # Learning rate decay
        self.device_id = 0 # Use greater than 0 for GPU
        self.device = 'cuda:' + str(self.device_id) if self.device_id > 0 else 'cpu'

        # Terminal display
        # ----------------
        self.display_interval = 100

        # Checkpoint config
        # -----------------
        self.best_acc = 0
        self.start_epoch = 0
        self.start_from = None  # Use None if training from epoch 0
        self.checkpoint_path = './checkpoints'
        self.load_best_model = False

