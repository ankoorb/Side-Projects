class Config(object):
    def __init__(self):
        # Encoder parameters
        # ------------------
        self.cnn_weight_file = './mobilenet_v2.pth.tar'
        self.feature_size = 14
        self.tune_layer = 15
        self.finetune = True

        # Normalizing constants
        # ---------------------
        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]

        # Decoder parameters
        # ------------------
        self.encoder_size = 1280  # MobileNetV2 output channels 
        self.decoder_size = 512  # LSTM output size (hidden state vector size)
        self.attention_size = 512  # Size of MLP used to compute attention scores
        self.embedding_size = 256  # Word embedding size
        self.dropout_prob = 0.5

        # Training config
        # ---------------
        self.use_gpu = True
        self.batch_size = 64
        self.start_epoch = 0
        self.num_epochs = 200
        self.encoder_lr = 0.0001  # Learning rate for encoder
        self.decoder_lr = 0.001  # Learning rate for decoder
        self.lr_multiplier = 0.9  # Learning rate decay
        self.alpha_c = 1.0
        self.clip_value = 5.0
        self.k = 5  # Top k accuracy
        self.device_id = 0  # select 1 or 2 for gpu
        self.device = 'cuda:' + str(self.device_id) if self.device_id > 0 else 'cpu'
        self.best_bleu = 0

        # Word to index mapping
        # ---------------------
        self.word2idx_file = './WORD2IDX_COCO.json'

        # Training data
        # -------------
        self.train_hdf5 = './TRAIN_IMAGES_COCO.hdf5'
        self.train_captions = './TRAIN_CAPTIONS_COCO.json'

        # Validation data
        self.val_hdf5 = './VAL_IMAGES_COCO.hdf5'
        self.val_captions = './VAL_CAPTIONS_COCO.json'

        # Terminal display
        # ----------------
        self.display_interval = 10

        # Checkpoint config
        # -----------------
        self.start_epoch = 0
        self.start_from = 119  # Use None if training from epoch 0
        self.checkpoint_path = './checkpoints'
        self.load_best_model = False

