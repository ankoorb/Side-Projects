import os
import time
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import Config
from yolo.network import YOLOv3Layer
from utils.helper import get_random_augmented_data, preprocess_true_boxes


class Trainer(object):
    """
    Trainer to train MobileNetV2-YOLOv3 object detector.
    """

    def __init__(self, opt):
        self.opt = opt

        # Start training
        self.start()

    # Helpers
    @staticmethod
    def get_optimizer(opt, net):
        params = []
        for key, value in dict(net.named_parameters()).items():
            if value.requires_grad:
                if 'base' in key:
                    params += [{'params': [value], 'lr': opt.base_lr}]
                else:
                    params += [{'params': [value], 'lr': opt.lr}]

        # Initialize optimizer class: ADAM or SGD (w/wo nesterov)
        if opt.optimizer == 'adam':
            optimizer = optim.Adam(params=params, weight_decay=opt.weight_decay)
        else:
            optimizer = optim.SGD(params=params, momentum=0.9, weight_decay=opt.weight_decay,
                                  nesterov=(opt.optimizer == 'nesterov'))

        return optimizer

    def create_model(self):
        info = {}

        # YOLOv3 and its optimizer
        model = YOLOv3Layer(params=self.opt)
        optimizer = self.get_optimizer(self.opt, model)

        if self.opt.start_from:
            if self.opt.load_best_model == 1:
                model_path = os.path.join(self.opt.checkpoint_path, 'MobileNetV2_YoloV3.pth.tar')
            else:
                epoch = self.opt.start_from
                model_path = os.path.join(self.opt.checkpoint_path,
                                          'MobileNetV2_YoloV3_{}.pth.tar'.format(epoch))

                # Load checkpoint
                checkpoint = torch.load(model_path)
                info['epoch'] = checkpoint['epoch'] + 1
                info['best_loss'] = checkpoint['best_loss']

                # Load state dicts for YOLOv3 and its optimizer
                model.load_state_dict(checkpoint['yolo'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print('loaded weights from: ', model_path)

                # Reference: https://github.com/pytorch/pytorch/issues/2830
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(self.opt.device)

        return model, optimizer, info

    def save_checkpoint(self, epoch, best_val_loss, model, optimizer, best_flag=False):
        if not os.path.exists(self.opt.checkpoint_path):
            os.makedirs(self.opt.checkpoint_path)

        checkpoint_name = 'MobileNetV2_YoloV3_{}.pth.tar'.format(epoch)

        state = {
            'epoch': epoch,
            'best_loss': best_val_loss,
            'yolo': model.state_dict(),
            'optimizer': optimizer.state_dict()}

        torch.save(state, os.path.join(self.opt.checkpoint_path, checkpoint_name))

        if best_flag:
            best_checkpoint_name = 'MobileNetV2_YoloV3.pth.tar'
            torch.save(state, os.path.join(self.opt.checkpoint_path, best_checkpoint_name))

    def data_generator(self, annotation_lines):
        """
        Reference function: 
            https://github.com/qqwweee/keras-yolo3/blob/master/train.py
        """
        n = len(annotation_lines)
        i = 0
        while True:
            image_data = []
            bbox_data = []
            for b in range(self.opt.batch_size):
                if i == 0:
                    np.random.shuffle(annotation_lines)
                image, bboxes = get_random_augmented_data(annotation_lines[i],
                                                          self.opt.input_shape,
                                                          augment=self.opt.augment)

                image_data.append(image)
                bbox_data.append(bboxes)
                i = (i + 1) % n
            image_data = np.array(image_data)
            bbox_data = np.array(bbox_data)
            y_true = preprocess_true_boxes(bbox_data, self.opt.input_shape, self.opt.anchors,
                                           self.opt.n_classes)

            yield [image_data, *y_true]

    def data_generator_wrapper(self, annotation_lines):
        """
        Reference function: 
            https://github.com/qqwweee/keras-yolo3/blob/master/train.py
        """
        n = len(annotation_lines)
        if n == 0 or self.opt.batch_size <= 0:
            return None
        return self.data_generator(annotation_lines)

    def generate_data(self):
        """
        Generates train and val data based on validation split.
        """
        with open(self.opt.annotation_file) as f:
            annotation_lines = f.readlines()

        # Shuffle the annotation lines
        np.random.seed(15)
        np.random.shuffle(annotation_lines)

        # Compute splitting lengths
        n_val = int(len(annotation_lines) * self.opt.val_split)
        n_train = len(annotation_lines) - n_val

        # Train and val data generators
        train_gen = self.data_generator_wrapper(annotation_lines[:n_train])
        val_gen = self.data_generator_wrapper(annotation_lines[n_train:])

        return n_train, train_gen, n_val, val_gen

    def train(self, n_train, train_gen, epoch, model, optimizer):
        """
        Reference function:
            https://github.com/jiasenlu/YOLOv3.pytorch/blob/master/main.py
        """
        # Display string
        display = '>>> step: {}/{} (epoch: {}), loss: {:f}, lr: {:f}, time/batch {:.3f}'

        # Set gradient calculation to on
        torch.set_grad_enabled(mode=True)

        # Set model mode to train (default is train, but calling it explicitly)
        model.train()

        temp_losses = 0
        n_batches = int(n_train / self.opt.batch_size)

        start = time.time()
        for batch in range(n_batches):
            img, y13, y26, y52 = next(train_gen)
            # Using CUDA as default for now
            img = torch.from_numpy(img).float().to(self.opt.device)
            # PyTorch -> Channel first
            img = img.view(img.shape[0], img.shape[1], img.shape[2], img.shape[3]).permute(0, 3, 1, 2).contiguous()
            y13 = torch.from_numpy(y13).to(self.opt.device)
            y26 = torch.from_numpy(y26).to(self.opt.device)
            y52 = torch.from_numpy(y52).to(self.opt.device)

            # Forward pass and compute loss
            losses = model(img, y13, y26, y52)

            # Get total loss
            loss = losses[0].sum() / losses[0].numel()
            loss = loss.sum() / loss.numel()

            temp_losses = temp_losses + loss.item()

            # Backward pass and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % self.opt.display_interval == 0 and batch != 0:
                end = time.time()
                temp_losses = temp_losses / self.opt.display_interval
                print(display.format(batch, n_batches, epoch, temp_losses, optimizer.param_groups[-1]['lr'],
                                     (end - start) / self.opt.display_interval))

                # Reset temp losses and time
                temp_losses = 0
                start = time.time()

    def validate(self, n_val, val_gen, epoch, model):
        # Display string
        display = '>>> Evaluation loss (epoch: {}): {:.3f}'

        # Set gradient calculation to off
        torch.set_grad_enabled(mode=False)

        # Set model mode to eval
        model.eval()

        temp_losses = 0
        n_batches = int(n_val / self.opt.batch_size)

        for batch in range(n_batches):
            img, y13, y26, y52 = next(val_gen)
            # Using CUDA as default for now
            img = torch.from_numpy(img).float().to(self.opt.device)
            # PyTorch -> Channel first
            img = img.view(img.shape[0], img.shape[1], img.shape[2], img.shape[3]).permute(0, 3, 1, 2).contiguous()
            y13 = torch.from_numpy(y13).to(self.opt.device)
            y26 = torch.from_numpy(y26).to(self.opt.device)
            y52 = torch.from_numpy(y52).to(self.opt.device)

            # Forward pass and compute loss
            losses = model(img, y13, y26, y52)

            # Get total loss
            loss = losses[0].sum() / losses[0].numel()
            temp_losses = temp_losses + loss.item()

        # Loss
        temp_losses = temp_losses / n_batches

        print('=' * (len(display) + 10))
        print(display.format(epoch, temp_losses))
        print('=' * (len(display) + 10))

        return temp_losses

    def start(self):
        """
        Setup and start training
        """
        # Create model
        model, optimizer, info = self.create_model()

        if self.opt.use_gpu:
            model = model.to(self.opt.device)

        # Create data generators
        n_train, train_gen, n_val, val_gen = self.generate_data()

        # Scheduler to reduce learning rate when a metric has stopped improving
        scheduler = ReduceLROnPlateau(optimizer, patience=3, verbose=True)

        # Start training
        start_epoch = info.get('epoch', 0) if info.get('epoch', 0) else self.opt.start_epoch
        best_val_loss = info.get('best_loss', 1e6) if info.get('best_loss', 1e6) else self.opt.best_loss

        for epoch in range(start_epoch, self.opt.max_epochs):
            # Train
            self.train(n_train, train_gen, epoch, model, optimizer)

            # Evaluate on validation set
            val_loss = self.validate(n_val, val_gen, epoch, model)

            # Scheduler step
            scheduler.step(val_loss)

            # Save checkpoint
            best_flag = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)

            self.save_checkpoint(epoch, best_val_loss, model, optimizer, best_flag)


if __name__ == '__main__':

    # Create configurations
    config = Config()

    # Solves the problem when PyTorch uses default GPU:0 and you set GPU:1 or GPU:2
    torch.cuda.set_device(config.device_id)

    # Train the model
    Trainer(config)