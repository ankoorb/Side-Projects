import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from config import Config
from utils.helpers import AverageMeter
from model.deeplab import DeepLabV3Plus
from dataset.cityscapes import Cityscapes
from utils.helpers import RandomHorizontalFlip, RandomScaleCrop, RandomGaussianBlur, Normalize, ToTensor, FixedScaleCrop, FixedResize


class Trainer(object):
    """
    Trainer for DeepLabV3+ with (modified) MobileNetV2 base
    """

    def __init__(self, opt):
        self.opt = opt

        # Start training
        self.start()

    # Helpers
    @staticmethod
    def get_optimizer(opt, net):
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, net.parameters()),
                                     lr=opt.lr)
        return optimizer

    @staticmethod
    def decay_learning_rate(opt, optimizer, epoch):
        """
        Adjust learning rate at each epoch as per policy stated in DeepLabV3 
        paper (page 5)
        """
        lr = opt.lr * (1 - float(epoch) / opt.num_epochs) ** opt.power
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Learning rate updated to %f' % (lr))

    @staticmethod
    def accuracy(scores, targets):
        """
        scores: PyTorch Tensor, output of DeepLabV3+ model, [M, 19, 1024, 2048]
        targets: PyTorch Tensor, labelIds of shape [M, 1024, 2048]
        """
        # Get indices maximum values
        preds = torch.argmax(scores, dim=1)  # size: [M, 1024, 2048]

        # Compute element wise equality and number of elements
        correct = torch.eq(preds, targets)  # targets size: [M, 1024, 2048]
        num_elements = correct.numel()

        # Total correct
        tot_correct = torch.sum(correct)

        return tot_correct.float().item() * 100.0 / num_elements

    def create_model(self):
        info = {}

        # DeepLabV3+ and its optimizer
        deeplab = DeepLabV3Plus(self.opt)

        optimizer = self.get_optimizer(self.opt, deeplab)

        if self.opt.start_from:
            if self.opt.load_best_model == 1:
                model_path = os.path.join(self.opt.checkpoint_path, 'MobileNetV2_DeepLabV3Plus.pth.tar')
            else:
                epoch = self.opt.start_from
                model_path = os.path.join(self.opt.checkpoint_path,
                                          'MobileNetV2_DeepLabV3Plus_{}.pth.tar'.format(epoch))

            # Load checkpoint
            checkpoint = torch.load(model_path)
            info['epoch'] = checkpoint['epoch'] + 1
            info['best_accuracy'] = checkpoint['accuracy']

            # Load state dicts for encoder, decoder, and their optimizers
            deeplab.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            # Reference: https://github.com/pytorch/pytorch/issues/2830
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.opt.device)

        return deeplab, optimizer, info

    def train(self, train_loader, model, loss_fn, optimizer, epoch):
        # Display string
        display = """>>> step: {}/{} (epoch: {}), loss: {ls.val:f}, avg loss: {ls.avg:f}, 
        time/batch: {proc_time.val:.3f}, avg time/batch: {proc_time.avg:.3f}, acc: {acc.val:f}"""

        # Training mode
        model.train()

        # Stats
        batch_time = AverageMeter()  # Forward propagation + back propatation time
        losses = AverageMeter()  # Loss
        accs = AverageMeter()  # Accuracy

        start = time.time()

        # Training loop for one epoch
        for i, batch in enumerate(train_loader):

            imgs = batch['image']
            masks = batch['label']

            batch_size = imgs.size(0)

            # Using CUDA as default
            imgs = imgs.to(self.opt.device)
            masks = masks.long().to(self.opt.device)

            # Forward pass
            logits = model(imgs)

            # Compute loss
            loss = loss_fn(logits.to(self.opt.device), masks)

            # Compute accuracy
            acc = self.accuracy(logits.cpu(), masks.cpu())

            # Backward propagation and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            losses.update(loss.item(), batch_size)
            accs.update(acc, batch_size)
            batch_time.update(time.time() - start)
            start = time.time()  # Restart timer

            if i % self.opt.display_interval == 0 and i != 0:
                print(display.format(i, len(train_loader), epoch, ls=losses,
                                     proc_time=batch_time, acc=accs))

        # Average Accuracy
        show = '>>> epoch: {}, avg training loss: {ls.avg:f}, avg training accuracy: {acc.avg:f}'
        print(show.format(epoch, ls=losses, acc=accs))

    def validate(self, val_loader, model, loss_fn, epoch):
        # Display string
        display = """>>> step: {}/{} (epoch: {}), loss: {ls.val:f}, avg loss: {ls.avg:f}, 
        time/batch: {proc_time.val:.3f}, avg time/batch: {proc_time.avg:.3f}, acc: {acc.val:f}"""

        # Stats
        batch_time = AverageMeter()  # Forward propagation
        losses = AverageMeter()  # Loss
        accs = AverageMeter()  # Accuracy

        # Evaluation mode
        model.eval()

        start = time.time()

        # Validation loop for one epoch
        for i, batch in enumerate(val_loader):

            imgs = batch['image']
            masks = batch['label']

            batch_size = imgs.size(0)

            # Using CUDA as default
            imgs = imgs.to(self.opt.device)
            masks = masks.long().to(self.opt.device)

            # Forward pass
            logits = model(imgs)

            # Compute loss
            loss = loss_fn(logits.to(self.opt.device), masks)

            # Compute accuracy
            acc = self.accuracy(logits.cpu(), masks.cpu())

            # Update metrics
            losses.update(loss.item(), batch_size)
            accs.update(acc, batch_size)
            batch_time.update(time.time() - start)

            start = time.time()  # Restart timer

            if i % self.opt.display_interval == 0 and i != 0:
                print(display.format(i, len(val_loader), epoch, ls=losses,
                                     proc_time=batch_time, acc=accs))

        # Average Accuracy
        show = '>>> epoch: {}, avg validation loss: {ls.avg:f}, avg validation accuracy: {acc.avg:f}'
        print(show.format(epoch, ls=losses, acc=accs))

        return accs.avg, losses.avg

    def test(self):
        """
        Test functionality seprately coded for the App.
        """
        raise NotImplementedError

    def save_checkpoint(self, epoch, best_acc, val_avg_loss, model, optimizer, best_flag=False):
        if not os.path.exists(self.opt.checkpoint_path):
            os.makedirs(self.opt.checkpoint_path)

        checkpoint_name = 'MobileNetV2_DeepLabV3Plus_{}.pth.tar'.format(epoch)

        state = {
            'epoch': epoch,
            'accuracy': best_acc,  # Best average accuracy on validation data so far
            'loss': val_avg_loss,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()}

        torch.save(state, os.path.join(self.opt.checkpoint_path, checkpoint_name))

        if best_flag:
            best_checkpoint_name = 'MobileNetV2_DeepLabV3Plus.pth.tar'
            torch.save(state, os.path.join(self.opt.checkpoint_path, best_checkpoint_name))

    def start(self):

        # Create model
        deeplab, optimizer, info = self.create_model()

        # Loss Function
        loss_function = nn.CrossEntropyLoss(ignore_index=255).to(self.opt.device)

        if self.opt.use_gpu:
            deeplab = deeplab.to(self.opt.device)
            loss_function = loss_function.to(self.opt.device)

        # Data Transforms: train, val and test
        train_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomScaleCrop(base_size=self.opt.base_size, crop_size=self.opt.image_size),
            RandomGaussianBlur(),
            Normalize(mean=self.opt.img_mean, std=self.opt.img_std),
            ToTensor()
        ])

        val_transforms = transforms.Compose([
            FixedScaleCrop(crop_size=self.opt.image_size),
            Normalize(mean=self.opt.img_mean, std=self.opt.img_std),
            ToTensor()
        ])

        test_transforms = transforms.Compose([
            FixedResize(size=self.opt.image_size),
            Normalize(mean=self.opt.img_mean, std=self.opt.img_std),
            ToTensor()
        ])

        # Data loaders
        train_data = Cityscapes(self.opt.dataset_root, split='train', mode='gtFine', target_type='semantic',
                                transform=train_transforms)
        train_loader = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True)

        val_data = Cityscapes(self.opt.dataset_root, split='val', mode='gtFine', target_type='semantic',
                              transform=val_transforms)
        val_loader = DataLoader(val_data, batch_size=self.opt.batch_size, shuffle=True)

        # Start training: Train for epochs
        start_epoch = info.get('epoch', 0) if info.get('epoch', 0) else self.opt.start_epoch
        best_acc = info.get('best_accuracy', 0) if info.get('best_accuracy', 0) else self.opt.best_acc

        # Train for epochs
        for epoch in range(start_epoch, self.opt.num_epochs):
            # One epoch training
            self.train(train_loader=train_loader, model=deeplab, loss_fn=loss_function, optimizer=optimizer,
                       epoch=epoch)

            # One epoch validation
            val_acc, val_loss = self.validate(val_loader=val_loader, model=deeplab, loss_fn=loss_function,
                                              epoch=epoch)

            # Decay learning rate after each epoch as per policy
            self.decay_learning_rate(self.opt, optimizer, epoch)

            # Check for best accuracy
            best_flag = val_acc > best_acc
            best_acc = max(val_acc, best_acc)

            # Save checkpoint
            self.save_checkpoint(epoch, best_acc, val_loss, deeplab, optimizer, best_flag=best_flag)


if __name__ == '__main__':

    # Create configurations
    config = Config()

    # Train the model
    Trainer(config)