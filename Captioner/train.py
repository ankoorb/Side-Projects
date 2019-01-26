import os
import time
import json
from nltk.translate.bleu_score import corpus_bleu

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence


from config import Config
from data.coco import COCODataset
from model.encoder import EncoderCNN
from utils.helper import AverageMeter
from model.decoder import DecoderAttentionRNN


class Trainer(object):
    def __init__(self, opt):
        self.opt = opt
        self.word2idx = self.read_json(self.opt.word2idx_file)
        self.vocab_size = len(self.word2idx)

        # Start training
        self.start()

    # Helpers
    def read_json(self, file):
        with open(file, 'r') as f:
            data = json.load(f)
        return data

    @staticmethod
    def get_optimizer(opt, net, coder='decoder'):
        """
        Adam optimizer
        """
        if coder == 'decoder':
            lr = opt.decoder_lr
        else:
            lr = opt.encoder_lr

        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

        return optimizer

    @staticmethod
    def decay_learning_rate(optimizer, lr_multiplier):
        """
        Decays learning rate by a multiplier.

        optimizer: PyTorch optim object
        lr_multiplier: float value in range (0, 1)
        """
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_multiplier
        print('Learning rate has been reduced!')

    @staticmethod
    def clip_gradient(optimizer, clip_value):
        """
        Clip gradients computed during back propagation (to solve exploding
        gradients)
        """
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-clip_value, clip_value)

    @staticmethod
    def top_k_accuracy(scores, targets, k):
        """
        scores and targets are PyTorch tensors, k is int.
        """
        num_elements = targets.numel()

        # Get indices of the k largest elements
        _, topk_idx = scores.data.topk(k, dim=1)  # size: [num_elements, k]

        # Compute element wise equality
        correct = torch.eq(topk_idx, targets.view(-1, 1).cpu())  # targets size: [num_elements]

        # Total correct
        tot_correct = torch.sum(correct)

        return tot_correct.float().item() * 100.0 / num_elements

    @staticmethod
    def prepare_bleu_data(captions, sorted_idx, scores, decode_lengths, word2idx):
        temp_references = []
        temp_hypotheses = []

        # Prepare y_true i.e. references for BLEU
        captions = captions[sorted_idx]  # Sort captions based on sorted indices from decoder
        remove_idx = [word2idx['<START>'], word2idx['<PAD>']]
        for c in range(captions.size(0)):
            img_caps = captions[c].tolist()
            # Remove indices corresponding to <START> and <PAD>
            img_caps = [[ix for ix in cap if ix not in remove_idx] for cap in img_caps]
            temp_references.append(img_caps)

        # Prepare y_pred i.e. hypotheses for BLEU
        scores_clone = scores.clone()
        _, preds = torch.max(scores_clone, dim=2)  # Get indixes of words with max score
        preds = preds.tolist()  # Convert PyTorch tensor to list
        for i, pred in enumerate(preds):
            img_hyp = preds[i][:decode_lengths[i]]
            temp_hypotheses.append(img_hyp)

        return temp_references, temp_hypotheses

    def create_model(self):
        info = {}

        # Encoder and its optimizer
        encoder = EncoderCNN(weight_file=self.opt.cnn_weight_file,
                             tune_layer=self.opt.tune_layer,
                             finetune=self.opt.finetune)

        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=self.opt.encoder_lr) if self.opt.finetune else None

        encoder_optimizer = self.get_optimizer(self.opt, encoder, coder='encoder') if self.opt.finetune else None

        # Decoder and its optimizer
        decoder = DecoderAttentionRNN(encoder_size=self.opt.encoder_size,
                                      decoder_size=self.opt.decoder_size,
                                      attention_size=self.opt.attention_size,
                                      embedding_size=self.opt.embedding_size,
                                      vocab_size=self.vocab_size,
                                      dropout_prob=self.opt.dropout_prob)

        decoder_optimizer = self.get_optimizer(self.opt, decoder)

        if self.opt.start_from:
            if self.opt.load_best_model == 1:
                model_path = os.path.join(self.opt.checkpoint_path, 'MobileNetV2_Show_Attend_Tell.pth.tar')
            else:
                epoch = self.opt.start_from
                model_path = os.path.join(self.opt.checkpoint_path,
                                          'MobileNetV2_Show_Attend_Tell_{}.pth.tar'.format(epoch))

            # Load checkpoint
            checkpoint = torch.load(model_path)
            info['epoch'] = checkpoint['epoch'] + 1
            info['epochs_since_improvement'] = checkpoint['epochs_since_improvement']
            info['best_bleu'] = checkpoint['best_bleu']

            # Load state dicts for encoder, decoder, and their optimizers
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
            decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])

            # Reference: https://github.com/pytorch/pytorch/issues/2830
            for state in decoder_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.opt.device)

            if encoder_optimizer and checkpoint['encoder_optimizer']:
                encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])

                # Reference: https://github.com/pytorch/pytorch/issues/2830
                for state in encoder_optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(self.opt.device)

        return encoder, decoder, encoder_optimizer, decoder_optimizer, info

    def save_checkpoint(self, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                        best_bleu, best_flag=False):
        if not os.path.exists(self.opt.checkpoint_path):
            os.makedirs(self.opt.checkpoint_path)

        checkpoint_name = 'MobileNetV2_Show_Attend_Tell_{}.pth.tar'.format(epoch)

        state = {
            'epoch': epoch,
            'epochs_since_improvement': epochs_since_improvement,
            'best_bleu': best_bleu,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict() if self.opt.finetune else None,
            'decoder_optimizer': decoder_optimizer.state_dict()}

        torch.save(state, os.path.join(self.opt.checkpoint_path, checkpoint_name))

        if best_flag:
            best_checkpoint_name = 'MobileNetV2_Show_Attend_Tell.pth.tar'
            torch.save(state, os.path.join(self.opt.checkpoint_path, best_checkpoint_name))

    def train(self, train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
        # Display string
        display = """>>> step: {}/{} (epoch: {}), loss: {ls.val:f}, avg loss: {ls.avg:f}, 
        time/batch: {proc_time.val:.3f}, avg time/batch: {proc_time.avg:.3f}, top-5 acc: {acc.val:f}, 
        avg top-5 acc: {acc.avg:f}"""

        # Training mode
        encoder.train()
        decoder.train()

        # Stats
        batch_time = AverageMeter()  # Forward propagation + back propatation time
        losses = AverageMeter()  # Loss
        top5_accs = AverageMeter()  # Top-5 accuracy

        start = time.time()

        # Training loop for one epoch
        for i, (imgs, caps, cap_lengths) in enumerate(train_loader):

            # Using CUDA as default
            imgs = imgs.to(self.opt.device)
            encoded_caps = caps.to(self.opt.device)
            cap_lengths = cap_lengths.to(self.opt.device)

            # Forward pass
            encoder_out = encoder(imgs)
            pred_scores, sorted_caps, decode_lengths, alphas, sorted_idx = decoder(encoder_out,
                                                                                   encoded_caps,
                                                                                   cap_lengths)

            # Select all words after <START> till <END>
            target_caps = sorted_caps[:, 1:]

            # Pack padded sequences. Before computing Cross Entropy Loss (Log Softmax and Negative Log
            # Likelihood Loss) we do not want to take into account padded items in the predicted scores
            scores, _ = pack_padded_sequence(pred_scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(target_caps, decode_lengths, batch_first=True)

            # Compute loss
            loss = criterion(scores.to(self.opt.device), targets.to(self.opt.device))

            # Add doubly stochastic attention regularization
            loss += (self.opt.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()).to(self.opt.device)

            # Backward propagation
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()

            loss.backward()

            # Clip gradients
            if self.opt.clip_value is not None:
                self.clip_gradient(decoder_optimizer, self.opt.clip_value)
                if encoder_optimizer is not None:
                    self.clip_gradient(encoder_optimizer, self.opt.clip_value)

            # Update weights
            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            # Compute top accuracy for top k words
            top5_acc = self.top_k_accuracy(scores.data, targets.data, k=self.opt.k)

            # Update metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5_accs.update(top5_acc, sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()  # Restart timer

            if i % self.opt.display_interval == 0 and i != 0:
                print(display.format(i, len(train_loader), epoch, ls=losses,
                                     proc_time=batch_time, acc=top5_accs))

    def validate(self, val_loader, encoder, decoder, criterion, epoch):
        # Display string
        display = """>>> step: {}/{} (epoch: {}), loss: {ls.val:f}, avg loss: {ls.avg:f}, 
        time/batch: {proc_time.val:.3f}, avg time/batch: {proc_time.avg:.3f}, top-5 acc: {acc.val:f}, 
        avg top-5 acc: {acc.avg:f}"""

        # Stats
        batch_time = AverageMeter()  # Forward propagation
        losses = AverageMeter()  # Loss
        top5_accs = AverageMeter()  # Top 5 accuracy

        # Evaluation mode
        encoder.eval()
        decoder.eval()

        # Caches for BLEU score computation
        references = []  # y_true
        hypotheses = []  # y_pres

        start = time.time()

        # Training loop for one epoch
        for i, (imgs, caps, cap_lengths, captions) in enumerate(val_loader):

            # Using CUDA as default
            imgs = imgs.to(self.opt.device)
            encoded_caps = caps.to(self.opt.device)
            cap_lengths = cap_lengths.to(self.opt.device)

            # Forward pass
            encoder_out = encoder(imgs)
            pred_scores, sorted_caps, decode_lengths, alphas, sorted_idx = decoder(encoder_out,
                                                                                   encoded_caps,
                                                                                   cap_lengths)

            pred_scores_copy = pred_scores.clone()

            # Select all words after <START> till <END>
            target_caps = sorted_caps[:, 1:]

            # Pack padded sequences. Before computing Cross Entropy Loss (Log Softmax and Negative Log
            # Likelihood Loss) we do not want to take into account padded items in the predicted scores
            scores, _ = pack_padded_sequence(pred_scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(target_caps, decode_lengths, batch_first=True)

            # Compute loss
            loss = criterion(scores.to(self.opt.device), targets.to(self.opt.device))

            # Add doubly stochastic attention regularization
            loss += (self.opt.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()).to(self.opt.device)

            # Compute top accuracy for top k words
            top5_acc = self.top_k_accuracy(scores.data, targets.data, k=self.opt.k)

            # Update metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5_accs.update(top5_acc, sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()  # Restart timer

            if i % self.opt.display_interval == 0 and i != 0:
                print(display.format(i, len(val_loader), epoch, ls=losses, proc_time=batch_time,
                                     acc=top5_accs))

            # Prepare data to compute BLEU score
            temp_refs, temp_hyps = self.prepare_bleu_data(captions, sorted_idx, pred_scores, decode_lengths,
                                                          self.word2idx)
            assert len(temp_refs) == len(temp_hyps)

            # Exted the caches
            references.extend(temp_refs)
            hypotheses.extend(temp_hyps)

        # Compute BLEU score
        bleu = corpus_bleu(references, hypotheses, weights=(0.5, 0.5))
        show = '>>> epoch: {}, avg loss: {ls.avg:f}, avg top-5 acc: {acc.avg:f}, bleu: {bleu}'
        print(show.format(epoch, ls=losses, acc=top5_accs, bleu=bleu))

        return bleu

    def start(self):
        # Create model
        encoder, decoder, encoder_optimizer, decoder_optimizer, info = self.create_model()

        # Loss criterion
        criterion = nn.CrossEntropyLoss().to(self.opt.device)

        if self.opt.use_gpu:
            decoder = decoder.to(self.opt.device)
            encoder = encoder.to(self.opt.device)
            criterion = criterion.to(self.opt.device)

        # Normalize image
        normalize = transforms.Normalize(mean=self.opt.img_mean, std=self.opt.img_std)

        # Data loaders
        train_data = COCODataset(self.opt, split='TRAIN', transform=transforms.Compose([normalize]))
        train_loader = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True)
        val_data = COCODataset(self.opt, split='VAL', transform=transforms.Compose([normalize]))
        val_loader = DataLoader(val_data, batch_size=self.opt.batch_size, shuffle=True)

        # Start training: Train for epochs
        epochs_since_improvement = info.get('epochs_since_improvement', 0)
        start_epoch = info.get('epoch', 0) if info.get('epoch', 0) else self.opt.start_epoch
        best_bleu = info.get('best_bleu', 0) if info.get('best_bleu', 0) else self.opt.best_bleu

        # Train for epochs
        for epoch in range(start_epoch, self.opt.num_epochs):

            if epochs_since_improvement > 0 and epochs_since_improvement % 10 == 0:
                self.decay_learning_rate(decoder_optimizer, self.opt.lr_multiplier)
                if self.opt.finetune:
                    self.decay_learning_rate(encoder_optimizer, self.opt.lr_multiplier)

            # One epoch training
            self.train(train_loader=train_loader, encoder=encoder, decoder=decoder,
                       encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer,
                       criterion=criterion, epoch=epoch)

            # One epoch validation
            val_bleu = self.validate(val_loader=val_loader, encoder=encoder, decoder=decoder,
                                     criterion=criterion, epoch=epoch)

            # Check for best bleu score
            best_flag = val_bleu > best_bleu
            best_bleu = max(val_bleu, best_bleu)
            if not best_flag:
                epochs_since_improvement += 1
                print('Number of epochs since last improvement: ', epochs_since_improvement)
            else:
                epochs_since_improvement = 0

            # Save checkpoint
            self.save_checkpoint(epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                                 decoder_optimizer, best_bleu, best_flag=best_flag)



if __name__ == '__main__':

    # Create configurations
    config = Config()

    # Train the model
    Trainer(config)

