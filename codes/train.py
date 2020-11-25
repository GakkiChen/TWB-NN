import os
import sys
import math
import time
import logging
import sys
import argparse
import torch
import glob
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from my_model import resnet20
import utils
from utils import adjust_learning_rate
import numpy as np
from random import shuffle
from utils import save_checkpoint

parser = argparse.ArgumentParser("CIFAR")
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=5e-3, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=10, help='gradient clipping')
parser.add_argument('--trimming', action='store_true', default=False, help='resume training')
parser.add_argument('--ResNet20_prunedArch', action='store_true', default=False, help='use pruned architecture')
parser.add_argument('--pretrain', action='store_true', default=False, help='use pretrained model')
parser.add_argument('--resume_dir', type=str, default='./weights/pretrain.pth.tar', help='save weights directory')
parser.add_argument('--load_epoch', type=int, default=30, help='random seed')
parser.add_argument('--weights_dir', type=str, default='./weights/', help='save weights directory')
parser.add_argument('--learning_step', type=list, default=[60,90,200], help='learning rate steps')


args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
if not os.path.exists(args.save):
    os.makedirs(args.save)
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


# Image Preprocessing 
    train_transform = transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,])

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,])


    num_epochs = args.epochs
    batch_size = args.batch_size

    train_dataset = datasets.CIFAR10(root='../Data', train=True, download=True,transform=train_transform)
    test_dataset = datasets.CIFAR10(root='../Data', train=False, download=True,transform=test_transform)    

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True, num_workers=10, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=256, 
                                          shuffle=False, num_workers=10, pin_memory=True)

    num_train = train_dataset.__len__()
    n_train_batches = math.floor(num_train / batch_size)

    criterion = nn.CrossEntropyLoss().cuda()
    bitW = 1
    bitA = 1

    if args.trimming:
        trim = True
    else:
        trim = False

    if args.ResNet20_prunedArch:
        pruned = True
    else:
        pruned = False

    if args.pretrain:
        pretrain = True
    else:
        pretrain = False

    model = resnet20(bitW, bitA, pruned, pretrained=pretrain)
    model = utils.dataparallel(model, 2)

    print("Compilation complete, starting training...")   

    for name, param in model.module.named_parameters():
        if 'masks' in name:
            if F.hardtanh((param+0.5),min_val=0,max_val=1)==0:
                print("{} is pruned from the architecture".format(name))
 

    test_record = []
    train_record = []
    learning_rate = args.learning_rate
    epoch = 0
    step_idx = 0
    best_top1 = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    while epoch < num_epochs:

        logging.info('epoch %d lr %e', epoch, learning_rate)
        epoch = epoch + 1
    # resume training    
        if (trim) and (epoch == 1): 
            print("==========> trimming starts <=========")
            checkpoint = torch.load(args.resume_dir)
            checkpoint_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']
            model_dict = model.state_dict()
            model_keys = model_dict.keys()
            for name, param in checkpoint_dict.items():
                if name in model_keys:
                    model_dict[name] = param
            model.load_state_dict(model_dict)
            optimizer = torch.optim.Adam([{"params": [param for name, param in model.module.named_parameters() if 'masks' not in name],'lr':5e-5},\
            {"params": [param for name, param in model.module.named_parameters() if 'masks' in name and F.hardtanh((param+0.5),min_val=0,max_val=1)!=0],'lr':5e-4},\
            {"params": [param for name, param in model.module.named_parameters() if 'masks' in name and F.hardtanh((param+0.5),min_val=0,max_val=1)==0],'lr':0}],weight_decay=args.weight_decay)
            test_record = list(
                np.load(args.weights_dir + 'test_record.npy'))
            train_record = list(
                np.load(args.weights_dir + 'train_record.npy'))

    # training
        train_acc_top1, train_acc_top5, train_obj = train(train_loader, model, criterion, optimizer, learning_rate, trim)
        logging.info('train_acc %f', train_acc_top1)
        train_record.append([train_acc_top1, train_acc_top5])
        np.save(args.weights_dir + 'train_record.npy', train_record)   

    # test
        test_acc_top1, test_acc_top5, test_obj = infer(test_loader, model, criterion)
        is_best = test_acc_top1 > best_top1
        if is_best:
            best_top1 = test_acc_top1

        logging.info('test_acc %f', test_acc_top1)
        test_record.append([test_acc_top1, test_acc_top5])
        np.save(args.weights_dir + 'test_record.npy', test_record)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'best_top1': best_top1,
                'learning_rate': learning_rate,
                }, args, is_best)
        step_idx, learning_rate = utils.adjust_learning_rate(args, epoch, step_idx,
                                           learning_rate)

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate


def train(train_queue, model, criterion, optimizer, lr, trim):

    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    model.train()

    for step, (input, target) in enumerate(train_queue):
        block_fac = 5

        n = input.size(0)
        input = input.cuda()
        target = target.cuda()


        logits = model(input)
        CE_loss = criterion(logits, target)
        
        block_loss = 0
        block_num = 0
        filter_loss = 0
        filter_num = 0  

        for name, param in model.module.named_parameters():
            if 'masks' in name:
                block_loss += F.hardtanh((param+0.5),min_val=0,max_val=1)
                block_num += 1

        block_loss /= float(block_num)
        
        if trim:
            #sparsity loss
            loss = CE_loss + block_fac*block_loss
        else:
            loss = CE_loss
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            
            if trim:
                #checking for pruning ratio
                block_count = 0
                pruned_block_count = 0
                for name, param in model.module.named_parameters():
                    if 'masks' in name:
                        block_count += 1
                        if F.hardtanh((param+0.5),min_val=0,max_val=1)==0:
                            pruned_block_count += 1
 
                block_pruned_ratio=pruned_block_count/float(block_count)
                print('block prune ratio: ', block_pruned_ratio)
            
                if block_pruned_ratio >= 0.2:
                    optimizer = torch.optim.Adam([{"params": [param for name, param in model.module.named_parameters() if 'scales' not in name],'lr':5e-5},\
                    {"params": [param for name, param in model.module.named_parameters() if 'scales' in name and F.hardtanh((param+0.5),min_val=0,max_val=1)==0],'lr':0},\
                    {"params": [param for name, param in model.module.named_parameters() if 'scales' in name and F.hardtanh((param+0.5),min_val=0,max_val=1)!=0],'lr':0}],weight_decay=args.weight_decay)
                    block_fac = 0
            else:
                pass
            
    return top1.avg, top5.avg, objs.avg



def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()

            logits = model(input)
            loss = criterion(logits, target)
 
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    return top1.avg, top5.avg, objs.avg
 

if __name__ == '__main__':
    utils.create_folder(args)
    main()

