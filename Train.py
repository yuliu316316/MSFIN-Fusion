import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
import glob
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from Networks.Mymodules_out4_basicBloV5_NoCA import MODEL as net
# from CEL import CEL
import pytorch_iou
from losses import structure_weighted_binary_cross_entropy_with_logits, structure_loss, wbce_loss

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')        # CUDA:1
use_gpu = torch.cuda.is_available()
# use_gpu = False
if use_gpu:
    print('GPU Mode Acitavted')
else:
    print('CPU Mode Acitavted')


def parse_args():                 # 参数解析器
    parser = argparse.ArgumentParser()
    # 增加属性
    parser.add_argument('--name', default='MSNetout', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float)    # 5e-4
    parser.add_argument('--weight', default=[1, 1, 1, 1], type=float)
    # parser.add_argument('--milestones', default='25,50', type=str)
    # MultiStepLR三段式lr，epoch进入milestones范围内即乘以gamma，离开milestones范围之后再乘以gamma
    # parser.add_argument('--gamma', default=0.1, type=float)            # MultiStepLR gamma
    parser.add_argument('--gamma', default=0.9, type=float)     # ExponentialLR gamma
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)             # 5e-4
    args = parser.parse_args()    # 属性给与args实例：add_argument 返回到 args 子类实例

    return args


def rotate(image, s):
    if s == 0:
        image = image
    if s == 1:
        HF = transforms.RandomHorizontalFlip(p=1)   # 随机水平翻转
        image = HF(image)
    if s == 2:
        VF = transforms.RandomVerticalFlip(p=1)     # 随机垂直翻转
        image = VF(image)
    return image


def color2gray(image, s):
    if s == 0:
        image = image
    if s ==1:
        l = image.convert('L')
        n = np.array(l)  # 转化成numpy数组
        image = np.expand_dims(n, axis=2)
        image = np.concatenate((image, image, image), axis=-1)  # axis=-1就是最后一个通道
        image = Image.fromarray(image).convert('RGB')
    return image


class GetDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):
        img0_tuple = self.imageFolderDataset[index]
        img1_tuple = img0_tuple[0:-20] + 'B' + img0_tuple[-19:]
        # VOCtrainval_128/VOCdevkit/
        truth_tuple = '/VOC2012_240/decisionmap' + img0_tuple[-19:-7] + '.png'
        truth_fuse = '/VOC2012_240/groundtruth' + img0_tuple[-19:-7] + '.jpg'

        img0 = Image.open(img0_tuple).convert('RGB')      # input color image pair
        img1 = Image.open(img1_tuple).convert('RGB')
        # Random grayscale
        # i = np.random.randint(0, 2, size=1)  # 随机0-2之间的整数
        # img0 = color2gray(img0, i)
        # img1 = color2gray(img1, i)
        truth32 = Image.open(truth_tuple).convert('L')       # single channel 0-1
        truth16 = truth32.resize((truth32.size[0]//2, truth32.size[1]//2))
        truth8 = truth16.resize((truth16.size[0]//2, truth16.size[1]//2))
        truth4 = truth8.resize((truth8.size[0]//2, truth8.size[1]//2))
        fuse = Image.open(truth_fuse).convert('RGB')
        # ------------------data enhancement--------------------------#
        j = np.random.randint(0, 3, size=1)  # 随机0-3之间的整数
        img0 = rotate(img0, j)
        img1 = rotate(img1, j)
        truth32 = rotate(truth32, j)
        truth16 = rotate(truth16, j)
        truth8 = rotate(truth8, j)
        truth4 = rotate(truth4, j)
        fuse = rotate(fuse, j)
        # ------------------To tensor------------------#
        if self.transform is not None:
            tran = transforms.ToTensor()
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            input_img = torch.cat((img0, img1), 0)
            fuse = self.transform(fuse)
            truth32 = tran(truth32)
            truth16 = tran(truth16)
            truth8 = tran(truth8)
            truth4 = tran(truth4)

            return input_img, truth32, truth16, truth8, truth4, fuse

    def __len__(self):
        return len(self.imageFolderDataset)


class AverageMeter(object):
    """Computes and stores the average and current value 计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader, model, criterion1, criterion2, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    losses_1 = AverageMeter()
    losses_2 = AverageMeter()
    losses_3 = AverageMeter()
    losses_4 = AverageMeter()
    weight = args.weight
    model.train()

    for i, (input, target32, target16, target8, target4, fusetruth) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if use_gpu:
            input = input.cuda()
            target32 = target32.cuda()
            target16 = target16.cuda()
            target8 = target8.cuda()
            target4 = target4.cuda()
            # fusetruth = fusetruth.cuda()
        else:
            input = input
            target32 = target32
            target16 = target16
            target8 = target8
            target4 = target4
            # fusetruth = fusetruth
        out32, out16, out8, out4 = model(input)
        loss_1 = weight[0] * criterion1(out32, target32)
        loss_2 = weight[1] * criterion1(out16, target16)
        loss_3 = weight[2] * criterion1(out8, target8)
        loss_4 = weight[3] * criterion1(out4, target4)
        loss = loss_1 + loss_2 + loss_3 + loss_4

        losses.update(loss.item(), input.size(0))
        losses_1.update(loss_1.item(), input.size(0))
        losses_2.update(loss_2.item(), input.size(0))
        losses_3.update(loss_3.item(), input.size(0))
        losses_4.update(loss_4.item(), input.size(0))
        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('loss1', losses_1.avg),
        ('loss2', losses_2.avg),
        ('loss3', losses_3.avg),
        ('loss4', losses_4.avg),
    ])

    return log


def validate(args, val_loader, model, criterion1, criterion2):
    losses = AverageMeter()
    weight = args.weight
    model.eval()       # switch to evaluate mode

    with torch.no_grad():
        for i, (input, target32, target16, target8, target4, fusetruth) in tqdm(enumerate(val_loader), total=len(val_loader)):
            if use_gpu:
                input = input.cuda()
                target32 = target32.cuda()
                target16 = target16.cuda()
                target8 = target8.cuda()
                target4 = target4.cuda()
                # fusetruth = fusetruth.cuda()
            else:
                input = input
                target32 = target32
                target16 = target16
                target8 = target8
                target4 = target4
                # fusetruth = fusetruth
            out32, out16, out8, out4 = model(input)
            loss_1 = weight[0] * criterion1(out32, target32)
            loss_2 = weight[1] * criterion1(out16, target16)
            loss_3 = weight[2] * criterion1(out8, target8)
            loss_4 = weight[3] * criterion1(out4, target4)
            loss = loss_1 + loss_2 + loss_3 + loss_4

            losses.update(loss.item(), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
    ])

    return log


def main():
    args = parse_args()

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)               # 创建文件夹保存模型

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))     # 打印参数配置
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:    # 写入参数文件
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)
    cudnn.benchmark = True

    # X_train, y_train, X_test, y_test = load_data()
    # 定义文件dataset
    training_dir = "/VOC2012_240/train/sourceA/"  # 训练集
    folder_dataset_train = glob.glob(training_dir + "*.jpg")
    test_dir = "/VOC2012_240/test/sourceA/"  # 验证集
    folder_dataset_test = glob.glob(test_dir + "*.jpg")
    # 定义图像dataset
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))
                                          ])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406),
                                                              (0.229, 0.224, 0.225))
                                         ])
    dataset_train = GetDataset(imageFolderDataset=folder_dataset_train,
                                                  transform=transform_train)
    dataset_test = GetDataset(imageFolderDataset=folder_dataset_test,
                                                 transform=transform_test)

    # 定义图像dataloader
    train_loader = DataLoader(dataset_train,
                              shuffle=True,
                              batch_size=args.batch_size)

    test_loader = DataLoader(dataset_test,
                             shuffle=True,
                             batch_size=args.batch_size)
    model = net(inplanes=6)
    if use_gpu:
        model = model.cuda()   # 导入模型
        model.cuda()
        # device_ids = range(torch.cuda.device_count())
        # if len(device_ids) > 1:
        #     model = nn.DataParallel(model, device_ids=device_ids)  # 前提是model已经.cuda() 了
    else:
        model = model
    criterion1 = wbce_loss              # nn.BCELoss()
    criterion2 = pytorch_iou.IOU(size_average=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)    # Adam法优化,filter是为了固定部分参数
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
    #                       momentum=args.momentum, weight_decay=args.weight_decay)           # 梯度下降法优化
    # scheduler = lr_scheduler.MultiStepLR(optimizer,
    #         milestones=[int(e) for e in args.milestones.split(',')], gamma=args.gamma)     # 学习率Lr调度程序
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    log = pd.DataFrame(index=[],
                       columns=['epoch',
                                'lr',
                                'loss',
                                'loss1',
                                'loss2',
                                'loss3',
                                'loss4',
                                'val_loss'])

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch+1, args.epochs))
        # train for one epoch
        train_log = train(args, train_loader, model, criterion1, criterion2, optimizer, epoch)     # 训练集
        # evaluate on validation set
        val_log = validate(args, test_loader, model, criterion1, criterion2)   # 验证集
        print('loss: %.4f - loss1: %.4f - loss2: %.4f - loss3: %.4f - loss4: %.4f - val_loss: %.4f'
              % (train_log['loss'],
                 train_log['loss1'],
                 train_log['loss2'],
                 train_log['loss3'],
                 train_log['loss4'],
                 val_log['loss']))

        tmp = pd.Series([
            epoch+1,
            scheduler.get_lr()[0],
            train_log['loss'],
            train_log['loss1'],
            train_log['loss2'],
            train_log['loss3'],
            train_log['loss4'],
            val_log['loss'],
        ], index=['epoch', 'lr', 'loss', 'loss1', 'loss2', 'loss3', 'loss4', 'val_loss'])         # Series创建字典

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)        # log:训练的日志记录

        scheduler.step()  # adjust lr
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), 'models/%s/model_{}.pth'.format(epoch+1) %args.name)
    # torch.save(model.state_dict(), 'models/%s/model.pth' %args.name)


if __name__ == '__main__':
    main()


