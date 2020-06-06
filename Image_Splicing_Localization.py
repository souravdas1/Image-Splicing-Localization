# Reference
# Exploiting Spatial Structure for Localizing Manipulated Image Regions
# https://escholarship.org/uc/item/4s13z9qm

from __future__ import print_function, absolute_import
from __future__ import division


import torch.optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import shutil
import datetime

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

########################################################################################################
########################################################################################################


__all__ = ['accuracy', 'AverageMeter']


def accuracy(output, target, thr=0.5):
    ''' Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    '''

    if output.dim() > 2:
        v,i = torch.max(output,1);
    else:
        v,i = torch.max(output,1);
    return torch.sum(target.long() == i).float()/target.numel()


class AverageMeter(object):
    """Computes and stores the average and current value"""
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

########################################################################################################
########################################################################################################

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', snapshot=None):
    # preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    # scipy.io.savemat(os.path.join(checkpoint, 'preds.mat'), mdict={'preds' : preds})

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state.epoch)))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
        # scipy.io.savemat(os.path.join(checkpoint, 'preds_best.mat'), mdict={'preds' : preds})


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


########################################################################################################
########################################################################################################



import torch.nn as nn
import numpy as np


import pandas as pd
import sys
import math



class LSTMEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(LSTMEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(3, 16, 5, padding=2, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 1, 5, padding=2, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=3, batch_first=True)
        self.lstm.flatten_parameters()

    def forward(self, x):
        bs = x.size(0)

        x = self.relu1(self.conv1(x))  # bsx16x64x64

        y = self.relu2(self.conv2(x))  # bsx1x64x64

        # split x to 8x8 blocks
        y_list = y.split(8, dim=3)  # 8x[(bsx1x8x64)]
        xy_list = [x.split(8, dim=2) for x in y_list]  # 8x8x( bsx 1x 8 x 8)

        xy = [item for items in xy_list for item in items]

        xy = torch.cat(xy, 1)  # bsx64x(8x8)

        xy = xy.view(bs, 64, 64)  # bs x 64 x 64

        self.lstm.flatten_parameters()
        # 8x8 list
        outputs, (ht, ct) = self.lstm(xy)

        return outputs


class Segmentation(nn.Module):
    def __init__(self, patch_size):
        super(Segmentation, self).__init__()

        self.sqrt_patch_size = int(math.sqrt(patch_size))
        self.patch_size = patch_size

        self.conv3 = nn.Conv2d(1, 32, 5, padding=2, bias=False)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(32, 2, 5, padding=2, bias=False)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(2, 2, 5, padding=2, bias=False)
        self.relu5 = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(2, stride=2)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, outputs_lstm):
        bs = outputs_lstm.size(0)
        # outputs = [bs,64,256]
        outputs_lstm = outputs_lstm.contiguous().view(bs, self.sqrt_patch_size, self.sqrt_patch_size,
                                                      self.sqrt_patch_size * 2, self.sqrt_patch_size * 2).permute(0, 1,
                                                                                                                  3, 2,
                                                                                                                  4).contiguous()

        # bs,8*16,8*16
        outputs_lstm = outputs_lstm.view(bs, 1, self.patch_size * 2, self.patch_size * 2)

        # bs x 32 x 96x96
        x = self.relu3(self.conv3(outputs_lstm))

        x = self.max_pool(x)

        x = self.relu4(self.conv4(x))

        output_mask = self.softmax(self.conv5(x))

        return output_mask


class Classification(nn.Module):
    """docstring for Classification"""

    def __init__(self, hidden_dim, number_of_class=2):
        super(Classification, self).__init__()
        self.hidden_dim = hidden_dim
        self.number_of_class = number_of_class

        self.linear = nn.Linear(self.hidden_dim, self.number_of_class)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, outputs_lstm):
        outputs = outputs_lstm[:, -1, :]  # bsx256

        y = self.softmax(self.linear(outputs))

        return y

########################################################################################################
########################################################################################################
import scipy.misc

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def save_pred(preds, checkpoint='checkpoint', filename='preds_valid.mat'):
    preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    scipy.io.savemat(filepath, mdict={'preds' : preds})


def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img

def load_image(img_path):
    # H x W x C => C x H x W
    return im_to_torch(scipy.misc.imread(img_path, mode='RGB'))

def load_image_gray(img_path):
    # H x W x C => C x H x W
    x = scipy.misc.imread(img_path, mode='L')
    x = x[:,:,np.newaxis]
    return im_to_torch(x)


########################################################################################################
########################################################################################################
from builtins import map



def blockwise_view(a, blockshape, aslist=False, require_aligned_blocks=True):
    """
    Return a 2N-D view of the given N-D array, rearranged so each ND block (tile)
    of the original array is indexed by its block address using the first N
    indexes of the output array.

    Note: This function is nearly identical to ``skimage.util.view_as_blocks()``, except:
          - "imperfect" block shapes are permitted (via require_aligned_blocks=False)
          - only contiguous arrays are accepted.  (This function will NOT silently copy your array.)
            As a result, the return value is *always* a view of the input.

    Args:
        a: The ND array

        blockshape: The tile shape

        aslist: If True, return all blocks as a list of ND blocks
                instead of a 2D array indexed by ND block coordinate.

        require_aligned_blocks: If True, check to make sure no data is "left over"
                                in each row/column/etc. of the output view.
                                That is, the blockshape must divide evenly into the full array shape.
                                If False, "leftover" items that cannot be made into complete blocks
                                will be discarded from the output view.

    Here's a 2D example (this function also works for ND):

    >>> a = np.arange(1,21).reshape(4,5)
    >>> print a
    [[ 1  2  3  4  5]
     [ 6  7  8  9 10]
     [11 12 13 14 15]
     [16 17 18 19 20]]

    >>> view = blockwise_view(a, (2,2), False)
    >>> print view
    [[[[ 1  2]
       [ 6  7]]
    <BLANKLINE>
      [[ 3  4]
       [ 8  9]]]
    <BLANKLINE>
    <BLANKLINE>
     [[[11 12]
       [16 17]]
    <BLANKLINE>
      [[13 14]
       [18 19]]]]

    Inspired by the 2D example shown here: http://stackoverflow.com/a/8070716/162094
    """
    assert a.flags['C_CONTIGUOUS'], "This function relies on the memory layout of the array."
    blockshape = tuple(blockshape)
    outershape = tuple(np.array(a.shape) // blockshape)
    view_shape = outershape + blockshape

    if require_aligned_blocks:
        assert (np.mod(a.shape, blockshape) == 0).all(), \
            "blockshape {} must divide evenly into array shape {}" \
                .format(blockshape, a.shape)

    # inner strides: strides within each block (same as original array)
    intra_block_strides = a.strides

    # outer strides: strides from one block to another
    inter_block_strides = tuple(a.strides * np.array(blockshape))

    # This is where the magic happens.
    # Generate a view with our new strides (outer+inner).
    view = np.lib.stride_tricks.as_strided(a,
                                              shape=view_shape,
                                              strides=(inter_block_strides + intra_block_strides))

    return view

########################################################################################################
########################################################################################################

import torch.utils.data as data

class Splicing(data.Dataset):
    def __init__(self, base_folder, img_folder, arch, patch_size=64):

        self.base_folder = base_folder
        print('base_folder: ', base_folder)
        self.patch_size = patch_size
        self.path = img_folder
        print('img_folder: ', img_folder)
        self.train = []
        self.anno = []
        self.arch = arch
        # Data loading code
        with open(base_folder + '/' + img_folder) as f:
            for file_name in f.readlines():
                recoders = file_name.rstrip().split(',')
                # [PATCH_PATH,MASK_PATCH_PATH,MASK_PATH,IMG_PATH])
                self.train.append(recoders[0])
                self.anno.append(recoders[1])

        print('total Dataset of ' + img_folder + ' is : ', len(self.train))

    def __getitem__(self, index):
        # different in tifs dataset and columbia dataset,
        # the mask in tifs dataset 1 represent image splicing region
        # the mask in columbia dataset 0 represent image splicing region

        img_path = self.base_folder + '/' + self.train[index]
        anno_path = self.base_folder + '/' + self.anno[index]


        img = load_image(img_path)  # CxHxW => automatically change
        mask = load_image_gray(anno_path)

        if 'columbia' in self.path:
            mask = mask * -1 + 1

        # here 1 will represent image splicing region in mask

        label = 1 if mask.sum() > mask.numel() * 0.875 else 0

        return_tuple = (img, mask[0], label)

        return return_tuple

    def __len__(self):

        return len(self.train)




########################################################################################################
########################################################################################################


import torch.utils.data as data


#from scripts.utils.transforms import *

#from tools.blockwise_view import blockwise_view
from PIL import Image


class SplicingFull(data.Dataset):
    def __init__(self, base_folder, img_folder, arch, patch_size=64):

        self.base_folder = base_folder
        self.patch_size = patch_size
        self.path = img_folder
        self.train = []
        self.anno = []
        self.original_image = []
        self.original_mask = []
        self.img_fulls = []  # resized
        self.arch = arch
        # Data loading code
        with open(base_folder + '/' + img_folder) as f:
            for file_name in f.readlines():
                recoders = file_name.rstrip().split(',')
                # [PATCH_PATH,MASK_PATCH_PATH,MASK_PATH,IMG_PATH])
                if recoders[3] not in self.original_image:
                    self.original_image.append(recoders[3])
                    if 'columbia' in self.path:
                        self.original_mask.append(recoders[2])

        print('total validation of ' + img_folder +
              ' is : ', len(self.original_image))

    def __getitem__(self, index):

        img_path = self.base_folder + '/' + self.original_image[index]
        anno_path = self.base_folder + '/' + self.original_mask[index]

        img = Image.open(img_path).convert('RGB')

        mask = Image.open(anno_path)
        mask = mask.resize(img.size, Image.ANTIALIAS)

        img = np.array(img).astype(np.float32) / 255  # [0,1】

        if 'columbia' in self.path:  # we don't need to process tifs dataset here.
            # convert rgb [red as 1]
            rgbmask = mask.convert('RGB')
            rgbmask = np.array(rgbmask)
            idx = np.argmax(rgbmask, 2)
            mask = np.ones([rgbmask.shape[0], rgbmask.shape[1]])
            mask[idx == 0] = 0  # red

        #        8*8*64*64*3
        image_patches = blockwise_view(
            np.array(img), (64, 64, 3), require_aligned_blocks=False).squeeze(axis=2)
        mask_patches = blockwise_view(
            np.array(mask), (64, 64), require_aligned_blocks=False).squeeze()

        mask = mask[0:64 * image_patches.shape[0], 0:64 * image_patches.shape[1], np.newaxis]
        batchsize = image_patches.shape[0] * image_patches.shape[1]
        image_patches = np.reshape(image_patches, (batchsize, 64, 64, 3))
        mask_patches = np.reshape(mask_patches, (batchsize, 64, 64))



        mask = im_to_torch(mask)[0]
        # patches to torch
        image_patches = to_torch(np.transpose(
            image_patches, axes=(0, 3, 1, 2)))

        labels = np.zeros(batchsize)

        threshold = 64 * 64 * 0.875

        for i in range(batchsize):
            if np.sum(mask_patches[i]) > threshold:  # 64*64*0.875
                labels[i] = 1

        return (load_image(img_path),image_patches, labels, mask)

    def __len__(self):

        return len(self.original_image)


########################################################################################################
########################################################################################################

best_acc = 0
alpha=0.5
arch='hybrid_network'
base_dir='D:/APPLIEDAI/souravD/image-splicing-localization-master/data'
#base_dir='G:/sourav/prog/notebook/Image_manipulation_detection-master/image-splicing-localization-master/data'
blocks=1
checkpoint='checkpoint/local_columbia64_hybrid_network'
data='columbia64'
debug=False
epochs=35
evaluate=False
features=256
finetune=''
flip=False
gamma=0.1
gpu=True
label_type='Gaussian'
lr=0.0001
momentum=0
num_classes=2
resume=''
schedule=[20, 40]
sigma=1
sigma_decay=0
start_epoch=0
test_batch=6
train_batch=1
update=''
weight_decay=0
workers=0

########################################################################################################
########################################################################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)

print("==> creating model Splicing ")

writer = SummaryWriter(checkpoint + '/' + 'ckpt')


def hybrid_network():
    model = HybridN()
    model.apply(weights_init)
    return model


class HybridN(nn.Module):
    def __init__(self):
        super(HybridN, self).__init__()

        patch_size = 64
        hidden_dim = 256
        number_of_class = 2
        self.encode_lstm = LSTMEncoder(patch_size, hidden_dim)
        self.classification = Classification(hidden_dim, number_of_class)
        self.Segmentation = Segmentation(patch_size)

    def forward(self, x):
        local_feature = self.encode_lstm(x)
        classification = self.classification(local_feature)
        seg = self.Segmentation(local_feature)
        return classification,seg


model = hybrid_network()
print(model)

wgt = torch.Tensor([1,5]);

# define loss function (criterion) and optimizer
criterion = torch.nn.NLLLoss(wgt)
criterion2d = torch.nn.NLLLoss(wgt)
criterionL1 = torch.nn.L1Loss()

if gpu:
    cudnn.benchmark = True
    model = torch.nn.DataParallel(model).cuda()
    criterion.cuda()
    criterion2d.cuda()
    criterionL1.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999), weight_decay=weight_decay)

BASE_DIR = base_dir
splicing_dataset_loader = Splicing

train_loader = torch.utils.data.DataLoader(
    splicing_dataset_loader(BASE_DIR, data+'/train.txt', arch=arch),
    batch_size=train_batch, shuffle=True,
    num_workers=workers, pin_memory=False)

val_loader = torch.utils.data.DataLoader(
    splicing_dataset_loader(BASE_DIR, data+'/val.txt', arch=arch),
    batch_size=64, shuffle=False,
    num_workers=workers, pin_memory=False)

test_loader = torch.utils.data.DataLoader(
    SplicingFull(BASE_DIR, data +
                            '/test.txt', arch=arch),
    batch_size=1, shuffle=False,
    num_workers=workers, pin_memory=False)


def train(train_loader, model, criterions, optimizer):
    losses_label = AverageMeter()
    acces_label = AverageMeter()
    losses_mask = AverageMeter()
    acces_mask = AverageMeter()

    criterion_classification = criterions[0]
    criterion_segmentation = criterions[1]

    # switch to train mode
    model.train()

    for i, (inputs, target, label) in enumerate(train_loader):
        # measure data loading time

        if gpu:
            inputs = inputs.cuda()
            target = target.cuda()
            label = label.cuda()

        input_var = torch.autograd.Variable(inputs)
        target_var = torch.autograd.Variable(target.long())
        label_var = torch.autograd.Variable(label.long())

        # compute output
        output_label,output_mask = model(input_var)
        loss_label = criterion_classification(output_label, label_var)
        loss_mask = criterion_segmentation(output_mask, target_var)

        loss = loss_label + 10*loss_mask

        acc_label = accuracy(output_label.data, label)
        acc_mask = accuracy(output_mask.data, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses_label.update(loss_label.item(), inputs.size(0))
        losses_mask.update(loss_mask.item(), inputs.size(0))
        acces_label.update(acc_label, inputs.size(0))
        acces_mask.update(acc_mask, inputs.size(0))

    return losses_label.avg, acces_label.avg, losses_mask.avg, acces_mask.avg


def validate(val_loader, model, criterions):
    criterion_classification = criterions[0]
    criterion_segmentation = criterions[1]

    losses_label = AverageMeter()
    acces_label = AverageMeter()
    losses_mask = AverageMeter()
    acces_mask = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (inputs, target, label) in enumerate(val_loader):
        # measure data loading time
        if gpu:
            inputs = inputs.cuda()
            target = target.cuda()
            label = label.cuda()

        with torch.no_grad():
            input_var = torch.autograd.Variable(inputs)
            target_var = torch.autograd.Variable(target.long())
            label_var = torch.autograd.Variable(label.long())

        # compute output
        output_label, output_mask = model(input_var)

        loss_label = criterion_classification(output_label, label_var)
        loss_mask = criterion_segmentation(output_mask, target_var)

        acc_label = accuracy(output_label.data.cpu(), label.cpu())
        acc_mask = accuracy(output_mask.data.cpu(), target.cpu())

        # measure accuracy and record loss
        losses_label.update(loss_label.item(), inputs.size(0))
        losses_mask.update(loss_mask.item(), inputs.size(0))
        acces_label.update(acc_label, inputs.size(0))
        acces_mask.update(acc_mask, inputs.size(0))

    return losses_label.avg, acces_label.avg, losses_mask.avg, acces_mask.avg


for epoch in tqdm(range(start_epoch, epochs)):
    print(datetime.datetime.now())
    lr = adjust_learning_rate(optimizer, epoch, lr, schedule, gamma)

    print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

    # decay sigma
    if sigma_decay > 0:
        train_loader.dataset.sigma *= sigma_decay
        val_loader.dataset.sigma *= sigma_decay

    # train for one epoch
    train_loss_label, train_acc_label, train_loss_mask, train_acc_mask = train(
        train_loader, model, [criterion, criterion2d], optimizer)

    print('train_loss_mask:', train_loss_mask, 'train_acc_mask:', train_acc_mask)

    # evaluate on validation set
    val_loss_label, val_acc_label, val_loss_mask, val_acc_mask = validate(
        val_loader, model, [criterion, criterion2d])

    print('val_loss_mask:', val_loss_mask, 'val_acc_mask:', val_acc_mask)

    # Visualization train
    writer.add_scalar('train/loss/label', train_loss_label, epoch)
    writer.add_scalar('train/loss/mask', train_loss_mask, epoch)
    writer.add_scalar('train/acc/label', train_acc_label, epoch)
    writer.add_scalar('train/acc/mask', train_acc_mask, epoch)
    # Visualization val
    writer.add_scalar('val/loss/label', val_loss_label, epoch)
    writer.add_scalar('val/loss/mask', val_loss_mask, epoch)
    writer.add_scalar('val/acc/label', val_acc_label, epoch)
    writer.add_scalar('val/acc/mask', val_acc_mask, epoch)

    # visualization learning rate
    writer.add_scalar('lr', lr, epoch)

    tmp_acc = 0

    for i, (img, inputs, labels, target) in enumerate(test_loader):
        # measure data loading time

        with torch.no_grad():
            inputs_var = torch.autograd.Variable(inputs.view(-1, 3, 64, 64))

        if gpu:
            inputs_var = inputs_var.cuda()

        pred_label, pred_mask = model(inputs_var)

        _, max_cls_channel = torch.max(pred_label.cpu().data,dim=1)
        _, max_seg_channel = torch.max(pred_mask.cpu().data, dim=1)

        # x,y
        pred_class = max_cls_channel.view(-1,1,1,1).repeat(1,1,64,64)
        pred_class = pred_class.contiguous().view(target.size(1)//64,target.size(2)//64,64,64).permute(0,2,1,3).contiguous().view(target.size(1), target.size(2))
        pred_mask = max_seg_channel.contiguous().view(target.size(1) // 64, target.size(2) //
                                                      64, 64, 64).permute(0, 2, 1, 3).contiguous().view(target.size(1),
                                                                                                        target.size(2))
        # Visualization test
        writer.add_scalar('test/acc/label/'+str(i), (max_cls_channel == labels[0].long()).sum(), epoch)
        writer.add_scalar('test/acc/mask/' + str(i),
                          (pred_mask == target.long()).sum(), epoch)

        writer.add_image('test/label/'+str(i),pred_class.float(),epoch)
        writer.add_image('test/' + str(i) + '/orig_image/', img[0], epoch)
        writer.add_image('test/' + str(i) + '/seg/', pred_mask.float(), epoch)
        writer.add_image('test/' + str(i) + '/seg_gt/', target, epoch)


        tmp_acc = tmp_acc + (target.long() == pred_mask).sum()

    valid_acc = tmp_acc / len(test_loader)

    # remember best acc and save checkpoint
    is_best = valid_acc > best_acc
    best_acc = max(valid_acc, best_acc)
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': arch,
        'state_dict': model.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }, is_best, checkpoint=checkpoint)

writer.close()
