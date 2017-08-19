from __future__ import division, print_function


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import os

import shutil
import numpy as np 
import argparse
from dataset import load_dataset
from densenet import DenseNet
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Densenet for Classification')
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--dataset', default="cifar10", type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epoch', default=300, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--checkpoint', default="model", type=str)
parser.add_argument('--resume_model', default=None, type=str)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
FLAGS = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu

def init():
	torch.manual_seed(FLAGS.seed)
	torch.cuda.manual_seed(FLAGS.seed)
	torch.backends.cudnn.benchmark = True

	data, loader = load_dataset(FLAGS)
	print('==> data loaded')

	model = DenseNet(classes=FLAGS.classes)
	model.cuda()
	print('==> model loaded')

	optimizer = optim.SGD(model.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum, nesterov=True)
	criterion = nn.CrossEntropyLoss().cuda()
	return model, optimizer, criterion, loader


def exp_lr_scheduler(optimizer, epoch, target_epoch, lr):
	lr = lr
	if float(epoch) / target_epoch > 0.75:
		lr = lr * 0.01
	elif float(epoch) / target_epoch > 0.5:
		lr = lr * 0.1 

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def train(model, optimizer, criterion, loader, epoch):
	model.train()
	Acc = 0.0
	Loss = 0.0
	for batch_idx, (input, label) in enumerate(loader):
		label = Variable(label.cuda(async=True))
		input = Variable(input.cuda())

		output = model(input)
		loss = criterion(output, label)

		prec = accuracy(output.data, label.data, topk=(1,))[0]
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if batch_idx % 50 == 0:
			print('[Train] Epoch: {}/{} Batch: {}/{} Accuracy: {:.6f} Loss: {:.7f}'.format(epoch, FLAGS.epoch, batch_idx, len(loader), prec, loss.data[0]))
		Acc += prec
		Loss += loss.data[0]
	Acc /= len(loader)
	Loss /= len(loader)
	print('[Train] Epoch {} Average Accuracy: {:.6f} Average Loss {:.7f}'.format(epoch, Acc, Loss))

def valid(model, criterion, loader, epoch):
	model.eval()
	Acc = 0.0
	Loss = 0.0
	for batch_idx, (input, label) in enumerate(loader):
		label = Variable(label.cuda(async=True), volatile=True)
		input = Variable(input.cuda(), volatile=True)
		output = model(input)
		loss = criterion(output, label)

		prec = accuracy(output.data, label.data, topk=(1,))[0]
		Acc += prec
		Loss += loss.data[0]
	Acc /= len(loader)
	Loss /= len(loader)
	print('[Valid] Epoch {} Average Accuracy: {:.6f} Average Loss {:.7f}'.format(epoch, Acc, Loss))	
	return Acc		



def save_checkpoint(state, filename='checkpoint.pth.tar'):
	torch.save(state, filename)

def resume(filename, optimizer, model):
	if os.path.isfile(filename):
		print('==> loading  checkpoint {}'.format(filename))
		checkpoint = torch.load(filename)
		start_epoch = checkpoint['epoch'] + 1
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print("==> loaded checkpoint '{}' (epoch {})".format(filename, start_epoch))
	else:
		print("==> no checkpoint found at '{}'".format(filename))
	return model, optimizer, start_epoch


def main():
	if not os.path.isdir(FLAGS.checkpoint):
		os.mkdir(FLAGS.checkpoint)
	model, optimizer, criterion, loader = init()
	start_epoch = 1
	best_acc = 0.0
	if FLAGS.resume_model is not None:
		filename = os.path.join(FLAGS.checkpoint, FLAGS.resume_model)
		model, optimizer, start_epoch = resume(filename, optimizer, model)
	for epoch in range(start_epoch, FLAGS.epoch):
		exp_lr_scheduler(optimizer, epoch, FLAGS.epoch, FLAGS.lr)
		train(model, optimizer, criterion, loader['train'], epoch)
		acc = valid(model, criterion, loader['test'], epoch)
		filename = os.path.join(FLAGS.checkpoint, "densenet_{}.pth".format(epoch))
		save_checkpoint({
			'epoch': epoch,
			'state_dict': model.state_dict(),
			'optimizer': optimizer.state_dict(),
		}, filename=filename)
		if acc > best_acc:
			best_acc =acc
			shutil.copyfile(filename,  os.path.join(FLAGS.checkpoint,"best_model.pth"))


def accuracy(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)

	value, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k[0] / float(batch_size))
	return res


if __name__ == '__main__':
	main()
