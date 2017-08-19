from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F 
import math


class BasicBlock(nn.Module):
	def __init__(self, numIn, numOut, dropRate = 0.2):
		super(BasicBlock, self).__init__()
		self.bn1 = nn.BatchNorm2d(numIn)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(numIn, numOut, kernel_size=3, stride=1, padding=1, bias=False)
		self.droprate = dropRate

	def forward(self, x):
		out = self.conv1(self.relu1(self.bn1(x)))
		if self.droprate > 0:
			out = F.dropout(out, p=self.droprate, training = self.training)
		out = torch.cat([x, out], 1)
		return out

class BottleneckBlock(nn.Module):
	def __init__(self, numIn, numOut, dropRate = 0.2):
		super(BottleneckBlock, self).__init__()
		numMid = numOut * 4
		self.bn1 = nn.BatchNorm2d(numIn)
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(numIn, numMid, kernel_size=1, stride=1, padding=0, bias=False)

		self.bn2 = nn.BatchNorm2d(numMid)
		self.conv2 = nn.Conv2d(numMid, numOut, kernel_size=3, stride=1, padding=1, bias=False)
		self.droprate = dropRate

	def forward(self, x):
		out = self.conv1(self.relu(self.bn1(x)))
		out = self.conv2(self.relu(self.bn2(out)))
		if self.droprate > 0:
			out = F.dropout(out, p=self.droprate, training = self.training)
		out = torch.cat([x, out], 1)
		return out


class TransitionBlock(nn.Module):
	def __init__(self, numIn, numOut,dropRate = 0.2):
		super(TransitionBlock, self).__init__()
		self.bn1 = nn.BatchNorm2d(numIn)
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(numIn, numOut, kernel_size=1, stride=1, padding=0, bias=False)
		self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
		self.droprate = dropRate

	def forward(self, x):
		out = self.conv1(self.relu(self.bn1(x)))
		if self.droprate > 0:
			out = F.dropout(out, p=self.droprate, training = self.training)
		out = self.pool1(out)
		return out


class DenseNet(nn.Module):
	def __init__(self, classes = 10, depth = 100, growth = 12, reduction = 0.5, bottleneck = True):
		super(DenseNet, self).__init__()

		blocks = (depth - 4) // 3
		if bottleneck:
			blocks //= 2

		self.modules = []

		numIn = 2 * growth
		self.modules.append(nn.Conv2d(3, numIn, kernel_size=3, padding=1, bias=False))
		self.modules.append(self._make_dense(numIn, growth, blocks, bottleneck))

		numIn += blocks * growth
		numOut = int(numIn * reduction)
		self.modules.append(TransitionBlock(numIn, numOut))

		numIn = numOut
		self.modules.append(self._make_dense(numIn, growth, blocks, bottleneck))

		numIn += blocks * growth
		numOut = int(numIn * reduction)
		self.modules.append(TransitionBlock(numIn, numOut))

		numIn = numOut
		self.modules.append(self._make_dense(numIn, growth, blocks, bottleneck))		

		numIn += blocks * growth
		self.modules.append(nn.Sequential(
			nn.BatchNorm2d(numIn),
			nn.ReLU(True),
			nn.AvgPool2d(8)
		))

		self.fc = nn.Linear(numIn, classes)
		self.channels = numIn
		self.net = nn.Sequential(*self.modules)

		for m in self.modules:
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.bias.data.zero_()


	def _make_dense(self, numIn, growth, blocks, bottleneck):
		layers = []

		for i in range(int(blocks)):
			if bottleneck:
				layers.append(BottleneckBlock(numIn, growth))
			else:
				layers.append(BasicBlock(numIn, growth))
			numIn += growth
		return nn.Sequential(*layers)

	def forward(self, x):
		out = self.net(x)
		out = out.view(-1, self.channels)
		out = self.fc(out)
		return out

if __name__ == '__main__':
	densenet = DenseNet()
	print(densenet)
