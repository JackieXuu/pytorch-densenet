from __future__ import print_function

import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from tqdm import tqdm

def load_dataset(FLAGS):
	data = {}
	if not os.path.exists('data'):
		os.mkdir('data')
	if FLAGS.dataset == 'cifar10':
		mean = [0.5071, 0.4867, 0.4408]
		stdv = [0.2675, 0.2565, 0.2761]
		train_transform = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=stdv),
		])
		test_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=stdv),
		])
		data['train'] = datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
		data['test'] = datasets.CIFAR10(root='data', train=False, download=True, transform=test_transform)
	else:
		raise NotImplementedError
	FLAGS.classes = 0
	for modal in data.keys():
		for input, label in tqdm(data[modal]):
			FLAGS.classes = max(label+1, FLAGS.classes)
	loader = {}
	kwargs={'num_workers':8, 'pin_memory':True}
	for modal in data.keys():
		loader[modal] = torch.utils.data.DataLoader(data[modal], batch_size=FLAGS.batch_size, shuffle=[modal == 'train'], **kwargs)
	return data, loader

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default = 'cifar10')
	parser.add_argument('--batch_size', default = 128, type=int)
	FLAGS = parser.parse_args()
	data, loader = load_dataset(FLAGS)




