import sys
import cv2
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import time

from wpod import WPODNet
from dataset import LPDataset
from torch.utils.data import DataLoader
from src.loss import MyLoss
from src.utils import image_files_from_folder
from src.label import readShapes

from os.path import isfile, isdir, basename, splitext
from os import makedirs

def weights_init(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv2d') != -1:
		nn.init.xavier_normal_(m.weight.data)
		nn.init.constant_(m.bias.data, 0.0)
	elif classname.find('Linear') != -1:
		nn.init.xavier_normal_(m.weight.data)
		nn.init.constant_(m.bias.data, 0.0)

def train(train_loader, model, optimizer):
	for step, (data, target) in enumerate(train_loader):
#		start = time.clock()
		data = data.cuda()
		target = target.cuda()

		optimizer.zero_grad()
		logits = model(data)
		Loss = criterion(logits,target)
		Loss.backward()
#		end = time.clock()
#		print("\ttime: %f s" % (end-start))
		Loss = Loss.data
		optimizer.step()
		return Loss

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
#	parser.add_argument('-m' 		,'--model'			,type=str   , required=True		,help='Path to previous model')
	parser.add_argument('-n' 		,'--name'			,type=str   , default='pytorch-model'		,help='Model name')
	parser.add_argument('-tr'		,'--train-dir'		,type=str   , default='samples/train-detector/'		,help='Input data directory for training')
	parser.add_argument('-its'		,'--iterations'		,type=int   , default=300000	,help='Number of mini-batch iterations (default = 300.000)')
	parser.add_argument('-bs'		,'--batch-size'		,type=int   , default=32		,help='Mini-batch size (default = 32)')
	parser.add_argument('-od'		,'--output-dir'		,type=str   , default='./'		,help='Output directory (default = ./)')
	parser.add_argument('-op'		,'--optimizer'		,type=str   , default='Adam'	,help='Optmizer (default = Adam)')
	parser.add_argument('-lr'		,'--learning-rate'	,type=float , default=.01		,help='Optmizer (default = 0.01)')
#	parser.add_argument('-gpu'		,'--gpu',			,type=int   , default=0,		,help='gpu device id')
	args = parser.parse_args()

	netname 	= basename(args.name)
	train_dir 	= args.train_dir
	outdir 		= args.output_dir

	iterations 	= args.iterations
	batch_size 	= args.batch_size
	dim 		= 208

	if not isdir(outdir):
		makedirs(outdir)

	if not torch.cuda.is_available():
		logging.info('no gpu device available')
		sys.exit(1)

	print('Checking input directory...')
	Files = image_files_from_folder(train_dir)
	Data = []
	for file in Files:
		labfile = splitext(file)[0] + '.txt'
		if isfile(labfile):
			L = readShapes(labfile)
			I = cv2.imread(file)
			Data.append([I,L[0]])
	print('%d images with labels found' % len(Data))

#	torch.cuda.set_device(args.gpu)

	train_data = LPDataset(path = train_dir, img_size = dim)
	train_loader = DataLoader(train_data,batch_size = batch_size, shuffle=False)

	model = WPODNet()
	model.apply(weights_init)
	model.cuda()
#	criterion = loss()

	optimizer = torch.optim.Adam(model.parameters(),lr = args.learning_rate)
	criterion = MyLoss()
	model_path_backup = '%s/%s_backup' % (outdir,netname)
	model_path_final  = '%s/%s_final'  % (outdir,netname)
	model.train()
	for it in range(args.iterations):
		train_loss = train(train_loader, model, optimizer)
		if (it+1) % 100 == 0:
			print('Iter. %d (of %d)' % (it+1,args.iterations))
			print('\tLoss: %f' % train_loss)
		# Save model every 1000 iterations
		if (it+1) % 1000 == 0:
			print('Saving model (%s)' % model_path_backup)
			torch.save(model.state_dict(), model_path_backup+'.pt')
	
	print('Over')

	print('Saving model (%s)' % model_path_final)
	torch.save(model.state_dict(), model_path_final+'.pt')

