import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class WPODNet(nn.Module):
	def __init__(self):
		super(WPODNet, self).__init__()

		self.conv1 = nn.Conv2d(3, 16, 3, 1,padding=(1,1))
		self.conv2 = nn.Conv2d(16, 16, 3, 1,padding=(1,1))
		self.conv3 = nn.Conv2d(16, 32, 3, 1,padding=(1,1))
		self.conv4 = nn.Conv2d(32, 32, 3, 1,padding=(1,1))
		self.conv5 = nn.Conv2d(32, 64, 3, 1,padding=(1,1))
		self.conv6 = nn.Conv2d(64, 64, 3, 1,padding=(1,1))
		self.conv7 = nn.Conv2d(64, 128, 3, 1,padding=(1,1))
		self.conv8 = nn.Conv2d(128, 128, 3, 1,padding=(1,1))
		self.conv9 = nn.Conv2d(128, 2, 3, 1,padding=(1,1))
		self.conv10 = nn.Conv2d(128, 6, 3, 1,padding=(1,1))
		self.bn1 = nn.BatchNorm2d(num_features=16, eps=0, affine=False, track_running_stats=False)
		self.bn2 = nn.BatchNorm2d(num_features=32, eps=0, affine=False, track_running_stats=False)
		self.bn3 = nn.BatchNorm2d(num_features=64, eps=0, affine=False, track_running_stats=False)
		self.bn4 = nn.BatchNorm2d(num_features=128, eps=0, affine=False, track_running_stats=False)
	def forward(self, x):
		
		x = F.relu(self.bn1(self.conv1(x)))# 
		
		x = F.relu(self.bn1(self.conv2(x)))
		x = F.max_pool2d(x,2,2) # # 1
		x = F.relu(self.bn2(self.conv3(x)))
		x1 = x                     #Res_block
		x1 = F.relu(self.bn2(self.conv4(x1)))
		x1 = self.bn2(self.conv4(x1))
		x = x + x1
		x = F.relu(x)              #end
		x = F.max_pool2d(x,2,2)#2
		x = F.relu(self.bn3(self.conv5(x)))
		x2 = x                     #Res_block
		x2 = F.relu(self.bn3(self.conv6(x2)))
		x2 = self.bn3(self.conv6(x2))
		x = x + x2
		x = F.relu(x)
		x3 = x                     #Res_block
		x3 = F.relu(self.bn3(self.conv6(x3)))
		x3 = self.bn3(self.conv6(x3))
		x = x + x3
		x = F.relu(x)
		x = F.max_pool2d(x,2,2) #3
		x4 = x                     #Res_block
		x4 = F.relu(self.bn3(self.conv6(x4)))
		x4 = self.bn3(self.conv6(x4))
		x = x + x4
		x = F.relu(x)
		x5 = x                     #Res_block
		x5 = F.relu(self.bn3(self.conv6(x5)))
		x5 = self.bn3(self.conv6(x5))
		x = x + x5
		x = F.relu(x)
		x = F.max_pool2d(x,2,2) #4
		x = self.bn4(self.conv7(x))
		x6 = x                     #Res_block
		x6 = F.relu(self.bn4(self.conv8(x6)))
		x6 = self.bn4(self.conv8(x6))
		x = x + x6
		x = F.relu(x)
#		x = F.max_pool2d(x,2,2) #5
		x7 = x                     #Res_block
		x7 = F.relu(self.bn4(self.conv8(x7)))
		x7 = self.bn4(self.conv8(x7))
		x = x + x7
		x = F.relu(x)
		#Detection
		xprob = x
		xaff = x
		xprob = self.conv9(xprob)
		xaff = self.conv10(xaff)
		xprob = F.softmax(xprob,dim = 1)
		x = torch.cat((xprob,xaff),1)
		
		return x 
		

