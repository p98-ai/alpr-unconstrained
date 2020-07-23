import torch
import torch.nn as nn
import numpy as np

class MyLoss(nn.Module): 
	def __init__(self): 
		super(MyLoss, self).__init__() 

	def forward(self, pred, target): 
		return loss(target, pred) 


def logloss(Ptrue,Pred,szs,eps=10e-10):
	b,h,w,ch = szs
	Pred = torch.clamp(Pred,eps,1.)
	Pred = -torch.log(Pred)
	Pred = Pred*Ptrue
	Pred = torch.reshape(Pred,(b,h*w*ch))
	Pred = torch.sum(Pred,1)
	return Pred

def l1(true,pred,szs):
	b,h,w,ch = szs
	res = torch.reshape(true-pred,(b,h*w*ch))
	res = torch.abs(res)   #qiujueduizhi
	res = torch.sum(res,1) #sum
	return res

def loss(Ytrue, Ypred):

	b = Ytrue.size()[0]
	h = Ytrue.size()[1]
	w = Ytrue.size()[2]


	obj_probs_true = Ytrue[...,0]
	obj_probs_pred = Ypred[:,0,:,:]


	non_obj_probs_true = 1. - Ytrue[...,0]
	non_obj_probs_pred = Ypred[:,1,:,:]

	affine_pred	= Ypred[:,2:,:,:]
	pts_true 	= Ytrue[:,:,:,1:]
	affinex = torch.stack([torch.max(affine_pred[:,0,:,:],torch.zeros(affine_pred[:,0,:,:].size()).cuda()),affine_pred[:,1,:,:],affine_pred[:,2,:,:]],3)
	affiney = torch.stack([affine_pred[:,3,:,:],torch.max(affine_pred[:,4,:,:],torch.zeros(affine_pred[:,4,:,:].size()).cuda()),affine_pred[:,5,:,:]],3)
#	print(affinex.size())
	v = 0.5
#	q = torch.tensor[-v,-v,1., v,-v,1., v,v,1., -v,v,1.]
	base = torch.from_numpy(np.ascontiguousarray(np.array([-v,-v,1., v,-v,1., v,v,1., -v,v,1.], dtype=np.float32))).cuda()
#	base = torch.stack([q,q,q,q])
	base = base.repeat([b,h,w,1])
#	pts = torch.zeros((b,h,w,1))
	
	for i in range(0,12,3):
		row = base[...,i:(i+3)]
#		print(row.size())
		ptsx = torch.sum(affinex*row,3)
		ptsy = torch.sum(affiney*row,3)

		pts_xy = torch.stack([ptsx,ptsy],3)
		if i==0:
			pts = pts_xy
		else:
			pts = (torch.cat([pts,pts_xy],3))
	pts = pts.cuda()
	flags = torch.reshape(obj_probs_true,(b,h,w,1))
	res   = 1.*l1(pts_true*flags,pts*flags,(b,h,w,4*2))
	res   = 1.*logloss(obj_probs_true,obj_probs_pred,(b,h,w,1))+res
	res  += 1.*logloss(non_obj_probs_true,non_obj_probs_pred,(b,h,w,1))+res
	res = torch.mean(res)
	return res