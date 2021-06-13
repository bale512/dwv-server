import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class PNet(nn.Module):
	def __init__(self, output_chn=2, C=16):
		super(PNet, self).__init__()

		dil = [1, 2, 3, 4, 5] # dilation rates

		self.block1 = nn.Sequential(
			nn.Conv3d(1, C, kernel_size=3, stride=1, padding=dil[0], dilation=dil[0], bias=False),
			nn.ReLU(True),
			nn.Conv3d(C, C, kernel_size=(3,3,1), stride=1, padding=(dil[0],dil[0],0), dilation=dil[0], bias=False),
			nn.ReLU(True))
		self.comp1 = nn.Conv3d(C, C//4, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

		self.block2 = nn.Sequential(
			nn.Conv3d(C, C, kernel_size=3, stride=1, padding=dil[1], dilation=dil[1], bias=False),
			nn.ReLU(True),
			nn.Conv3d(C, C, kernel_size=(3,3,1), stride=1, padding=(dil[1],dil[1],0), dilation=dil[1], bias=False),
			nn.ReLU(True))
		self.comp2 = nn.Conv3d(C, C//4, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

		self.block3 = nn.Sequential(
			nn.Conv3d(C, C, kernel_size=3, stride=1, padding=dil[2], dilation=dil[2], bias=False),
			nn.ReLU(True),
			nn.Conv3d(C, C, kernel_size=(3,3,1), stride=1, padding=(dil[2],dil[2],0), dilation=dil[2], bias=False),
			nn.ReLU(True),
			nn.Conv3d(C, C, kernel_size=(3,3,1), stride=1, padding=(dil[2],dil[2],0), dilation=dil[2], bias=False),
			nn.ReLU(True))
		self.comp3 = nn.Conv3d(C, C//4, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

		self.block4 = nn.Sequential(
			nn.Conv3d(C, C, kernel_size=3, stride=1, padding=dil[3], dilation=dil[3], bias=False),
			nn.ReLU(True),
			nn.Conv3d(C, C, kernel_size=(3,3,1), stride=1, padding=(dil[3],dil[3],0), dilation=dil[3], bias=False),
			nn.ReLU(True),
			nn.Conv3d(C, C, kernel_size=(3,3,1), stride=1, padding=(dil[3],dil[3],0), dilation=dil[3], bias=False),
			nn.ReLU(True))
		self.comp4 = nn.Conv3d(C, C//4, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

		self.block5 = nn.Sequential(
			nn.Conv3d(C, C, kernel_size=3, stride=1, padding=dil[4], dilation=dil[4], bias=False),
			nn.ReLU(True),
			nn.Conv3d(C, C, kernel_size=(3,3,1), stride=1, padding=(dil[4],dil[4],0), dilation=dil[4], bias=False),
			nn.ReLU(True),
			nn.Conv3d(C, C, kernel_size=(3,3,1), stride=1, padding=(dil[4],dil[4],0), dilation=dil[4], bias=False),
			nn.ReLU(True))
		self.comp5 = nn.Conv3d(C, C//4, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

		self.block6 = nn.Sequential(
			nn.Conv3d(5*C//4, C, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
			nn.ReLU(True),
			nn.Conv3d(C, output_chn, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
			nn.ReLU(True))

		self.weight_init()

	def weight_init(self):
		for m in self.modules():
			if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
				# nn.init.xavier_normal_(m.weight.data)
			elif isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight.data)
				if m.bias is not None:
					nn.init.normal_(m.bias.data)
	
	def forward(self, x):
		# batch x c x a x b x h
		# x = x.unsqueeze(1)
		x1 = self.block1(x)
		x2 = self.block2(x1)
		x3 = self.block3(x2)
		x4 = self.block4(x3)
		x5 = self.block5(x4)

		x1 = F.relu(self.comp1(x1))
		x2 = F.relu(self.comp2(x2))
		x3 = F.relu(self.comp3(x3))
		x4 = F.relu(self.comp4(x4))
		x5 = F.relu(self.comp5(x5))
		comp_concat = torch.cat((x1, x2, x3, x4, x5), dim=1)

		out = self.block6(comp_concat)
		out = nn.functional.softmax(out, dim=1)

		return out