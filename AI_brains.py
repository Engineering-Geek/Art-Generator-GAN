# %matplotlib inline
import random

import torch
import torch.nn as nn
import torch.utils.data
import yaml

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

doc = yaml.safe_load(open("data.yaml", "r"))
img_size = doc["Images"]["size"]
nz = doc["Images"]["seed size"]
batch_size = doc["Training"]["batch size"]
workers = 2
ngpu = doc["Training"]["gpu"]
ngf = doc["Neural Networks"]["number of generator features"]
ndf = doc["Neural Networks"]["number of discriminator features"]
nc = 3
num_epochs = doc["Training"]["epochs"]
lr = doc["Training"]["lr"]
beta = doc["Training"]["beta"]


# custom weights initialization called on netG and netD
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)
		
		
class Flatten(torch.nn.Module):
	def forward(self, x):
		return x.view(x.shape[0], -1)


class ArtistBrain(nn.modules.Module):
	def __init__(self):
		super(ArtistBrain, self).__init__()
		
		self.relu = nn.ReLU(True)
		self.conv_transpose_1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
		self.batch_norm_1 = nn.BatchNorm2d(ngf * 8)
		self.conv_transpose_2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
		self.batch_norm_2 = nn.BatchNorm2d(ngf * 4)
		self.conv_transpose_3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
		self.batch_norm_3 = nn.BatchNorm2d(ngf * 2)
		self.conv_transpose_4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
		self.batch_norm_4 = nn.BatchNorm2d(ngf)
		self.conv_transpose_final = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)
		self.final_activation = nn.Tanh()
	
	def forward(self, random_vector):
		
		# First Convolution Transposition
		layer_1 = self.conv_transpose_1(random_vector)
		layer_1 = self.batch_norm_1(layer_1)
		layer_1 = self.relu(layer_1)
		
		# Second Convolution Transposition
		layer_2 = self.conv_transpose_2(layer_1)
		layer_2 = self.batch_norm_2(layer_2)
		layer_2 = self.relu(layer_2)
		
		# Third Convolution Transposition
		layer_3 = self.conv_transpose_3(layer_2)
		layer_3 = self.batch_norm_3(layer_3)
		layer_3 = self.relu(layer_3)
		
		# Fourth Convolution Transposition
		layer_4 = self.conv_transpose_4(layer_3)
		layer_4 = self.batch_norm_4(layer_4)
		layer_4 = self.relu(layer_4)

		# Final Convolution Transposition
		final_layer = self.conv_transpose_final(layer_4)
		output = self.final_activation(final_layer)
		
		return output


class ArtCriticBrain(nn.modules.Module):
	def __init__(self):
		super(ArtCriticBrain, self).__init__()
		self.relu = nn.LeakyReLU(0.2, inplace=True)
		self.conv_1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
		# state size. (ndf) x 32 x 32
		self.conv_2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
		self.batch_norm_2 = nn.BatchNorm2d(ndf * 2)
		# state size. (ndf*2) x 16 x 16
		self.conv_3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
		self.batch_norm_3 = nn.BatchNorm2d(ndf * 4)
		# state size. (ndf*4) x 8 x 8
		self.conv_4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
		self.batch_norm_4 = nn.BatchNorm2d(ndf * 8)
		# state size. (ndf*8) x 4 x 4
		self.conv_final = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
		sample_output = self.conv_final(torch.randn((1, ndf*8, 4, 4)))
		print(sample_output.shape)
		self.flatten = Flatten()
		self.final_linear = nn.Linear(in_features=len(self.flatten(sample_output)), out_features=1)
		print(self.final_linear)
		self.final_activation = nn.Sigmoid()
	
	def forward(self, image):
		layer_1 = self.conv_1(image)
		layer_1 = self.relu(layer_1)
		
		layer_2 = self.conv_2(layer_1)
		layer_2 = self.batch_norm_2(layer_2)
		layer_2 = self.relu(layer_2)
		
		layer_3 = self.conv_3(layer_2)
		layer_3 = self.batch_norm_3(layer_3)
		layer_3 = self.relu(layer_3)
		
		layer_4 = self.conv_4(layer_3)
		layer_4 = self.batch_norm_4(layer_4)
		layer_4 = self.relu(layer_4)
		
		final_conv = self.conv_final(layer_4)
		print(final_conv.shape)
		flattened_layer = self.flatten(final_conv)
		output = self.final_linear(flattened_layer)
		output = self.final_activation(output).reshape(-1)
		
		return output
