import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision
import torchvision.datasets as dset
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from AI_brains import ArtistBrain, ArtCriticBrain, weights_init, doc


class Trainer:
	def __init__(self,
	             generator=torch.nn.modules.Sequential(), discriminator=torch.nn.modules.Sequential(),
	             device=torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu"), optimizer=torch.optim.Adam,
	             train_ratio=0.7, val_test_ratio=0.5, root="", weight_initializer=weights_init,
	             criterion=nn.BCELoss(), gpus=1, epochs=1, batch_size=1, seed_size=100, workers=2, img_size=32,
	             val_interval=1000, **optimizer_args):
		"""
		:param generator:
		:param discriminator:
		:param device:
		:param optimizer:
		:param train_ratio:
		:param val_test_ratio:
		:param dataset:
		:param weight_initializer:
		:param criterion:
		:param gpus:
		:param epochs:
		:param batch_size:
		:param seed_size:
		:param workers:
		:param val_interval:
		:param optimizer_args:
		"""
		self.generator_optimizer = None
		self.discriminator_optimizer = None
		self.generator = generator.to(device)
		self.discriminator = discriminator.to(device)
		self.device = device
		self.optimizer = optimizer
		self.optimizer_args = optimizer_args
		self.train_ratio = train_ratio
		self.val_test_ratio = val_test_ratio
		self.training_loader = None
		self.validation_loader = None
		self.testing_loader = None
		self.weight_initializer = weight_initializer
		self.criterion = criterion
		self.gpus = gpus
		self.epochs = epochs
		self.batch_size = batch_size
		self.seed_size = seed_size
		self.workers = workers
		self.val_interval = val_interval
		self.lowest_generator_loss = 100
		self.lowest_discriminator_loss = 100
		
		self.dataset = dset.ImageFolder(
			root=root,
			transform=transforms.Compose([
				transforms.Resize(img_size),
				transforms.CenterCrop(img_size),
				transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			])
		)
		
		self.real_label = 1
		self.fake_label = 0
	
	def compile_models(self):
		# setup gpus if multiple gpus are available
		if (self.device.type == "cuda") and (self.gpus > 1):
			self.generator = torch.nn.DataParallel(self.generator, list(range(1)))
			self.discriminator = torch.nn.DataParallel(self.discriminator, list(range(1)))
		
		# Initialize weights
		self.generator.apply(fn=self.weight_initializer)
		self.discriminator.apply(fn=self.weight_initializer)
		
		# Initialize optimizers
		self.generator_optimizer = self.optimizer(
			params=self.generator.parameters(), lr=self.optimizer_args["lr"], betas=self.optimizer_args["betas"])
		self.discriminator_optimizer = self.optimizer(
			params=self.discriminator.parameters(), lr=self.optimizer_args["lr"], betas=self.optimizer_args["betas"])
		
		# Configure Dataloader
		total_length = len(self.dataset)
		training_length = int(total_length * self.train_ratio)
		validation_length = int((total_length - training_length) * self.val_test_ratio)
		test_length = int((total_length - training_length) * (1 - self.val_test_ratio))
		training_length += total_length - training_length - validation_length - test_length
		print("Creating datasets")
		training_dataset, validation_dataset, testing_dataset = torch.utils.data.random_split(
			self.dataset, [training_length, validation_length, test_length])
		print("Datasets created")
		self.training_loader = torch.utils.data.DataLoader(
			training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers, drop_last=True)
		self.validation_loader = torch.utils.data.DataLoader(
			validation_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers, drop_last=True)
		self.testing_loader = torch.utils.data.DataLoader(
			testing_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers, drop_last=True)
	
	def __train__(self, data):
		"""
		:param data:
		:return:
		"""
		'''*****************************************[DISCRIMINATOR TRAINING]*****************************************'''
		self.discriminator.train()
		# Discriminator with all-real batch
		self.discriminator.zero_grad()
		label = torch.full((self.batch_size,), self.real_label, dtype=torch.float, device=self.device)
		output = self.discriminator(data[0].to(self.device))
		discriminator_error_real = self.criterion(output, label)
		discriminator_error_real.backward()
		
		# Discriminator with all-fake batch
		fake_results = self.generator(torch.randn(self.batch_size, self.seed_size, 1, 1, device=self.device))
		label.fill_(self.fake_label)
		output = self.discriminator(fake_results.detach()).view(-1)
		discriminator_error_fake = self.criterion(output, label)
		discriminator_error_fake.backward()
		
		# Combine real and fake results from Discriminator
		discriminator_error = discriminator_error_fake + discriminator_error_real
		self.discriminator_optimizer.step()
		
		'''*******************************************[GENERATOR TRAINING]*******************************************'''
		self.generator.train()
		self.generator.zero_grad()
		label.fill_(self.real_label)
		output = self.discriminator(fake_results).view(-1)
		generator_error = self.criterion(output, label)
		generator_error.backward()
		self.generator_optimizer.step()
		
		return generator_error.item(), discriminator_error.item()
	
	def __validation__(self, data):
		"""
		:param data:
		:return:
		"""
		'''*****************************************[DISCRIMINATOR VALIDATOR]*****************************************'''
		self.discriminator.eval()
		# Discriminator with all-real batch
		self.discriminator.zero_grad()
		label = torch.full((self.batch_size,), self.real_label, dtype=torch.float, device=self.device)
		output = self.discriminator(data[0].to(self.device))
		discriminator_error_real = self.criterion(output, label)
		
		# Discriminator with all-fake batch
		fake_results = self.generator(torch.randn(self.batch_size, self.seed_size, 1, 1, device=self.device))
		label.fill_(self.fake_label)
		output = self.discriminator(fake_results.detach()).view(-1)
		discriminator_error_fake = self.criterion(output, label)
		
		# Combine real and fake results from Discriminator
		discriminator_error = discriminator_error_fake + discriminator_error_real
		
		'''*******************************************[GENERATOR VALIDATOR]*******************************************'''
		self.generator.eval()
		self.generator.zero_grad()
		label.fill_(self.real_label)
		output = self.discriminator(fake_results).view(-1)
		generator_error = self.criterion(output, label)
		
		return generator_error.item(), discriminator_error.item()
	
	def train(self):
		counter = 0
		save_iter = 0
		
		validating_generator_loss = []
		training_generator_loss = []
		validating_discriminator_loss = []
		training_discriminator_loss = []
		
		len_training = int(len(self.training_loader.dataset) / self.batch_size)
		len_validation = int(len(self.validation_loader.dataset) / self.batch_size)
		
		epoch_progress_bar = tqdm(total=self.epochs, desc="[EPOCH]: ")
		training_progress_bar = tqdm(total=len_training, desc="[TRAINING]: ")
		validation_progress_bar = tqdm(total=len_validation, desc="[VALIDATION]: ")
		validation_progress_bar.set_postfix({
			"Generator Train Loss": 0,
			"Generator Valid Loss": 0,
			"Discriminator Train Loss": 0,
			"Discriminator Valid Loss": 0
		})
		for epoch in range(self.epochs):
			for training_batch in self.training_loader:
				train_gen_loss, train_dis_loss = self.__train__(data=training_batch)
				training_generator_loss.append(train_gen_loss)
				training_discriminator_loss.append(train_dis_loss)
				counter += 1
				if counter > self.val_interval:
					counter = 0
					save_iter += 1
					for validating_batch in self.validation_loader:
						val_gen_loss, val_dis_loss = self.__validation__(data=validating_batch)
						validating_generator_loss.append(val_gen_loss)
						validating_discriminator_loss.append(val_dis_loss)
						validation_progress_bar.update()
					# Display Loss Metrics and reset them
					
					validation_progress_bar.set_postfix({
						"Generator Train Loss": np.asarray(training_generator_loss).mean(),
						"Generator Valid Loss": np.asarray(validating_generator_loss).mean(),
						"Discriminator Train Loss": np.asarray(training_discriminator_loss).mean(),
						"Discriminator Valid Loss": np.asarray(validating_discriminator_loss).mean()
					})
					# Saving best model
					if self.lowest_generator_loss > np.asarray(validating_generator_loss).mean():
						self.lowest_generator_loss = np.asarray(validating_generator_loss).mean()
						torch.save(
							{
								"epoch": epoch,
								'model_state_dict': self.generator.state_dict(),
								'optimizer_state_dict': self.generator_optimizer.state_dict(),
								'loss': self.lowest_generator_loss,
							},
							"/home/nmelgiri/PycharmProjects/Fatima/Models/Generators/model_2_{}.pt".format(save_iter)
						)
					if self.lowest_discriminator_loss > np.asarray(validating_discriminator_loss).mean():
						self.lowest_discriminator_loss = np.asarray(validating_discriminator_loss).mean()
						torch.save(
							{
								"epoch": epoch,
								'model_state_dict': self.discriminator.state_dict(),
								'optimizer_state_dict': self.discriminator_optimizer.state_dict(),
								'loss': self.lowest_discriminator_loss,
							},
							"/home/nmelgiri/PycharmProjects/Fatima/Models/Discriminators/model_2_{}.pt".format(save_iter)
						)
					validating_generator_loss = []
					training_generator_loss = []
					validating_discriminator_loss = []
					training_discriminator_loss = []
					
				validation_progress_bar.reset()
				training_progress_bar.update()
			training_progress_bar.reset()
			epoch_progress_bar.update()
		epoch_progress_bar.reset()


if __name__ == '__main__':
	artist = ArtistBrain()
	art_critic = ArtCriticBrain()
	
	trainer = Trainer(
		root=doc["etc"]["Data Path"],
		generator=artist,
		discriminator=art_critic,
		weight_initializer=weights_init,
		lr=doc["Training"]["lr"],
		betas=(doc["Training"]["beta"], 0.999),
		epochs=doc["Training"]["epochs"],
		batch_size=doc["Training"]["batch size"],
		val_interval=doc["Training"]["validation interval"],
		img_size=doc["Images"]["size"],
		workers=doc["Training"]["workers"]
	)
	trainer.compile_models()
	print("Compiling complete")
	trainer.train()
