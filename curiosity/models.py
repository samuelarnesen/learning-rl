import sys, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicCNN(nn.Module):

	def __init__(self, in_channels, eligible_actions):
		super().__init__()

		out_channels_one = 64
		out_channels_two = 128
		out_channels_three = 256

		self.conv_layer_one = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_one, kernel_size=8)
		self.pooling_layer_one = nn.AvgPool2d(kernel_size=4)
		#self.batch_norm_layer_one = nn.BatchNorm2d()

		self.conv_layer_two = nn.Conv2d(in_channels=out_channels_one, out_channels=out_channels_two, kernel_size=8)
		self.pooling_layer_two = nn.AvgPool2d(kernel_size=4)

		self.conv_layer_three = nn.Conv2d(in_channels=out_channels_two, out_channels=out_channels_three, kernel_size=4)
		self.pooling_layer_three = nn.AvgPool2d(kernel_size=4)

		self.linear_layer = nn.Linear(out_channels_three, eligible_actions)

	def forward(self, images):
		image_tensor = images if torch.is_tensor(images) else torch.tensor(images).squeeze().unsqueeze(0)
		output_one = F.relu(self.pooling_layer_one(self.conv_layer_one(images)))
		output_two = F.relu(self.pooling_layer_two(self.conv_layer_two(output_one)))
		output_three = F.relu(self.pooling_layer_three(self.conv_layer_three(output_two)))
		final_output = self.linear_layer(output_three.squeeze(-1).squeeze(-1))
		return final_output

	def act(self, images):
		output = self(images)
		return torch.argmax(output, dim=1)

	def evaluate(self, images, actions=None):
		output = self(images)
		if actions != None:
			return torch.tensor(np.asarray([output[i, action] for i, action in enumerate(actions)]))
		return torch.max(output, dim=1)
