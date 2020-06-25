# Databricks notebook source
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms,datasets

# COMMAND ----------

# DBTITLE 1,Discriminator
# note the use of the dbfs prefix here to allow us to use local file APIs to talk through the fuze mount:
# https://docs.databricks.com/data/databricks-file-system.html#fuse

'''    out_dir = '/dbfs/tmp/ryansimpson/dataset'
'''
def mnist_data(out_dir):
    compose = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.Normalize((.5, .5, .5), (.5, .5, .5)) # expecting a gray-scale image? - see https://github.com/yunjey/pytorch-tutorial/issues/161#issuecomment-574908584
         transforms.Normalize([.5,], [.5,]) # expecting a gray-scale image?         
        ])

    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784
        n_out = 1
        
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
      

# COMMAND ----------

def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

# COMMAND ----------

# DBTITLE 1,Generator
class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 784
        
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        
        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
      

# COMMAND ----------

# we need some noise (Generators sample from noise, discriminators sample from the data)
def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 100))
    
    return n

# COMMAND ----------

# ADAM as the optimization algorithm, learning rate of 0.0002
# loss function for this task will be Binary Cross Entropy Loss (BCE)
# - resembles the log-loss for both the Generator and Discriminator

# try to score real images as 1 and fake images as 0:
def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data

# COMMAND ----------

def train_discriminator(optimizer, discriminator,real_data, fake_data, device):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data.to(device))
    prediction_real.to(device)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, ones_target(N).to(device) )
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data.to(device))
    prediction_fake.to(device)
    
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N).to(device))
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

# COMMAND ----------

def train_generator(optimizer, discriminator, fake_data, device):
    N = fake_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data.to(device))
    prediction.to(device)
    # Calculate error and backpropagate
    error = loss(prediction, ones_target(N).to(device))
    error.to(device)
    
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error


# COMMAND ----------

