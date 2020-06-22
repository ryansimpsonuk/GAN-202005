# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### Introduction
# MAGIC Some very cool applications here: https://github.com/nashory/gans-awesome-applications
# MAGIC 
# MAGIC Following sample taken from Mosquera, D.G., 2018 (https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f) Accessed May 2020
# MAGIC 
# MAGIC Some key take away's (Mosquera, D.G., 2018):
# MAGIC - belong to the familiy of generative models in the field of unsupervised learning
# MAGIC   - other Generative Models in this family include Variational Autoencoders, (VAE), pixelCNN, PixelRNN and realNVP
# MAGIC - Generative Models learn the intrinsic distribution function of the input data (p(x), p(x,y...)), rather than learning to map a function to the data (as in supervised learning)
# MAGIC 
# MAGIC Some call-out advantages/disadvantages (Goodfellow, 2018) (https://www.youtube.com/watch?v=HGYYEUSm-0Q):
# MAGIC - can generate some very sharp images
# MAGIC - no statistical inference is required so easy to train using back-propogation to obtain gradients
# MAGIC 
# MAGIC Disadvantages:
# MAGIC - can be difficult to stabalise the training (for example, consider modal instability)
# MAGIC - GAN's belong to the direct implicit density models so they can model p(x) without explicitly defining the Probablity Distribution Function
# MAGIC 
# MAGIC ### GAN Structure
# MAGIC Original diagram here
# MAGIC - training with a neural network means a gradient based approach can be taken
# MAGIC 
# MAGIC Steps to train:
# MAGIC The fundamental steps to train a GAN can be described as following:
# MAGIC 1. Sample a noise set and a real-data set, each with size m.
# MAGIC 1. Train the Discriminator on this data.
# MAGIC 1. Sample a different noise subset with size m.
# MAGIC 1. Train the Generator on this data.
# MAGIC 1. Repeat from Step 1.

# COMMAND ----------

# MAGIC %run ./00-utils

# COMMAND ----------

# working with MLRuntime 6.5 so have tensorFlow and Pytorch already available

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms,datasets

# COMMAND ----------

# note the use of the dbfs prefix here to allow us to use local file APIs to talk through the fuze mount:
# https://docs.databricks.com/data/databricks-file-system.html#fuse
dbutils.fs.mkdirs('/tmp/ryansimpson/dataset')

# COMMAND ----------

def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.Normalize((.5, .5, .5), (.5, .5, .5)) # expecting a gray-scale image? - see https://github.com/yunjey/pytorch-tutorial/issues/161#issuecomment-574908584
         transforms.Normalize([.5,], [.5,]) # expecting a gray-scale image?
         
        ])
    out_dir = '/dbfs/tmp/ryansimpson/dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)
# Load data
data = mnist_data()
# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
num_batches = len(data_loader)

# COMMAND ----------

# DBTITLE 1,Discriminator
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
      
discriminator = DiscriminatorNet()

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
      
generator = GeneratorNet()

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

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

loss = nn.BCELoss()

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

def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, ones_target(N) )
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

# COMMAND ----------

def train_generator(optimizer, fake_data):
    N = fake_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, ones_target(N))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

# COMMAND ----------

num_test_samples = 16
test_noise = noise(num_test_samples)

# COMMAND ----------

# Create logger instance
logger = Logger(model_name='VGAN', data_name='MNIST', data_path_root='/dbfs/tmp/ryansimpson/dataset')
# Total number of epochs to train
num_epochs = 200
for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(data_loader):
        N = real_batch.size(0)
        # 1. Train Discriminator
        real_data = Variable(images_to_vectors(real_batch))
        # Generate fake data and detach 
        # (so gradients are not calculated for generator)
        fake_data = generator(noise(N)).detach()
        # Train D
        d_error, d_pred_real, d_pred_fake = \
              train_discriminator(d_optimizer, real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(N))
        # Train G
        g_error = train_generator(g_optimizer, fake_data)
        # Log batch error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)
        # Display Progress every few batches
        if (n_batch) % 100 == 0: 
            test_images = vectors_to_images(generator(test_noise))
            test_images = test_images.data
            logger.log_images(
                test_images, num_test_samples, 
                epoch, n_batch, num_batches
            );
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )