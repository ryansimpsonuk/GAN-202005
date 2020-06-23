# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### ReadMe
# MAGIC adapting tutorial code to implement distributed approach detailed here: https://docs.databricks.com/_static/notebooks/deep-learning/mnist-pytorch.html

# COMMAND ----------

# MAGIC %run ./utils/Helpers

# COMMAND ----------

# MAGIC %run ./utils/Logger

# COMMAND ----------

# working with MLRuntime 6.5 so have tensorFlow and Pytorch already available

# seems the client environment must match the target environment
# so make sure these are installed locally:
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms,datasets

# environment setup for databricks connect:
# https://docs.databricks.com/dev-tools/databricks-connect.html

# install requirements using 
# pip install -r pytorch-research/tutorial/db-connect/requirements.txt

# ensure you've run source ~/db-connect.sh before running pyspark

# session set-up:
# from pyspark.dbutils import DBUtils
# dbutils = DBUtils(spark.sparkContext)

dbfs_out_dir = '/tmp/ryansimpson/dataset'
out_dir = '/dbfs' + dbfs_out_dir

dbutils.tensorboard.start(out_dir + '/runs')

if dbutils.fs.mkdirs(dbfs_out_dir):
    print(f"{dbfs_out_dir} already exists")

# would be nice to be able to add these without a concrete path:
# sc.addPyFile("./pytorch-research/tutorial/db-connect/utils/Helpers.py")
# sc.addPyFile("./pytorch-research/tutorial/db-connect/utils/Logger.py")


# import Helpers
# import Logger

data = mnist_data(out_dir)
# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
num_batches = len(data_loader)

# COMMAND ----------

# Discriminator
discriminator = DiscriminatorNet()

# Discriminator
generator = GeneratorNet()


# ADAM as the optimization algorithm, learning rate of 0.0002
# loss function for this task will be Binary Cross Entropy Loss (BCE)
# - resembles the log-loss for both the Generator and Discriminator

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

loss = nn.BCELoss()

num_test_samples = 16
test_noise = noise(num_test_samples)

# COMMAND ----------

# DBTITLE 1,function to train a single epoch
# Create logger instance
logger = Logger(model_name='VGAN', data_name='MNIST', data_path_root=out_dir)


def train_one_epoch(discriminator, generator, d_optimizer, g_optimizer, data_loader, epoch):
  
  for n_batch, (real_batch,_) in enumerate(data_loader):
        N = real_batch.size(0)
        # 1. Train Discriminator
        real_data = Variable(images_to_vectors(real_batch))
        # Generate fake data and detach 
        # (so gradients are not calculated for generator)
        fake_data = generator(noise(N)).detach()
        # Train D
        d_error, d_pred_real, d_pred_fake = \
              train_discriminator(d_optimizer, discriminator, real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(N))
        # Train G
        g_error = train_generator(g_optimizer, discriminator, fake_data)
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

# COMMAND ----------

# DBTITLE 1,Method for checkpointing and persisting the model
from time import time
import os

LOG_DIR = os.path.join(out_dir, str(time()), 'MNISTDemo')
os.makedirs(LOG_DIR)

def save_checkpoint(discriminator, generator, d_optimizer, g_optimizer, epoch):
  filepath = LOG_DIR + '/checkpoint-{epoch}.pth.tar'.format(epoch=epoch)
  state = {
    'discriminator': discriminator.state_dict(),
    'generator' : generator.state_dict(),
    'd_optimizer': d_optimizer.state_dict(),
    'g_optimizer': g_optimizer.state_dict(),
  }
  torch.save(state, filepath)

# COMMAND ----------

# DBTITLE 1,check with single node training
num_epochs=200

for epoch in range(1, num_epochs + 1):
    train_one_epoch(discriminator, generator, d_optimizer, g_optimizer, data_loader, epoch)
    save_checkpoint(discriminator, generator, d_optimizer, g_optimizer, epoch)

# COMMAND ----------

# DBTITLE 1,Add Horovod Runner [wip]
import horovod.torch as hvd
from sparkdl import HorovodRunner

def train_hvd():
  hvd.init()  # Initialize Horovod.
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  if device.type == 'cuda':
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())

  train_dataset = datasets.MNIST(
    root='data-%d'% hvd.rank(),  # Use different root directory for each worker to avoid race conditions.
    train=True, 
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
  )

  from torch.utils.data.distributed import DistributedSampler
  
  # Configure the sampler such that each worker obtains a distinct sample of input dataset.
  train_sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
  # Use trian_sampler to load a different sample of data on each worker.
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

  model = Net().to(device)
  
  # Effective batch size in synchronous distributed training is scaled by the number of workers.
  # An increase in learning rate compensates for the increased batch size.
  optimizer = optim.SGD(model.parameters(), lr=learning_rate * hvd.size(), momentum=momentum)

  # Wrap the optimizer with Horovod's DistributedOptimizer.
  optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
  
  # Broadcast initial parameters so all workers start with the same parameters.
  hvd.broadcast_parameters(model.state_dict(), root_rank=0)

  for epoch in range(1, num_epochs + 1):
    train_one_epoch(model, device, train_loader, optimizer, epoch)
    # Only save checkpoints on the first worker.
    if hvd.rank() == 0:
      save_checkpoint(model, optimizer, epoch)


# COMMAND ----------


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
              train_discriminator(d_optimizer, discriminator, real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(N))
        # Train G
        g_error = train_generator(g_optimizer, discriminator, fake_data)
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