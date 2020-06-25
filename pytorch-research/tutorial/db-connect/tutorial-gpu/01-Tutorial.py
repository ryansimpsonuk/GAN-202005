# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### Tutorial implementation with pytorch
# MAGIC advantages are that it is reasonably imperitive, however requires you to push everything to the GPU manually which requires working out of dependencies.
# MAGIC 
# MAGIC Steps to train:
# MAGIC The fundamental steps to train a GAN can be described as following:
# MAGIC 1. Sample a noise set and a real-data set, each with size m.
# MAGIC 1. Train the Discriminator on this data.
# MAGIC 1. Sample a different noise subset with size m.
# MAGIC 1. Train the Generator on this data.
# MAGIC 1. Repeat from Step 1.

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# COMMAND ----------



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

data = mnist_data(out_dir)
# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)

num_batches = len(data_loader)

data.to(device)
data_loader.to(device)


# COMMAND ----------

# Num batches
print(num_batches)

# COMMAND ----------

# Discriminator
discriminator = DiscriminatorNet()

# Discriminator
generator = GeneratorNet()

discriminator.to(device)
generator.to(device)
# ADAM as the optimization algorithm, learning rate of 0.0002
# loss function for this task will be Binary Cross Entropy Loss (BCE)
# - resembles the log-loss for both the Generator and Discriminator

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

loss = nn.BCELoss()


num_test_samples = 16
test_noise = noise(num_test_samples)

test_noise.to(device)
loss.to(device)

# COMMAND ----------

dbutils.fs.rm('/tmp/ryansimpson/dataset/data/images/VGAN/MNIST',True)
# Create logger instance
logger = Logger(model_name='VGAN', data_name='MNIST', data_path_root=out_dir, exp_name='gpu')

# COMMAND ----------

import time
num_epochs = 200

for epoch in range(num_epochs):
    start_time = time.time()
    for n_batch, (real_batch,_) in enumerate(data_loader):
        N = real_batch.size(0)
        # 1. Train Discriminator
        real_data = Variable(images_to_vectors(real_batch))
        real_data.to(device)

        # Generate fake data and detach 
        # (so gradients are not calculated for generator)
        fake_data = generator(noise(N).to(device)).detach()
        fake_data.to(device)
        
        # Train D
        d_error, d_pred_real, d_pred_fake = \
              train_discriminator(d_optimizer, discriminator, real_data, fake_data, device)
        
        d_error.cpu()
        d_pred_real.cpu()
        d_pred_fake.cpu()

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(N).to(device))
        fake_data.to(device)
        # Train G
        g_error = train_generator(g_optimizer, discriminator, fake_data, device)
        g_error.cpu() # must come back to the cpu?
        
        # Log batch error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)
        # Display Progress every few batches
        if (n_batch) % 600 == 0: 
            test_images = vectors_to_images(generator(test_noise.to(device)))

            test_images = test_images.data.cpu()
            logger.log_images(
                test_images, num_test_samples, 
                epoch, n_batch, num_batches
            );
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )
            
    duration = time.time() - start_time
    print(f"##### epoch duration : {duration}s #####")