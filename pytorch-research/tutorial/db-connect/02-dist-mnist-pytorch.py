# Databricks notebook source
# MAGIC %md # Distributed deep learning training using PyTorch with HorovodRunner for MNIST
# MAGIC This notebook demonstrates how to migrate a single-node deep learning (DL) code with PyTorch to distributed training code with Horovod on Databricks with HorovodRunner.
# MAGIC 
# MAGIC This guide consists of the following sections:
# MAGIC 1. Prepare Single-Node Code
# MAGIC 2. Migrate to HorovodRunner
# MAGIC 
# MAGIC **Note:**
# MAGIC The notebook runs without code changes on CPU or GPU-enabled Databricks clusters.
# MAGIC To run the notebook, create a cluster with **2 workers**. 

# COMMAND ----------

# MAGIC %md ## Preparing Deep Learning Storage
# MAGIC 
# MAGIC We recommend save training data under `dbfs:/ml`, which maps to `file:/dbfs/ml` on driver and worker nodes.

# COMMAND ----------

PYTORCH_DIR = '/dbfs/ml/horovod_pytorch'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Single-Node Code
# MAGIC 
# MAGIC First you need to have working single-node PyTorch code. This is modified from   [Horovod's PyTorch MNIST Example](https://github.com/uber/horovod/blob/master/examples/pytorch_mnist.py).

# COMMAND ----------

# MAGIC %md ####Defining a simple Convolutional Network

# COMMAND ----------

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# COMMAND ----------

# MAGIC %md ####Configuring single-node training

# COMMAND ----------

# Setting training parameters
batch_size = 100
num_epochs = 5
momentum = 0.5
log_interval = 100

# COMMAND ----------

def train_one_epoch(model, device, data_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader) * len(data),
                100. * batch_idx / len(data_loader), loss.item()))

# COMMAND ----------

# DBTITLE 1,Preparing log directory
from time import time
import os

LOG_DIR = os.path.join(PYTORCH_DIR, str(time()), 'MNISTDemo')
os.makedirs(LOG_DIR)

# COMMAND ----------

# DBTITLE 1,Defining a method for checkpointing and persisting model
def save_checkpoint(model, optimizer, epoch):
  filepath = LOG_DIR + '/checkpoint-{epoch}.pth.tar'.format(epoch=epoch)
  state = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
  }
  torch.save(state, filepath)

# COMMAND ----------

# DBTITLE 1,Running single-node training with PyTorch
import torch.optim as optim
from torchvision import datasets, transforms

def train(learning_rate):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  train_dataset = datasets.MNIST(
    'data', 
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
  data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  model = Net().to(device)

  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

  for epoch in range(1, num_epochs + 1):
    train_one_epoch(model, device, data_loader, optimizer, epoch)
    save_checkpoint(model, optimizer, epoch)

# COMMAND ----------

train(learning_rate = 0.001)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Migrate to HorovodRunner
# MAGIC 
# MAGIC HorovodRunner takes a Python method that contains DL training code w/ Horovod hooks. This method gets pickled on the driver and sent to Spark workers.  A Horovod MPI job is embedded as a Spark job using barrier execution mode.

# COMMAND ----------

import horovod.torch as hvd
from sparkdl import HorovodRunner

# COMMAND ----------

def train_hvd(learning_rate):
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

# MAGIC %md
# MAGIC With the function `run_training_horovod` defined previously with Horovod hooks, you can easily build the `HorovodRunner` and run distributed training.

# COMMAND ----------

hr = HorovodRunner(np=2) # We assume cluster consists of two workers.
hr.run(train_hvd, learning_rate = 0.001)

# COMMAND ----------

# MAGIC %md 
# MAGIC Under the hood, HorovodRunner takes a Python method that contains deep learning training code with Horovod hooks. This method gets pickled on the driver and sent to Spark workers. A Horovod MPI job is embedded as a Spark job using the barrier execution mode. The first executor collects the IP addresses of all task executors using BarrierTaskContext and triggers a Horovod job using `mpirun`. Each Python MPI process loads the pickled user program back, deserializes it, and runs it.
# MAGIC 
# MAGIC For further information on HorovodRunner API, please refer to the [documentation](https://databricks.github.io/spark-deep-learning/docs/_site/api/python/index.html#sparkdl.HorovodRunner). Note that you can use `np=-1` to spawn a subprocess on the driver node for quicker development cycle.
# MAGIC ```
# MAGIC hr = HorovodRunner(np=-1)
# MAGIC hr.run(run_training)
# MAGIC ```