# Databricks notebook source
dbutils.fs.ls('/tmp/ryansimpson/dataset/runs/')

# COMMAND ----------

dbutils.tensorboard.start('/dbfs/tmp/ryansimpson/dataset/runs/')