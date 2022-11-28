import keras
import keras.losses
import keras.optimizers
import keras.layers as layers
import keras.losses as losses
import keras.callbacks as callbacks
import tensorflow as tf
from genericpath import exists
import os
import numpy as np

image_size = (224, 224)
downscale_factor = 4
embedding_length = 2048
embedding_dim = 4
d_model = 128
beta = 0.25
runeager = False
small_dataset = False
ds_size = 1024
batch_size = 8
test_size = batch_size
epochs = 1000

version = 1

img_output_path = f"generated-transformer-vqgan-v{version}"
log_dir = "logs/transformer_vqgan"
encoder_model = "models/vqgan_faces_encoder_v6.keras"

model_path = f"models/transformer_vqgan_v{version}.keras"

# Multi head attention

sequence_length = (image_size[0] // downscale_factor) * (image_size[1] // downscale_factor)

# Input = sequence * one-hot encoded
encoder_input = keras.Input(shape=(sequence_length, embedding_length))
x = input
# add positional embedding



def attention(q, k, v, d_attention_block):
  d_model = tf.shape(q)[-1]
  q = layers.Conv1D(filters=d_attention_block, kernel_size=d_model)(q)
  k = layers.Conv1D(filters=d_attention_block, kernel_size=d_model)(k)
  v = layers.Conv1D(filters=d_attention_block, kernel_size=d_model)(v)
  
  x = tf.matmul(q, k, transpose_b=True)

  x = layers.Multiply()([x, 1.0 / tf.sqrt(d_attention_block)])
  x = layers.Softmax()(x)
  x = layers.Multiply()([x, v])
  return x


def encoder_block(x, d_model, d_attention_block, num_heads):
  # self-attention
  residual = x

  xs = [attention(x, x, x, d_attention_block) for _ in range(num_heads)]
  x = layers.Concatenate()(xs)
  x = layers.Conv1D(filters=d_model, kernel_size=num_heads * d_attention_block)(x)

  # add and layer-norm
  x = layers.Add()([residual, x])
  x = layers.LayerNormalization()(x)

  # position-wise dense
  residual = x
  x = layers.Conv1D(filters=d_model, kernel_size=d_model)(x)

  #add and layer-norm
  x = layers.Add()([residual, x])
  x = layers.LayerNormalization()(x)

  return x


class VectorQuantizationIndexes(layers.Layer):
  def __init__(self, embedding_length, embedding_dim, beta=0.25, **kwargs):
    super(VectorQuantizationIndexes, self).__init__(**kwargs)
    self.embedding_length = embedding_length
    self.embedding_dim = embedding_dim
    self.beta = beta
    self.embedding = self.add_weight("embedding",
      shape=(embedding_length, embedding_dim), 
      initializer=tf.random_uniform_initializer(0, 1), 
      trainable=True)

  def call(self, input):
    (_, w, h, c) = input.shape
    B = tf.shape(input)[0]
    flat = tf.reshape(input, shape=(B * w * h, c))
    flat = tf.tile(flat, [1, self.embedding_length])
    flat = tf.reshape(flat, shape=(B * w * h, self.embedding_length, c))
    diff = tf.pow(flat - self.embedding, 2)
    diff = tf.reduce_sum(diff, axis=-1)
    embedding_indexes = tf.argmin(diff, axis=-1)
    embedding_indexes = tf.reshape(embedding_indexes, shape=(B, w, h))
    
    return embedding_indexes

  def get_config(self):
    config = super(VectorQuantizationIndexes, self).get_config()
    config.update({
      "embedding_length": self.embedding_length,
      "embedding_dim": self.embedding_dim,
      "beta": self.beta
    })
    return config

os.makedirs(img_output_path, exist_ok=True)

fname = os.path.join("list_attr_celeba.txt")

with open(fname) as f:
  data = f.read()

lines = data.split("\n")
header = lines[1].split()
lines = lines[2:-1]
raw_data = np.zeros((len(lines), len(header)))
for i, line in enumerate(lines):
  line_data = [int(x) for x in line.split()[1:]]
  raw_data[i, :] = line_data[:]


label_dataset = tf.data.Dataset.from_tensor_slices(raw_data)
image_dataset = keras.utils.image_dataset_from_directory(
  f"img_align_celeba{'_small' if small_dataset else ''}",
  label_mode=None,
  image_size=image_size,
  batch_size=1,
  smart_resize=True,
  shuffle=False
)
image_dataset = image_dataset.map(lambda x: x / (255. / 2) - 1.)

dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
dataset = dataset.take(ds_size).batch(batch_size)


    