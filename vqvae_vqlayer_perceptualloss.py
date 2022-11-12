import datetime
from genericpath import exists
import keras
import keras.losses
import keras.optimizers
import keras.layers as layers
import keras.losses as losses
import keras.callbacks as callbacks
from keras.applications.convnext import ConvNeXtTiny
import tensorflow as tf

image_size = (224, 224)
embedding_length = 512
embedding_dim = 1
beta = 0.25
runeager = False

version = 5
model_path = f"vqvae_faces_v{version}.keras"


convnext = ConvNeXtTiny(weights='imagenet')
input_layer = convnext.get_layer("input_1")
perception_layer_names = [
  #"convnext_tiny_stage_0_block_0_depthwise_conv",
  "convnext_tiny_stage_0_block_0_identity",
  #"convnext_tiny_stage_0_block_1_depthwise_conv",
  "convnext_tiny_stage_0_block_1_identity",
  #"convnext_tiny_stage_0_block_2_depthwise_conv",
  "convnext_tiny_stage_0_block_2_identity",

  #"convnext_tiny_stage_1_block_0_depthwise_conv",
  "convnext_tiny_stage_1_block_0_identity",
  #"convnext_tiny_stage_1_block_1_depthwise_conv",
  "convnext_tiny_stage_1_block_1_identity",
  #"convnext_tiny_stage_1_block_2_depthwise_conv",
  "convnext_tiny_stage_1_block_2_identity",
  
  #"convnext_tiny_stage_2_block_0_depthwise_conv",
  "convnext_tiny_stage_2_block_0_identity",
  #"convnext_tiny_stage_2_block_1_depthwise_conv",
  "convnext_tiny_stage_2_block_1_identity",
  #"convnext_tiny_stage_2_block_2_depthwise_conv",
  "convnext_tiny_stage_2_block_2_identity",
]
perception_layers = [convnext.get_layer(name) for name in perception_layer_names]
perception_layer_outputs = [layer.output for layer in perception_layers]
perception_activation = keras.models.Model(input_layer.input, perception_layer_outputs)
perception_activation.summary()

def perception_loss_fn(original, generated):
  original_activation = perception_activation(original)
  generated_activation = perception_activation(generated)
  diffs = [tf.reduce_mean((tf.nn.l2_normalize(x, axis=-1) - tf.nn.l2_normalize(x0, axis=-1)) ** 2, axis=None) for (x, x0) in zip(original_activation, generated_activation)]
  loss = sum(diffs) / len(diffs)
  return loss

class VectorQuantization(layers.Layer):
  def __init__(self, embedding_length, embedding_dim, **kwargs):
    super(VectorQuantization, self).__init__(**kwargs)
    self.embedding_length = embedding_length
    self.embedding_dim = embedding_dim
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
    quantized_vectors = tf.gather(self.embedding, embedding_indexes)

    embedding_loss = tf.reduce_mean((tf.stop_gradient(input) - quantized_vectors) ** 2)
    encoding_loss = tf.reduce_mean((input - tf.stop_gradient(quantized_vectors)) ** 2)
    self.add_loss(embedding_loss + beta * encoding_loss)

    # Straight through estimator
    quantized_vectors = input + tf.stop_gradient(quantized_vectors - input)
    return quantized_vectors

  def get_config(self):
    config = super(VectorQuantization, self).get_config()
    config.update({
      "embedding_length": self.embedding_length,
      "embedding_dim": self.embedding_dim
    })
    return config

def conv_block(filters, x):
  x = layers.Conv2D(filters, kernel_size=3, padding="same", activation="relu", strides=2)(x)
  x = layers.Dropout(0.2)(x)
  x = layers.Conv2D(filters, kernel_size=3, padding="same", activation="relu")(x)
  return x

def decoder_loss_fn(actual_images, predicted):
  return (actual_images - predicted) ** 2

def get_model():
  if exists(model_path):
    return keras.models.load_model(model_path, custom_objects={
      "VectorQuantization": VectorQuantization,
      "decoder_loss_fn": decoder_loss_fn,
      "perception_loss_fn": perception_loss_fn
    })
    
  encoder_inputs = keras.Input(shape=(*image_size, 3))
  x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(encoder_inputs)
  x = conv_block(128, x)
  x = conv_block(256, x)
  encoder_output = layers.Conv2D(embedding_dim, kernel_size=3, padding="same")(x)

  encoder = keras.models.Model(encoder_inputs, encoder_output, name="encoder")

  decoder_input = keras.Input(shape=(image_size[0]//4, image_size[1]//4, embedding_dim))
  x = layers.Conv2D(256, 3, activation="relu", padding="same")(decoder_input)
  x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
  x = layers.Conv2DTranspose(128, 4, strides=2, padding="same")(x)
  x = layers.Conv2DTranspose(64, 4, strides=2, padding="same")(x)
  decoder_output = layers.Conv2D(3, 3, padding="same")(x)

  decoder = keras.models.Model(decoder_input, decoder_output, name="decoder")

  assert encoder.output.shape == decoder.input.shape

  vector_quant = VectorQuantization(embedding_length, embedding_dim, name="vector_quantization")

  x = encoder(encoder_inputs)
  x = vector_quant(x)
  x = decoder(x)
  model = keras.models.Model(encoder_inputs, x)
  return model

model = get_model()
model.summary()

model.compile(run_eagerly=runeager, loss=perception_loss_fn)

dataset = keras.utils.image_dataset_from_directory(
  "img_align_celeba_small",
  label_mode=None,
  image_size=image_size,
  batch_size=8,
  smart_resize=True,
  shuffle=False
  )

class VQVaeMonitor(keras.callbacks.Callback):
  def __init__(self, test_ds):
    self.test_ds = test_ds

  def on_epoch_end(self, epoch, logs=None):
    autoencoded = self.model(self.test_ds)
    autoencoded.numpy()
    for i in range(len(autoencoded)):
      img = keras.utils.array_to_img(autoencoded[i])
      img.save(f"generated-vqvae2/autoencoded_{epoch:03d}_{i}.png")

dataset = dataset.take(30000)
test_size = 5
dataset = dataset.map(lambda x: x / (255. / 2) - 1.)
test_ds = dataset.take(test_size).as_numpy_iterator().next()[:test_size]
dataset = dataset.skip(test_size)
for i in range(len(test_ds)):
  img = keras.utils.array_to_img(test_ds[i])
  img.save(f"generated-vqvae2/aaref_{i}.png")

# log_dir = "logs/dbg"
# tf.debugging.experimental.enable_dump_debug_info(
#     log_dir,
#     tensor_debug_mode="NO_TENSOR",
#     circular_buffer_size=-1)

callbacks = [
  VQVaeMonitor(test_ds),
  callbacks.ModelCheckpoint(model_path)
  # keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]
autolabeled_ds = tf.data.Dataset.zip((dataset, dataset)).shuffle(100)
model.fit(autolabeled_ds, epochs=400, callbacks=callbacks)