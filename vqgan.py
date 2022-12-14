import os
import keras
import keras.losses
import keras.optimizers
import keras.layers as layers
import keras.losses as losses
import keras.callbacks as callbacks
import tensorflow as tf
from keras.applications.efficientnet_v2 import EfficientNetV2B0
import shutil

image_size = (224, 224)
embedding_length = 2048
embedding_dim = 8
beta = 0.25
runeager = False
small_dataset = False
ds_size = 1024
batch_size = 8
test_size = batch_size
epochs = 1000
eg_learning_rate = 0.0001
d_learning_rate = 0.0001

version = 7

img_output_path = f"generated-vqgan-v{version}"
log_dir = "logs/vqgan"
def model_path(model):
  return f"models/vqgan_faces_{model}_v{version}.keras"


if os.path.exists(os.path.join(log_dir, "train")):
  shutil.rmtree(os.path.join(log_dir, "train"))

perceptual_loss_net = EfficientNetV2B0(weights='imagenet')
input_layer = perceptual_loss_net.get_layer("input_1")
perception_layer_names = [
  "stem_conv",
  "block1a_project_conv",
  "block2a_project_conv",

  "block2b_project_conv",
  "block3a_project_conv",
  "block3b_project_conv",

  "block4a_project_conv",
  "block4b_project_conv",
  "block4c_project_conv",
]

perception_layers = [perceptual_loss_net.get_layer(name) for name in perception_layer_names]
perception_layer_outputs = [layer.output for layer in perception_layers]
perception_activation = keras.models.Model(input_layer.input, perception_layer_outputs, name="perceptual_loss")
perception_activation.trainable = False
perception_activation.summary()

def perceptual_loss_fn(original, generated):
  original_activation = perception_activation(original)
  generated_activation = perception_activation(generated)
  diffs = [tf.reduce_mean((tf.nn.l2_normalize(x, axis=-1) - tf.nn.l2_normalize(x0, axis=-1)) ** 2, axis=None) for (x, x0) in zip(original_activation, generated_activation)]
  loss = sum(diffs) / len(diffs)
  return loss + tf.reduce_mean(tf.abs(original - generated), axis=None)

class VectorQuantization(layers.Layer):
  def __init__(self, embedding_length, embedding_dim, beta=0.25, **kwargs):
    super(VectorQuantization, self).__init__(**kwargs)
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
    quantized_vectors = tf.gather(self.embedding, embedding_indexes)

    embedding_loss = tf.reduce_mean((tf.stop_gradient(input) - quantized_vectors) ** 2)
    encoding_loss = tf.reduce_mean((input - tf.stop_gradient(quantized_vectors)) ** 2)
    self.add_loss(embedding_loss + self.beta * encoding_loss)

    # Straight through estimator
    quantized_vectors = input + tf.stop_gradient(quantized_vectors - input)
    return quantized_vectors

  def get_config(self):
    config = super(VectorQuantization, self).get_config()
    config.update({
      "embedding_length": self.embedding_length,
      "embedding_dim": self.embedding_dim,
      "beta": self.beta
    })
    return config

def sum_grads(*args):
  sum = None
  for grad in args:
    if sum is None:
      sum = grad
    elif grad is not None:
      sum += grad
  return sum


class VQGAN(keras.models.Model):
  def __init__(self, encoder, decoder, discriminator, discriminator_weight=1.4, **kwargs):
    super().__init__(**kwargs)
    self.discriminator = discriminator
    self.encoder = encoder
    self.decoder = decoder
    self.discriminator_weight = discriminator_weight
    self.enc_dec = keras.Model(encoder.inputs, decoder(encoder.outputs))
    self.d_loss_metric = keras.metrics.Mean(name="d_loss")
    self.eg_loss_metric = keras.metrics.Mean(name="eg_loss")
    self.perceptual_loss_metric = keras.metrics.Mean(name="perceptual_loss")
    self.decode_disc_loss_metric = keras.metrics.Mean(name="decode_disc_loss")
    self.embedding_loss_metric = keras.metrics.Mean(name="embedding_loss")
    self.adaptive_weight_metric = keras.metrics.Mean(name="adaptive_weight")

  @property
  def metrics(self):
    return [
      self.d_loss_metric, 
      self.eg_loss_metric, 
      self.embedding_loss_metric, 
      self.decode_disc_loss_metric, 
      self.perceptual_loss_metric,
      self.adaptive_weight_metric
    ]

  def compile(self, eg_optimizer, d_optimizer, loss_fn, perceptual_loss_fn, **kwargs):
    super(VQGAN, self).compile(**kwargs)
    self.eg_optimizer = eg_optimizer
    self.d_optimizer = d_optimizer
    self.loss_fn = loss_fn
    self.perceptual_loss_fn = perceptual_loss_fn
    self.encoder.compile()
    self.decoder.compile()
    self.discriminator.compile()

  def train_step(self, input):
    (eg_input, disc_input) = input[0]
    batch_size = tf.shape(eg_input)[0]
    with tf.GradientTape(persistent=True) as decoder_tape:
      decoded = self.enc_dec(eg_input)
      real_labels = tf.zeros(shape=(batch_size,))
      predictions = self.discriminator(decoded)
      perceptual_loss = self.perceptual_loss_fn(eg_input, decoded)
      gan_loss = self.loss_fn(real_labels, predictions)
      embedding_loss = sum(self.enc_dec.losses)

    gan_grads = decoder_tape.gradient(gan_loss, self.enc_dec.trainable_weights)
    perceptual_loss_grads = decoder_tape.gradient(perceptual_loss, self.enc_dec.trainable_weights)
    embedding_grads = decoder_tape.gradient(embedding_loss, self.enc_dec.trainable_weights)
    adaptive_weight = tf.norm(perceptual_loss_grads[-1]) / (tf.norm(gan_grads[-1]) + 1e-6) * self.discriminator_weight
    grads = [sum_grads(perceptual_loss_grads[i], embedding_grads[i], gan_grads[i] * adaptive_weight if gan_grads[i] is not None else None)
              for i, _ in enumerate(gan_grads)]
    self.eg_optimizer.apply_gradients(zip(grads, self.enc_dec.trainable_weights))

    decode_loss = perceptual_loss + embedding_loss + gan_loss * adaptive_weight

    self.perceptual_loss_metric.update_state(perceptual_loss)
    self.decode_disc_loss_metric.update_state(gan_loss)
    self.eg_loss_metric.update_state(decode_loss)
    self.embedding_loss_metric.update_state(embedding_loss)
    self.adaptive_weight_metric.update_state(adaptive_weight)

    with tf.GradientTape() as disc_tape:
      real_and_decoded_images = tf.concat([decoded, disc_input], axis=0)
      real_labels = tf.zeros(shape=(tf.shape(decoded)[0],))
      fake_labels = tf.ones(shape=(tf.shape(disc_input)[0],))

      labels = tf.concat([fake_labels, real_labels], axis=0)
      labels = labels + 0.05 * tf.random.uniform(shape=tf.shape(labels))
      predictions = self.discriminator(real_and_decoded_images)
      discriminator_loss = self.loss_fn(labels, predictions)

    disc_grads = disc_tape.gradient(discriminator_loss, self.discriminator.trainable_weights)
    self.d_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_weights))

    self.d_loss_metric.update_state(discriminator_loss)

    return {
      "d_loss": self.d_loss_metric.result(),
      "eg_loss": self.eg_loss_metric.result(),
      "embedding_loss": self.embedding_loss_metric.result(),
      "decode_disc_loss": self.decode_disc_loss_metric.result(),
      "perceptual_loss": self.perceptual_loss_metric.result(),
      "adaptive_weight": self.adaptive_weight_metric.result()
    }
  

class VQGanMonitor(keras.callbacks.Callback):
  def __init__(self, test_ds):
    self.test_ds = test_ds

  def on_epoch_end(self, epoch, logs=None):
    autoencoded = self.model.enc_dec(self.test_ds)
    autoencoded.numpy()
    for i in range(len(autoencoded)):
      img = keras.utils.array_to_img(autoencoded[i])
      img.save(os.path.join(img_output_path, f"autoencoded_{epoch:03d}_{i}.png"))

class VQGanCheckpoint(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    self.model.encoder.save(model_path("encoder"))
    self.model.decoder.save(model_path("decoder"))
    self.model.discriminator.save(model_path("discriminator"))

def get_discriminator():
  if os.path.exists(model_path("discriminator")):
    return keras.models.load_model(model_path("discriminator"))

  discriminator = keras.Sequential([
    keras.Input(shape=(*image_size, 3)),
    layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(2, activation="softmax")
  ], name="discriminator")
  return discriminator

def conv_block(filters, x):
  x = layers.Conv2D(filters, kernel_size=3, padding="same", strides=2)(x)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)
  x = layers.Dropout(0.2)(x)
  x = layers.Conv2D(filters, kernel_size=3, padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)
  return x

def residual_block(filters, x):
  residual = x


def positional_encoding2d():
  def inner(inputs):
    _, w, h, c = inputs.shape
    batch_size = tf.shape(inputs)[0]
    x = tf.range(start=0, limit=w, delta=1)
    x = x / w
    x = tf.expand_dims(x, axis=0)
    assert x.shape == (1, w)
    x = tf.tile(x, multiples=[h, 1])
    assert x.shape == (w, h)
    x = tf.reshape(x, shape=(w, h, 1))

    y = tf.range(start=0, limit=h, delta=1)
    y = y / h
    y = tf.expand_dims(y, axis=1)
    assert y.shape == (h, 1)
    y = tf.tile(y, [1, w])
    assert y.shape == (w, h)
    y = tf.reshape(y, shape=(w, h, 1))

    indexes = tf.concat([x, y], axis=-1)
    assert indexes.shape == (w, h, 2)

    indexes = tf.expand_dims(indexes, axis=0)
    indexes = tf.tile(indexes, [batch_size, 1, 1, 1])

    return layers.Conv2D(c, kernel_size=1, strides=1, padding="same")(indexes)
    # Todo: the sinusoidal way from All you need is attention
  return inner

def self_attention(num_heads=1, key_dim=64):
  def inner(x):
    pos = positional_encoding2d()(x)
    x = layers.add([x, pos])
    x = layers.MultiHeadAttention(num_heads, key_dim, attention_axes=(2, 3))(x, x, x)
    return x
  return inner

def get_encoder():
  if os.path.exists(model_path("encoder")):
    return keras.models.load_model(model_path("encoder"), custom_objects={
      "VectorQuantization": VectorQuantization
    })
  encoder_inputs = keras.Input(shape=(*image_size, 3))
  x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(encoder_inputs)
  x = conv_block(128, x)
  x = conv_block(256, x)
  x = self_attention(num_heads=3, key_dim=256)(x)
  x = layers.Conv2D(embedding_dim, kernel_size=3, padding="same")(x)
  x = VectorQuantization(embedding_length, embedding_dim)(x)
  encoder = keras.models.Model(encoder_inputs, x, name="encoder")
  return encoder

def get_decoder():
  if os.path.exists(model_path("decoder")):
    return keras.models.load_model(model_path("decoder"))
  decoder_input = keras.Input(shape=(image_size[0]//4, image_size[1]//4, embedding_dim))
  x = layers.Conv2D(256, 3, activation="relu", padding="same")(decoder_input)
  x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
  # x = layers.Dropout(0.2)(x)
  x = layers.Conv2DTranspose(128, 4, strides=2, padding="same")(x)
  x = layers.Conv2DTranspose(64, 4, strides=2, padding="same")(x)
  decoder_output = layers.Conv2D(3, 3, padding="same")(x)

  decoder = keras.models.Model(decoder_input, decoder_output, name="decoder")
  return decoder

encoder = get_encoder()
decoder = get_decoder()
discriminator = get_discriminator()

model = VQGAN(encoder, decoder, discriminator)

model.compile(
  eg_optimizer=keras.optimizers.Adam(learning_rate=eg_learning_rate),
  d_optimizer=keras.optimizers.Adam(learning_rate=d_learning_rate),
  loss_fn=keras.losses.SparseCategoricalCrossentropy(),
  perceptual_loss_fn=perceptual_loss_fn,
  run_eagerly=runeager)

dataset = keras.utils.image_dataset_from_directory(
  f"img_align_celeba{'_small' if small_dataset else ''}",
  label_mode=None,
  image_size=image_size,
  batch_size=batch_size,
  smart_resize=True,
  shuffle=False
)
dataset = dataset.map(lambda x: x / (255. / 2) - 1.)
test_ds = dataset.take(1).as_numpy_iterator().next()[:test_size]
dataset = dataset.skip(1)

ds_total_size = dataset.__len__()
split_size = ds_total_size // 2

eg_data = dataset.take(min(ds_size, split_size))
dataset = dataset.skip(min(ds_size, split_size))
disc_data = dataset.take(min(ds_size, split_size))
dataset = tf.data.Dataset.zip(((eg_data, disc_data),))

os.makedirs(img_output_path, exist_ok=True)

for i in range(len(test_ds)):
  img = keras.utils.array_to_img(test_ds[i])
  img.save(os.path.join(img_output_path,  f"aaref_{i}.png"))

# log_dir = "logs/dbg"
# tf.debugging.experimental.enable_dump_debug_info(
#     log_dir,
#     tensor_debug_mode="NO_TENSOR",
#     circular_buffer_size=-1)

callbacks = [
  VQGanMonitor(test_ds),
  #callbacks.ModelCheckpoint(model_path),
  VQGanCheckpoint(),
  callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]
model.fit(dataset, epochs=epochs, callbacks=callbacks)