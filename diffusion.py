from turtle import position
import keras
import keras.utils
import keras.layers as layers
import keras.losses
import keras.metrics
import keras.callbacks
import keras.optimizers
import keras.losses
import tensorflow as tf
from os.path import exists
from os import mkdir

image_size = (64, 64)

steps = 500
variance_schedule_start = 0.0001
variance_schedule_end = 0.02
variance_schedule = [i * (variance_schedule_end - variance_schedule_start) / steps + variance_schedule_start for i in range(steps)]
num_layers = 2
version = 5

model_path = f"diffusion_faces_{steps}_v{version}.keras"

class Diffusion(keras.models.Model):
  def __init__(self, model,  num_steps, variance_schedule, **kwargs):
    super().__init__(kwargs)
    self.num_steps = num_steps
    self.loss_tracker = keras.metrics.Mean("loss")
    self.model = model
    self.loss_fn = keras.losses.MSE
    self.variance_schedule = variance_schedule
    self.alpha = [1 - b for b in variance_schedule]
    self.alpha_accumulated = []
    total = 1.
    for a in self.alpha:
      total *= a
      self.alpha_accumulated.append(total)

  @property
  def metrics(self):
    return [self.loss_tracker]

  def train_step(self, real_images):
    _, width, height, channels = real_images.shape
    batch_size = tf.shape(real_images)[0]
    t = tf.random.uniform(shape=(batch_size,), minval=0, maxval=self.num_steps, dtype=tf.int32)
    t_input = t / self.num_steps
    input_shape = (batch_size, width, height, channels)
    noise = tf.random.normal(shape=input_shape)
    alpha_t = tf.gather(self.alpha_accumulated, t)
    alpha_t = tf.reshape(alpha_t, shape=(batch_size, 1, 1, 1))
    noise_variance = tf.sqrt(1 - alpha_t) * noise
    img_median = tf.sqrt(alpha_t) * real_images
    noisy_input = img_median + noise_variance

    with tf.GradientTape() as tape:
      predicted_noise = self.model([noisy_input, t_input])
      noise_loss = self.loss_fn(noise, predicted_noise)
    grads = tape.gradient(noise_loss, self.model.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
    self.loss_tracker.update_state(noise_loss)

    return {
      "loss": self.loss_tracker.result()
    }

  def sample(self, num_images):
    result_shape = (num_images, image_size[0], image_size[1], 3)
    result = tf.random.normal(shape=result_shape)
    for t in reversed(range(1, self.num_steps)):
      if t > 1:
        z = tf.random.normal(shape=result_shape)
      else:
        z = tf.zeros(shape=result_shape)
      
      alpha = self.alpha[t]
      alpha_t = self.alpha_accumulated[t]
      t_input = tf.constant(t / self.num_steps, dtype=tf.float32)
      t_input = tf.broadcast_to(t_input, shape=(num_images,))
      predicted_noise = self.model([result, t_input])
      noise_factor = (1 - alpha) / tf.sqrt(1 - alpha_t)
      sigma = tf.sqrt(self.variance_schedule[t])
      result = (1 / tf.sqrt(alpha)) * (result - noise_factor * predicted_noise) + sigma * z

    return result

  
def conv_block(filters):
  def inner(x):
    x = layers.Conv2D(filters=filters, kernel_size=3, padding="same", activation="relu")(x)
    return x
  return inner

def time_embedding_proj(shape, dense_dim=8):
  def inner(x):
    b, w, h, c = shape
    x = layers.Dense(units=dense_dim, activation="relu")(x)
    x = layers.Dense(units=w * h * c, activation="relu")(x)
    x = layers.Reshape(target_shape=(w, h, c))(x)
    return x
  return inner

def downsample():
  def inner(x):
    x = layers.MaxPooling2D(pool_size=2)(x)
    return x
  return inner

def upsample(filters):
  def inner(x):
    x = layers.Conv2DTranspose(kernel_size=4, strides=2, filters=filters, padding="same")(x)
    return x
  return inner

def dropout(rate):
  def inner(x):
    x = layers.SpatialDropout2D(rate)(x)
    return x
  return inner

def unet_layer(filters, next_layer):
  def inner(x, time_embedding):
    x = conv_block(filters)(x)
    # In bottom layer, do self attention
    if next_layer is None:
      x = self_attention(3)(x)
    x = dropout(0.2)(x)
    x = conv_block(filters)(x)
    residual = x
    if next_layer is not None:
      x = downsample()(x)
      x = next_layer(x, filters * 2, time_embedding)
      x = upsample(filters)(x)
      time = time_embedding_proj(x.shape)(time_embedding)
      x = layers.add([residual, x, time])
      x = conv_block(filters)(x)
      x = conv_block(filters)(x)
    return x
  return inner

def sublayer(next_layer):
  def inner(x, filters, time_embedding):
    return unet_layer(filters, next_layer)(x, time_embedding)
  return inner

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
    # add positional encoding?
    pos = positional_encoding2d()(x)
    x = layers.add([x, pos])
    x = layers.MultiHeadAttention(num_heads, key_dim, attention_axes=(2, 3))(x, x, x)
    return x
  return inner

def cross_attention(num_heads=1, key_dim=64):
  def inner(qk, v):
    pos_qk = positional_encoding2d()(qk)
    pos_v = positional_encoding2d()(v)
    qk = layers.add([qk, pos_qk])
    v = layers.add([v, pos_v])
    return layers.MultiHeadAttention(num_heads, key_dim, attention_axes=(2, 3))(qk, v, qk)
  return inner


def get_model():
  if exists(model_path):
    return keras.models.load_model(model_path)

  w, h = image_size
  c = 3
  image_input = keras.Input(shape=(w, h, c))
  t_input = keras.Input(shape=(1,))
  sublayers = sublayer(None)
  for i in range(num_layers):
    sublayers = sublayer(sublayers)

  x = unet_layer(64, sublayers)(image_input, t_input)
  output = layers.Conv2D(filters=3, kernel_size=3, padding="same")(x)
  model = keras.Model([image_input, t_input], output)
  return model

model = get_model()
model.summary()

diffusion = Diffusion(model, steps, variance_schedule)
diffusion.compile(optimizer="rmsprop")

class DiffusionMonitor(keras.callbacks.Callback):
  def __init__(self, num_img=3):
    self.num_img = num_img

  def on_epoch_end(self, epoch, logs=None):
    generated_images = self.model.sample(self.num_img)
    img_path = f"generated-diffusion-v{version}"
    if not exists(img_path):
      mkdir(img_path)

    for i in range(self.num_img):
      img = keras.utils.array_to_img(generated_images[i])
      img.save(f"{img_path}/generated_img_{epoch:03d}_{i}.png")

class Save(keras.callbacks.Callback):
  def __init__(self, model_path):
    self.model_path = model_path

  def on_epoch_end(self, epoch, logs=None):
    self.model.model.save(self.model_path)

callbacks = [
  DiffusionMonitor(),
  Save(model_path)
]

dataset = keras.utils.image_dataset_from_directory(
  "img_align_celeba",
  label_mode=None,
  image_size=image_size,
  batch_size=16,
  smart_resize=True,
  seed=1,
  subset="training",
  validation_split=0.95)

dataset = dataset.map(lambda x: x / (255. / 2) - 1.)

diffusion.fit(dataset, callbacks=callbacks, epochs=100)

