import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, optimizers
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from utils import generate_and_save_images


# HYPERPARAMETERS
BATCH_SIZE = 256
EPOCHS = 200
mse = losses.MeanSquaredError()
model_optimizer = optimizers.legacy.Adam(1E-4)
TIMESTEPS = 100

linear_beta_schedule = tf.linspace(0.0001, 0.005, TIMESTEPS)

betas = linear_beta_schedule
alphas = 1. - betas
alphas_cumprod = tf.math.cumprod(alphas, axis=0)

# PATHS
GEN_DIR = 'diffusion_generated'
CHECKPOINT_DIR ='diffusion_training_checkpoints'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")
CHECKPOINT_SAVE_FREQUENCY = EPOCHS // 10

# DATASET
INPUT_SHAPE = (32, 32, 3)
(train_images, _), (test_images, _) = datasets.cifar10.load_data()
train_images = train_images.reshape((-1, *INPUT_SHAPE)).astype('float32')
test_images = test_images.reshape((-1, *INPUT_SHAPE)).astype('float32')
train_images = train_images / 255
test_images = test_images / 255
train_dataset = tf.data.Dataset.from_tensor_slices((train_images)).shuffle(len(train_images)).batch(256)


def forward_diffusion(x0, t):
    """
        q(xt|xt-1) ~ N(√a*xt-1; (1-a)*I)
        q(xt|x0) ~ N(√ā*x0; (1-ā)*I)
        q(xt|x0) = √ā*x0 + √(1-ā)*ε
    """
    x0_shape = tf.shape(x0)
    noise = tf.random.normal(shape=x0_shape)
    alphas_cumprod_t = tf.reshape(tf.gather(alphas_cumprod, t), (-1, *((1,)*(len(x0_shape)-1))))
    return tf.sqrt(alphas_cumprod_t) * x0 + tf.sqrt(1 - alphas_cumprod_t) * noise, noise

def backward_model():
    """
        pθ(xt-1|xt) ~ N(µθ(xt,t); σθ²(xt,t))
    """
    x1 = layers.Input(shape=INPUT_SHAPE, dtype='float32')
    x2 = layers.Input(shape=(TIMESTEPS,), dtype='float32')

    x2_ = layers.Dense(INPUT_SHAPE[0]*INPUT_SHAPE[1]*INPUT_SHAPE[2], use_bias=False)(x2)
    x2_ = layers.Reshape(INPUT_SHAPE)(x2_)
    x = layers.concatenate([x1, x2_])

    # (32, 32, 64)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.LeakyReLU()(x)

    # (16, 16, 128)
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    # (8, 8, 128)
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    # (4, 4, 256)
    x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Dropout(0.3)(x)

    # (8, 8, 128)
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # (16, 16, 128)
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # (32, 32, 3)
    y = layers.Conv2DTranspose(3, (3, 3), padding='same', use_bias=False, activation='sigmoid')(x)

    return models.Model(inputs=[x1, x2], outputs=[y])

def train(diffusion_model, dataset, epochs, seed=None, checkpoint=None):
    for epoch in tqdm(range(epochs)):
        start = time.time()

        for step, image_batch in enumerate(dataset):
            mse_loss = train_step(diffusion_model, image_batch)
            print(f'  Step: {step} - Train loss: {mse_loss}')

        if seed is not None:
            generate_and_save_images(diffusion_model, seed, f'{GEN_DIR}/{epoch+1:03d}.png')

        if checkpoint is not None and (epoch+1) % CHECKPOINT_SAVE_FREQUENCY == 0:
            checkpoint.save(file_prefix=CHECKPOINT_PREFIX)

        print(f'Epoch {epoch+1:03d}: {time.time()-start:.0f} seconds')

    checkpoint.save(file_prefix=CHECKPOINT_PREFIX)

@tf.function
def train_step(diffusion_model, images):
    random_ts = tf.random.uniform((len(images),), 0, TIMESTEPS, dtype=tf.int32)
    noisy_images, noises = forward_diffusion(images, random_ts)

    with tf.GradientTape() as tape:
        noises_pred = diffusion_model(noisy_images, tf.one_hot(random_ts, TIMESTEPS))
        train_loss = mse(noises, noises_pred)

    grads = tape.gradient(train_loss, diffusion_model.trainable_variables)
    model_optimizer.apply_gradients(zip(grads, diffusion_model.trainable_variables))

    return train_loss

def plot_img_forward_diffusion(x0, num_images=10, path=None):
    fig, axes = plt.subplots(1, num_images, figsize=(9, 2))

    for i, t in enumerate(range(0, TIMESTEPS, TIMESTEPS // num_images)):
        noisy_image, _ = forward_diffusion(x0, t)
        axes[i].set_title(f't = {t}')
        axes[i].imshow(noisy_image)

    if path is not None:
        plt.savefig(path)

    plt.close(fig)

def plot_img_backward_diffusion(diffusion_model, seed=None, num_images=10, path=None):
    fig, axes = plt.subplots(1, num_images, figsize=(9, 2))

    xT = seed if seed is not None else tf.random.normal(shape=(1, *INPUT_SHAPE))
    for i, t in enumerate(reversed(range(0, TIMESTEPS, TIMESTEPS // num_images))):
        axes[i].set_title(f't = {t}')
        axes[i].imshow(xT[0])
        noise_pred = diffusion_model(xT, tf.one_hot(t, TIMESTEPS))
        xT = 1/tf.sqrt(alphas[t]) * (xT - betas[t]/tf.sqrt(1-alphas_cumprod[t])*noise_pred)
        if t > 0:
            xT += tf.sqrt(betas[t]) * tf.random.normal(shape=tf.shape(xT))

    if path is not None:
        plt.savefig(path)

    plt.close(fig)


if __name__ == '__main__':
    diffusion_model = backward_model()
    seed = tf.random.normal(shape=(1, *INPUT_SHAPE))

    plot_img_forward_diffusion(train_images[0])