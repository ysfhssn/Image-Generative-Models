import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, optimizers
import os
from tqdm import tqdm
import time
from utils import generate_and_save_images


# HYPERPARAMETERS
BATCH_SIZE = 256
EPOCHS = 200
mse = losses.MeanSquaredError()
vae_optimizer = optimizers.legacy.Adam(1E-4)
LATENT_SPACE_SHAPE = (100,)
KL_WEIGHT = 0.01

# PATHS
GEN_DIR = 'vae_generated'
CHECKPOINT_DIR ='vae_training_checkpoints'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")
CHECKPOINT_SAVE_FREQUENCY = EPOCHS // 10

# DATASET
INPUT_SHAPE = (32, 32, 3)
(train_images, _), (test_images, _) = datasets.cifar10.load_data()
train_images = train_images.reshape((-1, *INPUT_SHAPE)).astype('float32')
test_images = test_images.reshape((-1, *INPUT_SHAPE)).astype('float32')
train_images = train_images / 255
test_images = test_images / 255
train_dataset = tf.data.Dataset.from_tensor_slices((train_images)).shuffle(len(train_images)).batch(BATCH_SIZE)


def encoder_model():
    x1 = layers.Input(shape=INPUT_SHAPE)

    # (32, 32, 64)
    x = layers.Conv2D(64, (3, 3), padding='same')(x1)
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
    x = layers.Flatten()(x)

    mu = layers.Dense(*LATENT_SPACE_SHAPE)(x)
    log_var = layers.Dense(*LATENT_SPACE_SHAPE)(x)
    z = layers.Lambda(compute_latent_space, output_shape=LATENT_SPACE_SHAPE)([mu, log_var]) # Reparametrization trick

    return models.Model(inputs=[x1], outputs=[mu, log_var, z])

def compute_latent_space(mu_log_var):
    mu, log_var = mu_log_var
    eps = tf.random.normal(shape=tf.shape(log_var))
    return mu + tf.exp(0.5*log_var) * eps

def kl_loss(mu, log_var):
    """
        KL divergence loss: https://stats.stackexchange.com/questions/318184/kl-loss-with-a-unit-gaussian
        kl_loss = -0.5 * (ln(σ²) - σ² - μ² + 1)
    """
    return tf.reduce_mean(-0.5 * (log_var - tf.exp(log_var) - tf.square(mu) + 1))

def decoder_model():
    x1 = layers.Input(shape=LATENT_SPACE_SHAPE)

    # (4, 4, 256)
    x = layers.Dense(4*4*256, use_bias=False)(x1)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((4, 4, 256))(x)

    # (8, 8, 128)
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # (16, 16, 128)
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # (32, 32, 3)
    y = layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid')(x)
    assert y.shape == (None, *INPUT_SHAPE)

    return models.Model(inputs=[x1], outputs=[y])

def reconstruction_loss(real_images, fake_images):
    return mse(real_images, fake_images)

def train(vae, dataset, epochs, seed=None, checkpoint=None):
    for epoch in tqdm(range(epochs)):
        start = time.time()

        for step, image_batch in enumerate(dataset):
            mse_loss, kl, train_loss = train_step(vae, image_batch)
            print(f'  Step: {step} - MSE loss: {mse_loss} - KL loss: {kl} - Train loss: {train_loss}')

        if seed is not None:
            generate_and_save_images(decoder, seed, f'{GEN_DIR}/{epoch+1:03d}.png')

        if checkpoint is not None and (epoch+1) % CHECKPOINT_SAVE_FREQUENCY == 0:
            checkpoint.save(file_prefix=CHECKPOINT_PREFIX)

        print(f'Epoch {epoch+1:03d}: {time.time()-start:.0f} seconds')

    checkpoint.save(file_prefix=CHECKPOINT_PREFIX)

@tf.function
def train_step(vae, images):
    with tf.GradientTape() as tape:
        reconstructed = vae(images)

        mse_loss = reconstruction_loss(images, reconstructed)
        kl = sum(vae.losses)
        train_loss = KL_WEIGHT*kl + mse_loss

    grads = tape.gradient(train_loss, vae.trainable_variables)
    vae_optimizer.apply_gradients(zip(grads, vae.trainable_variables))

    return mse_loss, kl, train_loss


if __name__ == '__main__':
    seed = tf.random.normal((16, *LATENT_SPACE_SHAPE))
    encoder = encoder_model()
    decoder = decoder_model()
    mu, log_var, z = encoder(encoder.input)
    vae = models.Model(inputs=[encoder.input], outputs=[decoder(z)])
    vae.add_loss(kl_loss(mu, log_var))

    checkpoint = tf.train.Checkpoint(vae_optimizer=vae_optimizer, vae=vae)
    # checkpoint.restore(f"./{CHECKPOINT_DIR}/ckpt-10")

    # train(vae, train_dataset, EPOCHS, seed, checkpoint)
    # tf.keras.utils.save_img('test.png', decoder(seed)[0])