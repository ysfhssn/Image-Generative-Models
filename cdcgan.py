import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, optimizers
import os
from tqdm import tqdm
import time
from utils import generate_and_save_images


# HYPERPARAMETERS
BATCH_SIZE = 256
EPOCHS = 200
bce = losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = optimizers.Adam(1E-4)
discriminator_optimizer = optimizers.Adam(1E-4)
NOISE_SHAPE = (100,)

# PATHS
GEN_DIR = 'cdcgan_generated'
CHECKPOINT_DIR ='cdcgan_training_checkpoints'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, 'ckpt')
CHECKPOINT_SAVE_FREQUENCY = EPOCHS // 10

# DATASET
INPUT_SHAPE = (32, 32, 3)
NUMBER_LABELS = 10
(train_images, train_labels), (_, _) = datasets.cifar10.load_data()
train_images = train_images.reshape((-1, *INPUT_SHAPE)).astype('float32')
train_images = train_images / 255
gen_output_activation = 'sigmoid'
train_labels = tf.one_hot(train_labels, NUMBER_LABELS)
train_labels = tf.reshape(train_labels, (-1, NUMBER_LABELS))
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(len(train_images)).batch(BATCH_SIZE)


def generator_model():
    x1 = layers.Input(shape=NOISE_SHAPE, dtype='float32')
    x2 = layers.Input(shape=(NUMBER_LABELS,), dtype='float32')
    x = layers.concatenate([x1, x2])

    # (4, 4, 256)
    x = layers.Dense(4*4*256, use_bias=False)(x)
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
    y = layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same', use_bias=False, activation=gen_output_activation)(x)
    assert y.shape == (None, *INPUT_SHAPE)

    return models.Model(inputs=[x1, x2], outputs=y)

def generator_loss(fake_output):
    return bce(tf.ones_like(fake_output), fake_output)

def discriminator_model():
    x1 = layers.Input(shape=INPUT_SHAPE, dtype='float32')
    x2 = layers.Input(shape=(NUMBER_LABELS,), dtype='float32')

    x2_ = layers.Dense(INPUT_SHAPE[0]*INPUT_SHAPE[1]*INPUT_SHAPE[2], use_bias=False)(x2)
    x2_ = layers.Reshape(INPUT_SHAPE)(x2_)
    x = layers.concatenate([x1, x2_])

    # (32, 32, 3)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.LeakyReLU()(x)

    # (16, 16, 3)
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    # (8, 8, 3)
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    # (4, 4, 3)
    x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)

    y = layers.Dense(1)(x)

    return models.Model(inputs=[x1, x2], outputs=y)

def discriminator_loss(real_output, fake_output):
    real_loss = bce(tf.ones_like(real_output), real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def train(generator, discriminator, dataset, epochs, seed=None, checkpoint=None):
    for epoch in tqdm(range(epochs)):
        start = time.time()

        for step, (image_batch, label_batch) in enumerate(dataset):
            gen_loss, disc_loss = train_step(generator, discriminator, image_batch, label_batch)
            print(f'  Step: {step} - Disc loss: {disc_loss} - Gen loss: {gen_loss}')

        if seed is not None:
            generate_and_save_images(generator, seed, f'{GEN_DIR}/{epoch+1:03d}.png')

        if checkpoint is not None and (epoch+1) % CHECKPOINT_SAVE_FREQUENCY == 0:
            checkpoint.save(file_prefix=CHECKPOINT_PREFIX)

        print(f'Epoch {epoch+1:03d}: {time.time()-start:.0f} seconds')

    checkpoint.save(file_prefix=CHECKPOINT_PREFIX)

@tf.function
def train_step(generator, discriminator, images, labels):
    noise = tf.random.normal((len(labels), *NOISE_SHAPE))
    gen_labels = tf.random.uniform(shape=(len(labels),), minval=0, maxval=NUMBER_LABELS, dtype=tf.int32)
    gen_labels = tf.one_hot(gen_labels, NUMBER_LABELS)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_images = generator([noise, gen_labels], training=True)

        real_output = discriminator([images, labels], training=True)
        fake_output = discriminator([gen_images, gen_labels], training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss


if __name__ == '__main__':
    seed = [tf.random.normal((16, *NOISE_SHAPE)), tf.one_hot(tf.range(0, 16)%NUMBER_LABELS, NUMBER_LABELS)]
    generator = generator_model()
    discriminator = discriminator_model()

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    # checkpoint.restore(f'./{CHECKPOINT_DIR}/ckpt-10')

    train(generator, discriminator, train_dataset, EPOCHS, seed, checkpoint)
    # tf.keras.utils.save_img('test.png', generator(seed)[0])