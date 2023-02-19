import matplotlib.pyplot as plt
import os


def generate_and_save_images(generator, gen_input, path, show=False):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    predictions = generator(gen_input, training=False)

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i])
        plt.axis('off')

    plt.savefig(path)
    plt.close(fig)

    if show:
        plt.show()