# Image generative models

<p align="center"><img src="https://img-blog.csdnimg.cn/360bca80d4d146f48d1bd40684e0b998.png"></p>

## cDCGAN
<img src="https://miro.medium.com/max/1400/1*GAEHmW30RXtcf8MineB49w.png" width="500">

* [GAN — Why is it hard to train GANs ?](https://jonathan-hui.medium.com/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b)
* [cGAN — What's a cGAN ?](https://datascientest.com/conditional-generative-adversarial-network-cgan)

### Results (200 epochs)
<img src="./docs/cdcgan.gif" width="300">

## Diffusion Model
<img src="https://theaisummer.com/static/ecb7a31540b18a8cbd18eedb446b468e/ee604/diffusion-models.png" width="600">

* Forward Process
<p align="center"><img src="./docs/forward.png"></p>

* [Backward Process Mathematics](https://youtu.be/HoKDTa5jHvg?t=819)
* [Diffusion model from sratch (Pytorch)](https://youtu.be/a4Yfz2FxXiY)
* [How DALL-E 2 works ?](https://www.assemblyai.com/blog/how-dall-e-2-actually-works/)

## Variational Autoencoder
<img src="https://miro.medium.com/max/1400/1*r1R0cxCnErWgE0P4Q-hI0Q.jpeg" width="600">

* [The Reparameterization Trick](https://www.baeldung.com/cs/vae-reparameterization)
* [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
    * [DKL(P || Q) ≥ 0](https://stats.stackexchange.com/questions/335197/why-kl-divergence-is-non-negative)

### Results (100 epochs then no improvements)
<img src="./docs/vae.gif" width="300">