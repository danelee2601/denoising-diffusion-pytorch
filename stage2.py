import os
from argparse import ArgumentParser

import wandb
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from encoder_decoders.vq_vae_encdec import VQVAEEncoder, VQVAEDecoder
from vector_quantization import VectorQuantize
from utils import load_yaml_param_settings, get_root_dir, freeze


def load_pretrained_encoder_decoder_vq(config: dict, dirname, freeze_models: bool = True, load_cond_models=False):
    dim = config['encoder']['dim']
    bottleneck_dim = config['encoder']['bottleneck_dim']
    in_channels = config['dataset']['in_channels']
    downsampling_rate = config['encoder']['downsampling_rate']
    img_size = config['dataset']['img_size']

    encoder = VQVAEEncoder(dim, bottleneck_dim, in_channels, downsampling_rate, config['encoder']['n_resnet_blocks'], config['encoder']['output_norm'])
    decoder = VQVAEDecoder(dim, bottleneck_dim, in_channels, downsampling_rate, config['decoder']['n_resnet_blocks'], img_size)
    vq_model = VectorQuantize(bottleneck_dim, **config['VQ-VAE'])

    if not load_cond_models:
        encoder_fname = 'encoder.ckpt'
        decoder_fname = 'decoder.ckpt'
        vq_model_fname = 'vq_model.ckpt'
    else:
        encoder_fname = 'encoder_cond.ckpt'
        decoder_fname = 'decoder_cond.ckpt'
        vq_model_fname = 'vq_model_cond.ckpt'

    encoder.load_state_dict(torch.load(os.path.join(dirname, encoder_fname)))
    decoder.load_state_dict(torch.load(os.path.join(dirname, decoder_fname)))
    vq_model.load_state_dict(torch.load(os.path.join(dirname, vq_model_fname)))

    encoder.eval()
    decoder.eval()
    vq_model.eval()

    if freeze_models:
        freeze(encoder)
        freeze(decoder)
        freeze(vq_model)

    return encoder, decoder, vq_model


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    return parser.parse_args()


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    # load the pretrained encoder, decoder, and vq
    encoder, decoder, vq_model = load_pretrained_encoder_decoder_vq(config, 'saved_models', freeze_models=True)
    encoder_cond, decoder_cond, vq_model_cond = load_pretrained_encoder_decoder_vq(config, 'saved_models', freeze_models=True, load_cond_models=True)
    encoder, decoder, vq_model = encoder.cuda(), decoder.cuda(), vq_model.cuda()
    encoder_cond, decoder_cond, vq_model_cond = encoder_cond.cuda(), decoder_cond.cuda(), vq_model_cond.cuda()

    # model
    model = Unet(
        in_channels=config['encoder']['bottleneck_dim'],
        dim=64,
        dim_mults=(1, 2, 4, 8),
        self_condition=config['diffusion']['unet']['self_condition'],
        z_size=encoder.H_prime[0].item(),  # width or height of z
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        in_size=encoder.H_prime[0].item(),  # width or height of z
        timesteps=1000,  # number of steps
        sampling_timesteps=1000,
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type='l1',  # L1 or L2
        auto_normalize=False,
    ).cuda()

    # train
    wandb.init(project='GeoDiffusion-stage2',config=config)
    trainer = Trainer(
        diffusion,
        config,
        encoder,
        decoder,
        vq_model,
        encoder_cond,
        vq_model_cond,
        train_batch_size=config['dataset']['batch_sizes']['stage2'],
        train_lr=8e-5,
        train_num_steps=700000,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        fp16=False,
        save_and_sample_every=1000, #1000,
        num_samples=9,
        augment_horizontal_flip=False
    )

    trainer.train()
