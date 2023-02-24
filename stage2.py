from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


if __name__ == '__main__':
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l1'            # L1 or L2
    ).cuda()

    trainer = Trainer(
        diffusion,
        'dataset/img/facies',
        train_batch_size = 16,
        train_lr = 8e-5,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp=False,                        # turn on mixed precision
        fp16=True,
        save_and_sample_every=1000,
        num_samples=9,
        augment_horizontal_flip=False
    )

    trainer.train()
