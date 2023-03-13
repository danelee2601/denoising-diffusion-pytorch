import torch
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import wandb
from einops import rearrange

from encoder_decoders.vq_vae_encdec import VQVAEEncoder, VQVAEDecoder
from vector_quantization import VectorQuantize
from experiments.exp_base import ExpBase, detach_the_unnecessary
from utils import freeze, timefreq_to_time, time_to_timefreq, zero_pad_low_freq, zero_pad_high_freq, quantize


class ExpVQVAE(ExpBase):
    def __init__(self,
                 img_size: int,
                 config: dict,
                 n_train_samples: int):
        """
        :param config: configs/config.yaml
        :param n_train_samples: number of training samples
        """
        super().__init__()
        self.config = config
        self.T_max = config['trainer_params']['max_epochs']['stage1'] * (np.ceil(n_train_samples / config['dataset']['batch_sizes']['stage1']) + 1)

        # self.n_fft = config['VQ-VAE']['n_fft']
        dim = config['encoder']['dim']
        in_channels = config['dataset']['in_channels']
        # downsampled_width = config['encoder']['downsampled_width']
        # downsample_rate = compute_downsample_rate(input_length, downsampled_width)
        downsampling_rate = config['encoder']['downsampling_rate']

        # encoder
        self.encoder = VQVAEEncoder(dim, in_channels, downsampling_rate, config['encoder']['n_resnet_blocks'])
        self.decoder = VQVAEDecoder(dim, in_channels, downsampling_rate, config['decoder']['n_resnet_blocks'], img_size)
        self.vq_model = VectorQuantize(dim, **config['VQ-VAE'])

        self.encoder_cond = VQVAEEncoder(dim, in_channels, downsampling_rate, config['encoder']['n_resnet_blocks'])
        self.decoder_cond = VQVAEDecoder(dim, in_channels, downsampling_rate, config['decoder']['n_resnet_blocks'], img_size)
        self.vq_model_cond = VectorQuantize(dim, **config['VQ-VAE'])

    def forward(self, x, kind):
        assert kind in ['x', 'x_cond']

        if kind == 'x':
            encoder = self.encoder
            decoder = self.decoder
            vq_model = self.vq_model
        elif kind == 'x_cond':
            encoder = self.encoder_cond
            decoder = self.decoder_cond
            vq_model = self.vq_model_cond
        else:
            raise ValueError

        # forward
        z = encoder(x)  # (b d h' w')
        z_q, indices, vq_loss, perplexity = quantize(z, vq_model)  # z_q: (b d h' w')
        vq_loss = vq_loss['loss']
        xhat = decoder(z_q)  # (b c h w)

        y_true = x.argmax(dim=1)  # (b h w)
        y_true = y_true.flatten()  # (b*h*w)
        y_pred = rearrange(xhat, 'b c h w -> (b h w) c')
        categorical_recons_loss = torch.nn.functional.cross_entropy(y_pred, y_true)

        # plot `x` and `xhat`
        r = np.random.rand()
        if self.training and r <= 0.05:
            x = x.cpu()
            xhat = xhat.detach().cpu()
            b = np.random.randint(0, x.shape[0])

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            plt.suptitle(f'ep_{self.current_epoch}')
            axes[0].imshow(x[b].argmax(dim=0))
            axes[0].invert_yaxis()
            axes[0].set_xticks([])
            axes[0].set_yticks([])

            axes[1].imshow(xhat[b].argmax(dim=0))
            axes[1].invert_yaxis()
            axes[1].set_xticks([])
            axes[1].set_yticks([])

            plt.tight_layout()
            if kind == 'x':
                wandb.log({"x vs xhat (training)": wandb.Image(plt)})
            elif kind == 'x_cond':
                wandb.log({"x_cond vs xhat_cond (training)": wandb.Image(plt)})
            plt.close()

        # # plot histogram of z
        # r = np.random.rand()
        # if self.training and r <= 0.05:
        #     z_q = z_q.detach().cpu().flatten().numpy()
        #
        #     fig, ax = plt.subplots(1, 1, figsize=(5, 2))
        #     ax.hist(z_q, bins='auto')
        #     plt.tight_layout()
        #     if kind == 'x':
        #         wandb.log({"hist(z_q)": wandb.Image(plt)})
        #     elif kind == 'x_cond':
        #         wandb.log({"hist(z_q_cond)": wandb.Image(plt)})
        #     plt.close()

        return categorical_recons_loss, vq_loss, perplexity

    def training_step(self, batch, batch_idx):
        x, x_cond = batch
        categorical_recons_loss, vq_loss, perplexity = self.forward(x, kind='x')
        categorical_recons_loss_cond, vq_loss_cond, perplexity_cond = self.forward(x_cond, kind='x_cond')
        loss = categorical_recons_loss + vq_loss
        loss += categorical_recons_loss_cond + vq_loss_cond

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {'loss': loss,
                     'categorical_recons_loss': categorical_recons_loss,
                     'vq_loss': vq_loss,
                     'perplexity': perplexity,
                     'categorical_recons_loss_cond': categorical_recons_loss_cond,
                     'vq_loss_cond': vq_loss_cond,
                     'perplexity_cond': perplexity_cond,
                     }

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def validation_step(self, batch, batch_idx):
        x, x_cond = batch
        categorical_recons_loss, vq_loss, perplexity = self.forward(x, kind='x')
        categorical_recons_loss_cond, vq_loss_cond, perplexity_cond = self.forward(x_cond, kind='x_cond')
        loss = categorical_recons_loss + vq_loss
        loss += categorical_recons_loss_cond + vq_loss_cond

        # log
        loss_hist = {'loss': loss,
                     'categorical_recons_loss': categorical_recons_loss,
                     'vq_loss': vq_loss,
                     'perplexity': perplexity,
                     'categorical_recons_loss_cond': categorical_recons_loss_cond,
                     'vq_loss_cond': vq_loss_cond,
                     'perplexity_cond': perplexity_cond,
                     }

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), weight_decay=self.config['exp_params']['weight_decay'])
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, self.T_max)}

    def test_step(self, batch, batch_idx):
        x, x_cond = batch
        categorical_recons_loss, vq_loss, perplexity = self.forward(x, kind='x')
        categorical_recons_loss_cond, vq_loss_cond, perplexity_cond = self.forward(x_cond, kind='x_cond')
        loss = categorical_recons_loss + vq_loss
        loss += categorical_recons_loss_cond + vq_loss_cond

        # log
        loss_hist = {'loss': loss,
                     'categorical_recons_loss': categorical_recons_loss,
                     'vq_loss': vq_loss,
                     'perplexity': perplexity,
                     'categorical_recons_loss_cond': categorical_recons_loss_cond,
                     'vq_loss_cond': vq_loss_cond,
                     'perplexity_cond': perplexity_cond,
                     }

        detach_the_unnecessary(loss_hist)
        return loss_hist