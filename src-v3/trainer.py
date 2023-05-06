import os
import math
from decimal import Decimal
from tensorboardX import SummaryWriter
import utility
import numpy as np
import torch
import torch.nn.utils as utils
from tqdm import tqdm
import wandb

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.log_dir = self.ckp.get_path('tensorboard_logs')
        if self.args.start_wandb:
            wandb.init(project=args.save, entity="stefen")
        self.current_test_iteration = 0
        utility.count_parameters(self.model, 'This model')
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8


    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.param_groups[0]['lr']
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        losses = []
#        print(f"Successfuly loaded {len(self.loader_train)} training data")
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            losses.append(loss.item())
            if self.args.start_tensorboard:
                writer.add_scalar('loss', loss, batch)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            timer_data.tic()
        # Log the loss for this batch to Wandb
        avg_loss = np.mean(losses)
        if self.args.start_wandb:
            wandb.log({"train_loss": avg_loss}, step=epoch)
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        test_count = 0
        ssimes = []
        psnres = []
        sr_list = []
        avg_ssim = 0
        avg_psnr = 0
#        print(f"Successfuly loaded {len(self.loader_test)} data")
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    if self.args.using_npy or self.args.using_mat:
                        matrix = hr[0][0]
                        max_value = torch.max(matrix) # Calcuate the rgb range for ssim calculation
                        max_value = "{:.4f}".format(max_value.item())
                        max_value = float(max_value)
                        ssim = utility.calc_ssim(sr, hr, max_value)
                        psnr = utility.calc_psnr(sr, hr, scale, max_value, dataset=d) 
                        self.ckp.log[-1, idx_data, idx_scale] += psnr
                    else:
                        sr = utility.quantize(sr, self.args.rgb_range)
                        ssim = utility.calc_ssim(sr, hr, self.args.rgb_range)
                        psnr = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)
                        self.ckp.log[-1, idx_data, idx_scale] += psnr
#                    save_list = [sr, lr, hr]
                    ssimes.append(ssim)
                    psnres.append(psnr)
                    avg_ssim = np.mean(ssimes)
                    avg_psnr = np.mean(psnres)
                    save_list = [sr]
                    sr_list.append(sr)                  
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)
                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} SSIM: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        avg_ssim,
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        if self.args.start_wandb:
            wandb.log({"SSIM": avg_ssim,
                       "PSNR": avg_psnr,
                       "samples": [wandb.Image(sample) for sample in sr_list]})
        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)
         
    def prepare(self, *args):
        if self.args.cpu:
            device = torch.device('cpu')
        else:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch > self.args.epochs

