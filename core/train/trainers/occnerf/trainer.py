import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from third_parties.lpips import LPIPS

from core.train import create_lr_updater
from core.data import create_dataloader
from core.utils.network_util import set_requires_grad
from core.utils.train_util import cpu_data_to_gpu, Timer
from core.utils.image_util import tile_images, to_8b_image

from configs import cfg

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2l1 = lambda x, y : torch.mean(torch.abs(x-y))
to8b = lambda x : (255.*np.clip(x,0.,1.)).astype(np.uint8)

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height']


def _unpack_imgs(rgbs, patch_masks, bgcolor, targets, div_indices):
    N_patch = len(div_indices) - 1
    assert patch_masks.shape[0] == N_patch
    assert targets.shape[0] == N_patch

    patch_imgs = bgcolor.expand(targets.shape).clone() # (N_patch, H, W, 3)
    #print(patch_imgs.shape, rgbs.shape, patch_masks[0].shape)
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i]] = rgbs[div_indices[i]:div_indices[i+1]]

    return patch_imgs


def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.


class Trainer(object):
    def __init__(self, network, optimizer):
        print('\n********** Init Trainer ***********')

        network = network.cuda().deploy_mlps_to_secondary_gpus()
        self.network = network

        self.optimizer = optimizer
        self.update_lr = create_lr_updater()

        if cfg.resume and Trainer.ckpt_exists(cfg.load_net):
            self.load_ckpt(f'{cfg.load_net}')
        else:
            self.iter = 0
            self.save_ckpt('init')
            self.iter = 1

        self.timer = Timer()

        if "lpips" in cfg.train.lossweights.keys():
            self.lpips = LPIPS(net='vgg')
            set_requires_grad(self.lpips, requires_grad=False)
            self.lpips = nn.DataParallel(self.lpips).cuda()

        print("Load Progress Dataset ...")
        self.prog_dataloader = create_dataloader(data_type='progress')

        print('************************************')

        # if hasattr(network, 'point_cloud'):
        #     self.prev_points = network.point_cloud.detach().cpu().numpy()
        self.prev_points = None

    @staticmethod
    def get_ckpt_path(name):
        return os.path.join(cfg.logdir, f'{name}.tar')

    @staticmethod
    def ckpt_exists(name):
        return os.path.exists(Trainer.get_ckpt_path(name))

    ######################################################3
    ## Training 

    def get_img_rebuild_loss(self, loss_names, rgb, target, occupancy=None, sigma=None):
        losses = {}

        if "mse" in loss_names:
            losses["mse"] = img2mse(rgb, target)

        if "l1" in loss_names:
            losses["l1"] = img2l1(rgb, target)

        if "lpips" in loss_names:
            lpips_loss = self.lpips(scale_for_lpips(rgb.permute(0, 3, 1, 2)), 
                                    scale_for_lpips(target.permute(0, 3, 1, 2)))
            losses["lpips"] = torch.mean(lpips_loss)

        return losses

    def get_sigma_loss(self, distance, threshold, scale=0.1):
        # distance: N * 2
        # thred: float
        distance = F.relu(distance - threshold)
        loss = torch.exp(distance * scale) - 1.
        return torch.max(loss)

    def compute_tv_norm(self, values, losstype='l2', weighting=None):  # pylint: disable=g-doc-args
        """Returns TV norm for input values.
        Note: The weighting / masking term was necessary to avoid degenerate
        solutions on GPU; only observed on individual DTU scenes.
        """
        v00 = values[:, :-1, :-1]
        v01 = values[:, :-1, 1:]
        v10 = values[:, 1:, :-1]

        if losstype == 'l2':
            loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
        elif losstype == 'l1':
            loss = torch.abs(v00 - v01) + torch.abs(v00 - v10)
        else:
            raise ValueError('Not supported losstype.')

        if weighting is not None:
            loss = loss * weighting
        return loss

    def get_loss(self, net_output, 
                 patch_masks, bgcolor, targets, div_indices):

        lossweights = cfg.train.lossweights
        loss_names = list(lossweights.keys())

        rgb = net_output['rgb']
        losses = self.get_img_rebuild_loss(
                        loss_names, 
                        _unpack_imgs(rgb, patch_masks, bgcolor,
                                     targets, div_indices), 
                        targets)


        use_tv_loss = False
        if use_tv_loss:
            #print(rgb.shape, net_output['depth'].shape, net_output['alpha'].shape)
            N_patch = len(div_indices) - 1
            #patch_imgs = bgcolor.expand(targets.shape).clone() # (N_patch, H, W, 3)
            patch_depth = torch.zeros(N_patch, patch_masks.shape[1], patch_masks.shape[2]).to(targets)
            patch_acc = torch.zeros(N_patch, patch_masks.shape[1], patch_masks.shape[2]).to(targets)
            for i in range(N_patch):
                patch_depth[i, patch_masks[i]] = net_output['depth'][div_indices[i]:div_indices[i+1]]
                patch_acc[i, patch_masks[i]] = net_output['alpha'][div_indices[i]:div_indices[i+1]]

            # patch_depth = patch_depth.view(-1, patch_masks.shape[1], patch_masks.shape[2])
            # patch_acc = patch_acc.view(-1, patch_masks.shape[1], patch_masks.shape[2])
            tv_loss = self.compute_tv_norm(patch_depth, weighting=patch_acc[:,:-1,:-1].detach())
            tv_loss = torch.mean(tv_loss)
            losses.update({'tv_loss': tv_loss})
            lossweights.update({'tv_loss': 1.0})

        if 'comp_loss' in net_output:
            comp_loss = torch.mean(net_output['comp_loss'])
            losses.update({'comp_loss': comp_loss})
            lossweights.update({'comp_loss': 1.0})
        if 'alpha_loss' in net_output:
            sigma_loss = torch.mean(net_output['alpha_loss'])
            losses.update({'alpha_loss': sigma_loss})
            lossweights.update({'alpha_loss': 1.0})
        if 'kd_loss' in net_output:
            surface_loss = torch.mean(net_output['kd_loss'])
            losses.update({'kd_loss': surface_loss})
            lossweights.update({'kd_loss': 1.0})
        if 'surface_loss' in net_output:
            surface_loss = torch.mean(net_output['surface_loss'])
            losses.update({'surface_loss': surface_loss})
            lossweights.update({'surface_loss': 1.0})

        train_losses = [
            weight * losses[k] for k, weight in lossweights.items()
        ]

        return sum(train_losses), \
               {loss_names[i]: train_losses[i] for i in range(len(loss_names))}

    def train_begin(self, train_dataloader):
        assert train_dataloader.batch_size == 1

        self.network.train()
        cfg.perturb = cfg.train.perturb

    def train_end(self):
        pass

    def train(self, epoch, train_dataloader):
        self.train_begin(train_dataloader=train_dataloader)

        ########### prune point cloud ##########
        if hasattr(self.network, 'point_cloud_mask'):
            self.network.point_cloud_mask *= 0.


        self.timer.begin()
        for batch_idx, batch in enumerate(train_dataloader):
            if self.iter > cfg.train.maxiter:
                break

            if hasattr(self.network, 'grow_point'):
                #if self.iter > 1900 and self.iter % 100 == 99:
                if self.iter > 0 and self.iter % 20 == 19:
                    self.network.grow_point = True
                else:
                    self.network.grow_point = False

            self.optimizer.zero_grad()

            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]

            batch['iter_val'] = torch.full((1,), self.iter)
            data = cpu_data_to_gpu(
                batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)
            data['iter'] = self.iter
            net_output = self.network(**data)

            if net_output is None:
                continue

            if 'rgb' not in net_output:
                print('NO RGB!')
                continue
            
            train_loss, loss_dict = self.get_loss(
                net_output=net_output,
                patch_masks=data['patch_masks'],
                bgcolor=data['bgcolor'] / 255.,
                targets=data['target_patches'],
                div_indices=data['patch_div_indices'])

            train_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

            if self.iter % cfg.train.log_interval == 0:
                loss_str = f"Loss: {train_loss.item():.4f} ["
                for k, v in loss_dict.items():
                    loss_str += f"{k}: {v.item():.4f} "
                loss_str += "]"

                log_str = 'Epoch: {} [Iter {}, {}/{} ({:.0f}%), {}] {}'
                log_str = log_str.format(
                    epoch, self.iter,
                    batch_idx * cfg.train.batch_size, len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), 
                    self.timer.log(),
                    loss_str)
                print(log_str)

            is_reload_model = False
            
            #if self.iter in [500, 1000, 2500] or \
            if self.iter in [20, 100, 300, 1000, 2500] or \
                self.iter % cfg.progress.dump_interval == 0:
                is_reload_model = self.progress()
                is_reload_model = False

            if not is_reload_model:
                if self.iter % cfg.train.save_checkpt_interval == 0:
                    self.save_ckpt('latest')

                if cfg.save_all:
                    if self.iter % cfg.train.save_model_interval == 0:
                        self.save_ckpt(f'iter_{self.iter}')

                self.update_lr(self.optimizer, self.iter)

                self.iter += 1


    def finalize(self):
        self.save_ckpt('latest')

    ######################################################3
    ## Progress

    def progress_begin(self):
        self.network.eval()
        cfg.perturb = 0.

    def progress_end(self):
        self.network.train()
        cfg.perturb = cfg.train.perturb

    def progress(self):
        self.progress_begin()

        if hasattr(self.network, 'point_cloud'):
            print('Saving Neural Points ...')
            
            points = self.network.point_cloud.detach().cpu().numpy() # N, 3

            if self.prev_points is not None:
                print('total change:', np.sum(np.sum(np.abs(self.prev_points - points))))
            self.prev_points = points

            fig = plt.figure()
            # syntax for 3-D projection
            ax = plt.axes(projection ='3d')
            if hasattr(self.network, 'point_counter'):
                c = self.network.point_counter.detach().clone()
                c = c - c.min()
                c = c / c.max()
                c = c.cpu().squeeze().numpy() 
                c = plt.cm.jet(c)
                ax.scatter(points[:,0], points[:,1], points[:,2], c = c, s=1)
            else:
                ax.scatter(points[:,0], points[:,1], points[:,2], c = 'green', s=1)


            ax.view_init(60, 60)

            ax.set_aspect('equal')

            plt.savefig(os.path.join(cfg.logdir, "neural_points_{:06}.jpg".format(self.iter)))
            plt.close()

        print('Evaluate Progress Images ...')

        images = []
        is_empty_img = False

        for _, batch in enumerate(tqdm(self.prog_dataloader)):

            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]

            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']

            rendered = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)/255., 
                        dtype='float32')
            truth = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)/255., 
                        dtype='float32')

            batch['iter_val'] = torch.full((1,), self.iter)
            data = cpu_data_to_gpu(
                    batch, exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])
            with torch.no_grad():
                data['iter'] = self.iter
                net_output = self.network(**data)

            rgb = net_output['rgb'].data.to("cpu").numpy()
            target_rgbs = batch['target_rgbs']

            rendered[ray_mask] = rgb
            truth[ray_mask] = target_rgbs

            truth = to_8b_image(truth.reshape((height, width, -1)))
            rendered = to_8b_image(rendered.reshape((height, width, -1)))
            images.append(np.concatenate([rendered, truth], axis=1))

             # check if we create empty images (only at the begining of training)
            if self.iter <= 5000 and \
                np.allclose(rendered, np.array(cfg.bgcolor), atol=3.):
                is_empty_img = True
                break

        tiled_image = tile_images(images)
        
        Image.fromarray(tiled_image).save(
            os.path.join(cfg.logdir, "prog_{:06}.jpg".format(self.iter)))

        if is_empty_img:
            print("Produce empty images!")
            #print("Produce empty images; reload the init model.")
            #self.load_ckpt('init')
            
        self.progress_end()

        return is_empty_img


    ######################################################3
    ## Utils

    def save_ckpt(self, name):
        path = Trainer.get_ckpt_path(name)
        print(f"Save checkpoint to {path} ...")

        torch.save({
            'iter': self.iter,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load_ckpt(self, name):
        path = Trainer.get_ckpt_path(name)
        print(f"Load checkpoint from {path} ...")
        
        ckpt = torch.load(path, map_location='cuda:0')
        self.iter = ckpt['iter'] + 1

        #handle point cloud
        if hasattr(self.network, 'point_cloud'):
            current_point_cloud_size = self.network.point_cloud.shape
            #print(ckpt['network'].keys())
            if 'point_cloud' in ckpt['network']:
                ckpt_point_cloud_size = ckpt['network']['point_cloud'].shape
                self.network.point_cloud.data = torch.zeros(*ckpt_point_cloud_size).float().to(self.network.point_cloud.device)
            if 'neural_point' in ckpt['network']:
                ckpt_neural_point_size = ckpt['network']['neural_point'].shape
                self.network.neural_point.data = torch.zeros(*ckpt_neural_point_size).float().to(self.network.neural_point.device)
            if 'point_counter' in ckpt['network']:
                ckpt_point_cloud_size = ckpt['network']['point_counter'].shape
                self.network.point_counter.data = torch.zeros(*ckpt_point_cloud_size).float().to(self.network.point_counter.device)

        self.network.load_state_dict(ckpt['network'], strict=True)
        self.optimizer.load_state_dict(ckpt['optimizer'])
