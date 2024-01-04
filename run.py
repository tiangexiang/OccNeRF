import os

import torch
import numpy as np
from tqdm import tqdm

from configs import cfg, args
cfg.bgcolor = [255., 255., 255.]

from core.data import create_dataloader
from core.nets import create_network
from core.utils.train_util import cpu_data_to_gpu
from core.utils.image_util import ImageWriter, to_8b_image, to_8b3ch_image




EXCLUDE_KEYS_TO_GPU = ['frame_name',
                       'img_width', 'img_height', 'ray_mask']

def PSNR(img1, img2, scale=255.):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(scale / torch.sqrt(mse))


def load_network(model):

    ckpt_path = os.path.join(cfg.logdir, f'{cfg.load_net}.tar')
    ckpt = torch.load(ckpt_path, map_location='cuda:0')

    if 'point_cloud' in ckpt['network']:
        ckpt_point_cloud_size = ckpt['network']['point_cloud'].shape
        model.point_cloud.data = torch.zeros(*ckpt_point_cloud_size).float()

    model.load_state_dict(ckpt['network'], strict=True)
    print('load network from ', ckpt_path)
    return model.cuda().deploy_mlps_to_secondary_gpus()


def unpack_alpha_map(alpha_vals, ray_mask, width, height):
    alpha_map = np.zeros((height * width), dtype='float32')
    alpha_map[ray_mask] = alpha_vals
    return alpha_map.reshape((height, width))


def unpack_to_image(width, height, ray_mask, bgcolor,
                    rgb, alpha, truth=None):

    rgb_image = np.full((height * width, 3), bgcolor, dtype='float32')
    truth_image = np.full((height * width, 3), bgcolor, dtype='float32')

    rgb_image[ray_mask] = rgb

    rgb_image = to_8b_image(rgb_image.reshape((height, width, 3)))

    if truth is not None:
        truth_image[ray_mask] = truth
        truth_image = to_8b_image(truth_image.reshape((height, width, 3)))

    alpha_map = unpack_alpha_map(alpha, ray_mask, width, height)
    alpha_image  = to_8b3ch_image(alpha_map)

    return rgb_image, alpha_image, truth_image


def _freeview(
        data_type='freeview',
        folder_name=None):
    cfg.perturb = 0.

    model = create_network()
    
    test_loader = create_dataloader(data_type)

    if hasattr(model, 'generate_neural_points'):
        model.generate_neural_points(test_loader.dataset.avg_betas)

    model = load_network(model)
    writer = ImageWriter(
                output_dir=os.path.join(cfg.logdir, cfg.load_net),
                exp_name=folder_name)

    model.eval()
    for batch in tqdm(test_loader):
        for k, v in batch.items():
            batch[k] = v[0]

        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU)

        with torch.no_grad():
            net_output = model(**data, 
                               iter_val=cfg.eval_iter)

        rgb = net_output['rgb']
        alpha = net_output['alpha']

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']
        target_rgbs = batch.get('target_rgbs', None)

        rgb_img, alpha_img, _ = unpack_to_image(
            width, height, ray_mask, np.array(cfg.bgcolor) / 255.,
            rgb.data.cpu().numpy(),
            alpha.data.cpu().numpy())

        imgs = [rgb_img]
        if cfg.show_truth and target_rgbs is not None:
            target_rgbs = to_8b_image(target_rgbs.numpy())
            imgs.append(target_rgbs)
        if cfg.show_alpha:
            imgs.append(alpha_img)

        img_out = np.concatenate(imgs, axis=1)
        writer.append(img_out)

    writer.finalize()


def run_freeview():
    _freeview(
        data_type='freeview',
        folder_name=f"freeview_{cfg.freeview.frame_idx}" \
            if not cfg.render_folder_name else cfg.render_folder_name)


def run_tpose():
    cfg.ignore_non_rigid_motions = True
    _freeview(
        data_type='tpose',
        folder_name='tpose' \
            if not cfg.render_folder_name else cfg.render_folder_name)


def run_movement(render_folder_name='movement'):
    cfg.perturb = 0.

    model = create_network()
    
    test_loader = create_dataloader('movement')
    writer = ImageWriter(
        output_dir=os.path.join(cfg.logdir, cfg.load_net),
        exp_name=render_folder_name)

    model = load_network(model)
    model.eval()
    for idx, batch in enumerate(tqdm(test_loader)):
        for k, v in batch.items():
            batch[k] = v[0]

        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])

        with torch.no_grad():
            net_output = model(**data, iter_val=cfg.eval_iter)

        rgb = net_output['rgb']
        alpha = net_output['alpha']

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']

        rgb_img, alpha_img, truth_img = \
            unpack_to_image(
                width, height, ray_mask, np.array(cfg.bgcolor)/255.,
                rgb.data.cpu().numpy(),
                alpha.data.cpu().numpy(),
                batch['target_rgbs'])

        imgs = [rgb_img]
        if cfg.show_truth:
            imgs.append(truth_img)
        if cfg.show_alpha:
            imgs.append(alpha_img)
            
        img_out = np.concatenate(imgs, axis=1)
        writer.append(img_out, img_name=f"{idx:06d}")
    
    writer.finalize()

def run_allview():
    _freeview(
        data_type='allview',
        folder_name=f"allview_{cfg.freeview.frame_idx}" \
            if not cfg.render_folder_name else cfg.render_folder_name)

def run_evaluate():
    
    cfg.perturb = 0.

    model = create_network()
    
    prog_dataloader = create_dataloader(data_type='progress', evaluate=True)

    if hasattr(model, 'generate_neural_points'):
        model.generate_neural_points(prog_dataloader.dataset.avg_betas)

    model = load_network(model)
    model.eval()
    psnrs = []
    skips = [4, 15]
    for idx, batch in enumerate(tqdm(prog_dataloader)):
        if idx in skips:
            continue
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

        batch['iter_val'] = torch.full((1,), 1)
        data = cpu_data_to_gpu(
                batch, exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])
        with torch.no_grad():
            data['iter'] = 1
            net_output = model(**data)

        rgb = net_output['rgb'].data.to("cpu").numpy()
        target_rgbs = batch['target_rgbs']


        rendered[ray_mask] = rgb
        truth[ray_mask] = target_rgbs

        psnr = PSNR(torch.tensor(rgb), torch.tensor(target_rgbs), 1.)
        psnrs.append(torch.mean(psnr))

    print('AVG PSNR %.4f' % torch.mean(torch.stack(psnrs)))
        
if __name__ == '__main__':
    globals()[f'run_{args.type}']()
