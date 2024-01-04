import os
import skimage

import torch
import numpy as np
from tqdm import tqdm
from configs import cfg
cfg.bgcolor = [255., 255., 255.]
cfg.eval = True

from core.data import create_dataloader
from core.nets import create_network
from core.utils.train_util import cpu_data_to_gpu
from core.utils.image_util import ImageWriter, to_8b_image, to_8b3ch_image
from matplotlib import pyplot as plt

# from torch.utils.tensorboard import SummaryWriter

from third_parties.lpips import LPIPS

EVAL_METHOD = 'vis' #'full'

EXCLUDE_KEYS_TO_GPU = ['frame_name',
                       'img_width', 'img_height', 'ray_mask']


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def load_network(loader):
    model = create_network()

    # update avg beta to model
    if hasattr(model, 'generate_neural_points'):
        model.generate_neural_points(loader.dataset.avg_betas)
    
    # update motion wieghts prior
    if hasattr(model.mweight_vol_decoder, 'matrix'):
        model.mweight_vol_decoder.matrix.data = torch.log(torch.tensor(np.asarray(loader.dataset.motion_weights_priors).copy()))
        print('motion_weights_priors loaded!')


    ckpt_path = os.path.join(cfg.logdir, f'{cfg.load_net}.tar')
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
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
    alpha_image = to_8b3ch_image(alpha_map)

    return rgb_image, alpha_image, truth_image, alpha_map

def psnr_metric(img_pred, img_gt):
    ''' Caculate psnr metric
        Args:
            img_pred: ndarray, W*H*3, range 0-1
            img_gt: ndarray, W*H*3, range 0-1

        Returns:
            psnr metric: scalar
    '''
    mse = np.mean((img_pred - img_gt) ** 2)
    psnr = -10 * np.log(mse) / np.log(10)
    return psnr.item()


def lpips_metric(model, pred, target):
    # convert range from 0-1 to -1-1
    processed_pred = torch.from_numpy(pred).float().unsqueeze(0).to(cfg.primary_gpus[0]) * 2. - 1.
    processed_target=torch.from_numpy(target).float().unsqueeze(0).to(cfg.primary_gpus[0]) * 2. - 1.

    lpips_loss = model(processed_pred.permute(0, 3, 1, 2),
                       processed_target.permute(0, 3, 1, 2))
    return torch.mean(lpips_loss).cpu().detach().item()

def eval_model(render_folder_name='eval', show_truth=True, show_alpha=True):
    
    cfg.perturb = 0.
    cfg.occlude = False
    test_loader = create_dataloader('movement', evaluate=True)
    model = load_network(test_loader)
    
    writer = ImageWriter(
        output_dir=os.path.join(cfg.logdir, cfg.load_net),
        exp_name=render_folder_name)
    log_dir = os.path.join(cfg.logdir, cfg.load_net, render_folder_name, 'log')
    #swriter = SummaryWriter(log_dir)

    model.eval()
    PSNRA = []
    SSIMA = []
    LPIPSA = []
    PSNRfull = []
    SSIMfull = []
    IOU = []
    PSNRbody = []
    SSIMbody = []
    # create lpip model and config
    lpips_model = LPIPS(net='vgg')
    set_requires_grad(lpips_model, requires_grad=False)
    lpips_model.to(cfg.primary_gpus[0])

    for idx, batch in enumerate(tqdm(test_loader)):

        for k, v in batch.items():
            batch[k] = v[0]

        data = cpu_data_to_gpu(
            batch,
            exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])

        with torch.no_grad():
            net_output = model(**data, iter_val=cfg.eval_iter)

        rgb = net_output['rgb']
        alpha = net_output['alpha'] # 0 - 1+, 0 is background
        #pred_alpha = alpha

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']


        # *_img: ndarray, (512, 512, 3), value range 0-255
        rgb_img, alpha_img, truth_img, alpha_map = \
            unpack_to_image(
                width, height, ray_mask, np.array(cfg.bgcolor) / 255.,
                rgb.data.cpu().numpy(),
                alpha.data.cpu().numpy(),
                batch['target_rgbs'])

        body_mask = np.zeros((height * width, 3), dtype='float32')
        body_mask[ray_mask] = 1.
        body_mask = body_mask.astype(np.bool)

        alpha_mask = alpha_map.reshape([width*height,]) > 0.001
        pred_alpha_mask = alpha_map.reshape([width*height,]) > 0.1

 
        if 'ray_alpha' in batch.keys():
            ray_alpha = batch['ray_alpha'][:,0].data.cpu().numpy()
            gt_mask = np.zeros((height * width), dtype='float32')
            gt_mask[ray_mask] = ray_alpha
            alpha_mask = gt_mask > 0.5 #.reshape((height, width))

        ####### completeness metric #######
        comp_mask = pred_alpha_mask.reshape([height, width])
        comp_pred = batch['alpha'][:,:,0].cpu().numpy()
        comp_pred = comp_pred > 0.5
        SMOOTH = 0.
        intersection = (comp_pred & comp_mask).sum()
        union = (comp_pred | comp_mask).sum()
        
        iou = (intersection + SMOOTH) / (union + SMOOTH)
        IOU.append(iou)


        imgs = [rgb_img]
        if show_truth:
            imgs.append(truth_img)
        if show_alpha:
            imgs.append(alpha_img)
        img_out = np.concatenate(imgs, axis=1)
        writer.append(img_out, img_name=batch['frame_name'])
        # convert image to 0-1
        rgb_img_norm = rgb_img / 255.
        truth_img_norm = truth_img / 255.

        body_psnr = psnr_metric(rgb_img_norm.reshape([width*height, 3])[body_mask], truth_img_norm.reshape([width*height, 3])[body_mask])
        vis_psnr = psnr_metric(rgb_img_norm.reshape([width*height, 3])[alpha_mask], truth_img_norm.reshape([width*height, 3])[alpha_mask])

        psnr = psnr_metric(rgb_img_norm, truth_img_norm)
        ssim, full_ssim = skimage.metrics.structural_similarity(rgb_img_norm, truth_img_norm, multichannel=True, full=True)

        full_ssim = full_ssim.reshape([width*height, 3])
        body_ssim = full_ssim[body_mask]
        body_ssim = np.mean(body_ssim)


        vis_ssim = full_ssim[alpha_mask]
        vis_ssim = np.mean(vis_ssim)

        print('PSNR-vis: %.4f, SSIM-vis: %.4f; PSNR-body: %.4f, SSIM-body: %.4f; PSNR-full: %.4f, SSIM-full: %.4f, IOU: %.4f' % (vis_psnr, vis_ssim, body_psnr, body_ssim, psnr, ssim, iou))
        PSNRA.append(vis_psnr)
        SSIMA.append(vis_ssim)
        PSNRbody.append(body_psnr)
        SSIMbody.append(body_ssim)
        PSNRfull.append(psnr)
        SSIMfull.append(ssim)
        IOU.append(iou)

    psnr_final = np.mean(PSNRA).item()
    ssim_final = np.mean(SSIMA).item()
    psnr_body_final = np.mean(PSNRbody).item()
    ssim_body_final = np.mean(SSIMbody).item()
    psnr_full_final = np.mean(PSNRfull).item()
    ssim_full_final = np.mean(SSIMfull).item()
    iou_final = np.mean(IOU)
    
    print('IOU', iou_final)
    print(f"PSNR_vis {psnr_final}, SSIM_vis {ssim_final}; PSNR_body {psnr_body_final}, SSIM_body {ssim_body_final}; PSNR_full {psnr_full_final}, SSIM_full {ssim_full_final}")



if __name__ == '__main__':
    eval_model(render_folder_name='eval')
