import os
import sys

from shutil import copyfile

import pickle
import yaml
import numpy as np
from tqdm import tqdm

from pathlib import Path

sys.path.append(str(Path(os.getcwd()).resolve().parents[1]))

from third_parties.smpl.smpl_numpy import SMPL
from core.utils.file_util import split_path
from core.utils.image_util import load_image, save_image, to_3ch_image

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('cfg',
                    '387_eval.yaml',
                    'the path of config file')

MODEL_DIR = '../../third_parties/smpl/models'


def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)

    return config


def prepare_dir(output_path, name):
    out_dir = os.path.join(output_path, name)
    os.makedirs(out_dir, exist_ok=True)

    return out_dir


def get_mask(subject_dir, img_name):
    msk_path = os.path.join(subject_dir, 'mask',
                            img_name)[:-4] + '.png'
    msk = np.array(load_image(msk_path))[:, :, 0]
    msk = (msk != 0).astype(np.uint8)

    msk_path = os.path.join(subject_dir, 'mask_cihp',
                            img_name)[:-4] + '.png'
    msk_cihp = np.array(load_image(msk_path))[:, :, 0]
    msk_cihp = (msk_cihp != 0).astype(np.uint8)

    msk = (msk | msk_cihp).astype(np.uint8)
    msk[msk == 1] = 255

    return msk


def main(argv):
    del argv  # Unused.

    cfg = parse_config()
    subject = cfg['dataset']['subject']
    sex = cfg['dataset']['sex']
    max_frames = cfg['max_frames']

    dataset_dir = cfg['dataset']['zju_mocap_path']
    subject_dir = os.path.join(dataset_dir, f"CoreView_{subject}")
    smpl_params_dir = os.path.join(subject_dir, "new_params")

    select_view = []
    views = cfg['eval_view'].split(',')
    for view_range in views:
        view_range = view_range.strip()
        index = view_range.find('-')

        if index == -1:
            view_range = int(view_range)
            if view_range < 0 or view_range > 22:
                print(f'eval view invaild! camera index {view_range} is given!')
            else:
                select_view.append(view_range)
        else:
            # view_range is a real range
            for i in range(int(view_range[:index]),int(view_range[index+1:])+1):
                select_view.append(i)
    select_view = np.array(select_view)

    anno_path = os.path.join(subject_dir, 'annots.npy')
    annots = np.load(anno_path, allow_pickle=True).item()

    # load cameras
    cams = annots['cams']
    cam_Ks = np.array(cams['K'])[select_view].astype('float32')  # view_num*3*3
    cam_Rs = np.array(cams['R'])[select_view].astype('float32')  # view_num*3*3
    cam_Ts = np.array(cams['T'])[select_view].astype('float32') / 1000.  # view_num*3*1
    cam_Ds = np.array(cams['D'])[select_view].astype('float32')  # view_num*5*1

    K = cam_Ks  # view_num*3*3
    D = cam_Ds[..., 0]  # view_num*5
    E = np.zeros((cam_Ks.shape[0], 4, 4)).astype('float32')  # view_num*4*4
    cam_T = cam_Ts[..., 0]
    E[:, :3, :3] = cam_Rs
    E[:, :3, 3] = cam_T
    E[:, 3, 3] = 1.  # view_num*4*4

    # load image paths
    img_path_frames_views = annots['ims']
    img_paths = np.array([
        np.array(multi_view_paths['ims'])[select_view] \
        for multi_view_paths in img_path_frames_views
    ])
    img_paths = np.stack(img_paths, 0)  # change 2 dim list to matrix (frame_num, view_num)
    if max_frames > 0:
        img_paths = img_paths[:max_frames]
    # skip some frame
    index_keep = np.array(range(0, len(img_paths), cfg['skip']))
    img_paths = img_paths[index_keep]

    output_path = os.path.join(cfg['output']['dir'],
                               subject if 'name' not in cfg['output'].keys() else cfg['output']['name'])
    os.makedirs(output_path, exist_ok=True)
    out_img_dir = prepare_dir(output_path, 'images')
    out_mask_dir = prepare_dir(output_path, 'masks')

    # copy config file
    copyfile(FLAGS.cfg, os.path.join(output_path, 'config.yaml'))

    smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)

    cameras = {}
    mesh_infos = {}
    all_betas = []
    [all_betas.append([]) for i in range(len(img_paths))]

    for idx_frame, path_frame in enumerate(tqdm(img_paths)):
        for idx_camera, path_camera in enumerate(path_frame):
            real_idx_frame = idx_frame * cfg['skip']
            real_idx_camera = select_view[idx_camera].item() + 1
            out_name = 'camera_{:02d}_frame_{:06d}'.format(real_idx_camera, real_idx_frame)

            img_path = os.path.join(subject_dir, path_camera)

            # load image
            img = np.array(load_image(img_path))
            
            if subject in ['313', '315']:
                smpl_idx = real_idx_frame + 1  # index begin with 1
            else:
                smpl_idx = real_idx_frame

            # load smpl parameters
            smpl_params = np.load(
                os.path.join(smpl_params_dir, f"{smpl_idx}.npy"),
                allow_pickle=True).item()

            betas = smpl_params['shapes'][0]  # (10,)
            poses = smpl_params['poses'][0]  # (72,)
            Rh = smpl_params['Rh'][0]  # (3,)
            Th = smpl_params['Th'][0]  # (3,)

            all_betas[idx_frame].append(betas)

            # write camera info
            cameras[out_name] = {
                'intrinsics': K[idx_camera],
                'extrinsics': E[idx_camera],
                'distortions': D[idx_camera]
            }

            # write mesh info
            _, tpose_joints = smpl_model(np.zeros_like(poses), betas)
            _, joints = smpl_model(poses, betas)
            mesh_infos[out_name] = {
                'Rh': Rh,
                'Th': Th,
                'poses': poses,
                'betas': betas,
                'joints': joints,
                'tpose_joints': tpose_joints
            }

            # load and write mask
            mask = get_mask(subject_dir, path_camera)
            save_image(to_3ch_image(mask),
                       os.path.join(out_mask_dir, out_name + '.png'))

            # write image
            out_image_path = os.path.join(out_img_dir, '{}.png'.format(out_name))
            save_image(img, out_image_path)

    # write camera infos
    with open(os.path.join(output_path, 'cameras.pkl'), 'wb') as f:
        pickle.dump(cameras, f)

    # write mesh infos
    with open(os.path.join(output_path, 'mesh_infos.pkl'), 'wb') as f:
        pickle.dump(mesh_infos, f)

    # write canonical joints
    # eliminate duplicate values ​​in all_betas
    all_betas = [all_betas[i][0] for i in range(len(all_betas))]
    avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    smpl_model = SMPL(sex, model_dir=MODEL_DIR)
    _, template_joints = smpl_model(np.zeros(72), avg_betas)
    with open(os.path.join(output_path, 'canonical_joints.pkl'), 'wb') as f:
        pickle.dump(
            {
                'joints': template_joints,
                'avg_betas': avg_betas
            }, f)


if __name__ == '__main__':
    app.run(main)