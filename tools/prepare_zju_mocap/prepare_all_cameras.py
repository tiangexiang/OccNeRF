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
                    '387.yaml',
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

    select_view = cfg['training_view']

    anno_path = os.path.join(subject_dir, 'annots.npy')
    annots = np.load(anno_path, allow_pickle=True).item()
    
    # load cameras
    cams = annots['cams']
    cam_Ks = np.array(cams['K']).astype('float32')
    cam_Rs = np.array(cams['R']).astype('float32')
    cam_Ts = np.array(cams['T']).astype('float32') / 1000.
    cam_Ds = np.array(cams['D']).astype('float32')

    print('toatl %d cameras' % cam_Ks.shape[0])

    K = cam_Ks     #(N, 3, 3)
    D = cam_Ds[:, :, 0]
    E = np.repeat(np.eye(4)[None], K.shape[0], axis=0)  #(N, 4, 4)
    cam_T = cam_Ts[:, :3, 0]
    E[:, :3, :3] = cam_Rs
    E[:, :3, 3]= cam_T
    print(K.shape, E.shape, D.shape)
    
    # load image paths
    img_path_frames_views = annots['ims']
    img_paths = np.array([
        np.array(multi_view_paths['ims'])[select_view] \
            for multi_view_paths in img_path_frames_views
    ])
    if max_frames > 0:
        img_paths = img_paths[:max_frames]

    output_path = os.path.join(cfg['output']['dir'],
                               subject if 'name' not in cfg['output'].keys() else cfg['output']['name'])
    os.makedirs(output_path, exist_ok=True)

    cameras = {}
    for idx, ipath in enumerate(tqdm(img_paths)):
        out_name = 'frame_{:06d}'.format(idx)
        # write camera info
        cameras[out_name] = {
                'intrinsics': K,
                'extrinsics': E,
                'distortions': D
        }

    # write camera infos
    with open(os.path.join(output_path, 'all_cameras.pkl'), 'wb') as f:   
        pickle.dump(cameras, f)
    

if __name__ == '__main__':
    app.run(main)
