import os
import sys

import json
import yaml
import pickle
import numpy as np
from tqdm import tqdm

from pathlib import Path
sys.path.append(str(Path(os.getcwd()).resolve().parents[1]))
from third_parties.smpl.smpl_numpy import SMPL

from absl import app
from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_string('cfg',
                    'wild.yaml',
                    'the path of config file')

MODEL_DIR = '../../third_parties/smpl/models'


def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)

    return config


def main(argv):
    del argv  # Unused.

    cfg = parse_config()
    subject = cfg['dataset']['subject']
    sex = cfg['dataset']['sex']

    dataset_dir = cfg['dataset']['path']
    subject_dir = os.path.join(dataset_dir, subject)
    output_path = subject_dir
    
    frame_infoss = []
    for camera_idx in range(6):
        with open(os.path.join(subject_dir, 'metadata_%d.json' % camera_idx), 'r') as f:
            frame_infoss.append(json.load(f))

    smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)

    # all_cameras = []
    # all_mesh_infos = []
    # all_all_betas = []

    cameras = {}
    mesh_infos = {}
    all_betas = []
    
    for frame_base_name in tqdm(frame_infoss[0]):
        Ks = []
        Es = []
        Rhs= []
        Ths = []
        posess = []
        betass = []
        jointss = []
        tpose_jointss = []
        tmp_betas = []
        for frame_infos in frame_infoss:
            cam_body_info = frame_infos[frame_base_name] 
            poses = np.array(cam_body_info['poses'], dtype=np.float32)
            betas = np.array(cam_body_info['betas'], dtype=np.float32)
            K = np.array(cam_body_info['cam_intrinsics'], dtype=np.float32)
            E = np.array(cam_body_info['cam_extrinsics'], dtype=np.float32)

            if 'trans' in cam_body_info:
                trans = np.array(cam_body_info['trans'], dtype=np.float32)
            else:
                trans = None
            
            tmp_betas.append(betas)

            ##############################################
            # Below we tranfer the global body rotation to camera pose

            # Get T-pose joints
            _, tpose_joints = smpl_model(np.zeros_like(poses), betas)

            # get global Rh, Th
            pelvis_pos = tpose_joints[0].copy()
            Th = pelvis_pos
            Rh = poses[:3].copy()

            # get refined T-pose joints
            tpose_joints = tpose_joints - pelvis_pos[None, :]

            # remove global rotation from body pose
            poses[:3] = 0

            # get posed joints using body poses without global rotation
            _, joints = smpl_model(poses, betas)
            joints = joints - pelvis_pos[None, :]

            Rhs.append(Rh)
            Ths.append(Th)
            posess.append(poses)
            betass.append(betas)
            jointss.append(joints)
            tpose_jointss.append(tpose_joints)

            # mesh_infos[frame_base_name] = {
            #     'Rh': Rh,
            #     'Th': Th,
            #     'poses': poses,
            #     'betas': betas,
            #     'joints': joints,
            #     'tpose_joints': tpose_joints
            # }

            if trans is not None:
                # w 2 c
                w2c = np.zeros((4,4))
                w2c[:3,:3] = np.eye(3)
                w2c[-1,-1] = 1.
                w2c[:3, -1] = trans
                #print(w2c)
                # print("============")
                # print(E)
                E = E @ w2c
                # print(E)
                #print("============")
                #print(E.shape, trans.shape)
                #E = E
                # mesh_infos[frame_base_name].update({
                #     'trans': trans
                # })
            Ks.append(K)
            Es.append(E)
            # cameras[frame_base_name] = {
            #     'intrinsics': K,
            #     'extrinsics': E
            # }
        cameras[frame_base_name] = {
                'intrinsics': np.stack(Ks, axis=0),
                'extrinsics': np.stack(Es, axis=0)
            }

        mesh_infos[frame_base_name] = {
                'Rh': np.stack(Rhs, axis=0),
                'Th': np.stack(Ths, axis=0),
                'poses': np.stack(posess, axis=0),
                'betas': np.stack(betass, axis=0),
                'joints': np.stack(jointss, axis=0),
                'tpose_joints': np.stack(tpose_jointss, axis=0),
            }
        
        all_betas.append(np.mean(np.stack(tmp_betas, axis=0), axis=0))
        
        # all_cameras.append(cameras)
        # all_mesh_infos.append(mesh_infos)
        # all_all_betas.append(np.mean(np.stack(all_betas, axis=0), axis=0))
        

    # write camera infos
    with open(os.path.join(output_path, 'all_cameras.pkl'), 'wb') as f:   
        pickle.dump(cameras, f)
        
    # write mesh infos
    with open(os.path.join(output_path, 'all_mesh_infos.pkl'), 'wb') as f:   
        pickle.dump(mesh_infos, f)

    # write canonical joints
    #avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    # smpl_model = SMPL(sex, model_dir=MODEL_DIR)
    # template_joints = []

    # for avg_betas in all_betas:
    #     _, tj = smpl_model(np.zeros(72), avg_betas)
    #     template_joints.append(tj)

    # with open(os.path.join(output_path, 'canonical_joints.pkl'), 'wb') as f:   
    #     pickle.dump(
    #         {
    #             'joints': np.stack(template_joints, axis=0),
    #             'avg_betas': np.stack(all_betas, axis=0),
    #         }, f)


if __name__ == '__main__':
    app.run(main)
