from configs import cfg

class DatasetArgs(object):
    dataset_attrs = {}

    subjects = ['313', '315', '377', '386', '387', '390', '392', '393', '394']

    if cfg.category == 'occnerf' and cfg.task == 'zju_mocap':
        for sub in subjects:
            if not hasattr(cfg, 'eval') or not cfg.eval:
                dataset_attrs.update({
                    f"zju_{sub}_train": {
                        "dataset_path": f"dataset/zju_mocap/{sub}",
                        "keyfilter": cfg.train_keyfilter,
                        "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    },
                    f"zju_{sub}_test": {
                        "dataset_path": f"dataset/zju_mocap/{sub}", 
                        "keyfilter": cfg.test_keyfilter,
                        "ray_shoot_mode": 'image',
                        "src_type": 'zju_mocap'
                    },
                })
            else:
                dataset_attrs.update({
                    f"zju_{sub}_train": {
                        "dataset_path": f"dataset/zju_mocap/processed/{sub}",
                        "keyfilter": cfg.train_keyfilter,
                        "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    },
                    f"zju_{sub}_test": {
                        "dataset_path": f"dataset/zju_mocap/processed/{sub}_eval",
                        "keyfilter": cfg.test_keyfilter,
                        "ray_shoot_mode": 'image',
                        "src_type": 'zju_mocap'
                    },
                })


    if cfg.category == 'occnerf' and cfg.task == 'wild':
        dataset_attrs.update({
            "monocular_train": {
                "dataset_path": 'dataset/wild/' + cfg.subject,
                "keyfilter": cfg.train_keyfilter,
                "ray_shoot_mode": cfg.train.ray_shoot_mode,
            },
            "monocular_test": {
                "dataset_path": 'dataset/wild/' + cfg.subject,  
                "keyfilter": cfg.test_keyfilter,
                "ray_shoot_mode": 'image',
                "src_type": 'wild'
            },
        })


    @staticmethod
    def get(name):
        attrs = DatasetArgs.dataset_attrs[name]
        return attrs.copy()
