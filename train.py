import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

from configs import cfg
import torch
import numpy as np

from core.utils.log_util import Logger
from core.data import create_dataloader
from core.nets import create_network
from core.train import create_trainer, create_optimizer




def main():
    log = Logger()
    log.print_config()
    model = create_network()
    train_loader = create_dataloader('train')
    

    print('TRAIN DATA LENGTH:', len(train_loader.dataset))
    # update avg beta to model
    if hasattr(model, 'generate_neural_points'):
        model.generate_neural_points(train_loader.dataset.avg_betas)
    
    # update motion wieghts prior
    if hasattr(model.mweight_vol_decoder, 'matrix'):
        model.mweight_vol_decoder.matrix.data = torch.log(torch.tensor(np.asarray(train_loader.dataset.motion_weights_priors).copy()))
        print('motion_weights_priors loaded!')

    optimizer = create_optimizer(model)
    trainer = create_trainer(model, optimizer)
    
    # estimate start epoch
    epoch = trainer.iter // len(train_loader) + 1
    while True:
        if trainer.iter > cfg.train.maxiter:
            break
        
        trainer.train(epoch=epoch,
                      train_dataloader=train_loader)
        epoch += 1

    trainer.finalize()

if __name__ == '__main__':
    main()
