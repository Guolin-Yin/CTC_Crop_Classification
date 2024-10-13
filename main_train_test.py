import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from model.CNN import CNNModel
from model.Medianformer import Medianformer
from model.TFCNN import TFCNN
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from utils.datamodule import CTCDataModule
from utils.settings.config import RANDOM_SEED,  LINEAR_ENCODER, CLASS_WEIGHTS_EQUAL, CLASS_WEIGHT_PIXEL_SHUFFLED_CAP_2000
from utils.read_yml import read_yaml
# Set seed for everything
pl.seed_everything(RANDOM_SEED)
def make_new_run_path(results_path,num_epochs):
    run_ts = datetime.now().strftime("%Y-%m-%d-%Hh-%M:%S")
    run_path = results_path / f'run_{run_ts}'
    run_path.mkdir(exist_ok=True, parents=True)
    resume_from_checkpoint = None
    init_epoch = 0
    max_epoch = num_epochs
    return run_path, resume_from_checkpoint, max_epoch, init_epoch
def resume_or_start(results_path, train, num_epochs, load_checkpoint, **kwargs):
    if train:
        results_path = Path(results_path)
        if load_checkpoint:
            # Load the given checkpoint to resume training
            load_checkpoint = Path(load_checkpoint)

            run_path = load_checkpoint.parent.parent
            init_epoch = int(load_checkpoint.stem.split('=')[1].split('-')[0])
            max_epoch = init_epoch + num_epochs
            resume_from_checkpoint = load_checkpoint
        else:
            run_path, resume_from_checkpoint, max_epoch, init_epoch = make_new_run_path(results_path, num_epochs)
    else:
        assert load_checkpoint is not None, "Must provide a checkpoint to load for testing."
        # Load the given checkpoint to test with
        load_checkpoint = Path(load_checkpoint)
        run_path = load_checkpoint.parent.parent
        init_epoch = int(load_checkpoint.stem.split('=')[1].split('-')[0])
        max_epoch = init_epoch + 1
        resume_from_checkpoint = load_checkpoint

    return run_path, resume_from_checkpoint, max_epoch + 1, init_epoch
# def resume_or_start(results_path, resume, train, num_epochs, load_checkpoint):

#     results_path = Path(results_path)

#     if not train:
#         # Load the given checkpoint to test with
#         load_checkpoint = Path(load_checkpoint)
#         run_path = load_checkpoint.parent.parent
#         init_epoch = int(load_checkpoint.stem.split('=')[1].split('-')[0])
#         max_epoch = init_epoch + 1
#         resume_from_checkpoint = load_checkpoint
#     elif resume == 'last':
#         # Use last run's latest checkpoint to resume training
#         run_paths = sorted(results_path.glob('run_*'))
#         run_path = run_paths[-1]

#         epoch_ckpt = {int(x.stem.split('=')[-1]): x for x in (run_path / 'checkpoints').glob('*')}
#         init_epoch = sorted(epoch_ckpt.keys())[-1]
#         ckpt_path = epoch_ckpt[init_epoch]

#         init_epoch = int(init_epoch)
#         max_epoch = init_epoch + num_epochs
#         resume_from_checkpoint = ckpt_path
#     elif resume is not None:
#         # Load the given checkpoint to resume training
#         resume = Path(resume)
#         run_path = resume.parent.parent
#         init_epoch = int(resume.stem.split('=')[1].split('-')[0])
#         max_epoch = init_epoch + num_epochs
#         resume_from_checkpoint = resume
#     elif train:
#         # Create folder to save this run's results into
#         run_ts = datetime.now().strftime("%Y%m%d%H%M%S")
#         run_path = results_path / f'run_{run_ts}'
#         run_path.mkdir(exist_ok=True, parents=True)
#         resume_from_checkpoint = None
#         init_epoch = 0
#         max_epoch = num_epochs

#     return run_path, resume_from_checkpoint, max_epoch, init_epoch

def create_model_log_path(log_path, prefix, model, is_train=True):
    '''
    Creates the path to contain results for the given model.
    '''
        
    results_path = log_path / f'{model}' / f'{prefix}'
    if is_train:
        results_path.mkdir(exist_ok=True, parents=True)

    return results_path

def determine_prefix(args):
    """Determine the prefix based on the given arguments."""
    if args.prefix is None:
        # No prefix given, use current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d-%Hh%M")
        return timestamp
    else:
        return args.prefix

# def get_n_classes_encoding(CROP_ENCODING, LINEAR_ENCODER):
#     """Determine the number of classes based on the arguments."""
#     crop_encoding_rev = {v: k for k, v in CROP_ENCODING.items()}
#     crop_encoding     = {k: crop_encoding_rev[k] for k in LINEAR_ENCODER.keys() if k != 0}
#     crop_encoding[0]  = 'Background/Other'
#     return len(list(CROP_ENCODING.values())) + 1, crop_encoding



def get_data_module(args,collate_fn):
    """Get the appropriate data module based on the given arguments."""

    dm = CTCDataModule(
                        batch_size=args.batch_size, 
                        collate_fn = collate_fn
                        # num_workers=args.num_workers
                        )
    if args.train:
        dm.setup('fit')
    else:
        dm.setup('test')

    return dm

def get_callbacks(run_path):
    # Trainer callbacks
    callbacks = []
    early_stop_callback = EarlyStopping(
        monitor   = 'val_loss',
        min_delta = 0.00,    # Minimum change in the monitored quantity to qualify as an improvement
        patience  = 30,      # Number of epochs with no improvement after which training will be stopped
        verbose   = True,    # Whether to print logs to stdout
        mode      = 'min'             # `min` for minimization and `max` for maximization
    )
    ckp_saver = ModelCheckpoint(
        dirpath    = run_path / 'checkpoints',
        monitor    = 'val_loss',
        filename   = '{epoch}-{val_loss:.2f}',
        mode       = 'min',
        save_top_k = 3
    )
    step_checkpoint_saver = ModelCheckpoint(
        dirpath             = run_path / 'checkpoints/steps',
        filename            = '{epoch}-{step}',
        every_n_train_steps = 500,
        save_top_k          = -1  # Set to -1 to save all checkpoints; adjust as needed
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # reduce_lr_on_plateau = ReduceLROnPlateau(
    # monitor   = 'val_loss', # Metric to monitor
    # factor    = 0.1,        # Factor by which the learning rate will be reduced. new_lr = lr * factor
    # patience  = 3,          # Number of epochs with no improvement after which learning rate will be reduced.
    # verbose   = True,       # If True,                                                   prints a message for each update.
    # mode      = 'min',      # In 'min' mode,                                             lr will be reduced when the quantity monitored has stopped decreasing.
    # min_delta = 0.0001,     # Threshold for measuring the new optimum,                   to only focus on significant changes.
    # cooldown  = 5,          # Number of epochs to wait before resuming normal operation after lr has been reduced.
    # min_lr    = 0,          # Lower bound on the learning rate.
    # )

    callbacks += [early_stop_callback, ckp_saver, lr_monitor,]

    return callbacks
def extract_to_batched(batch):
    data = [torch.from_numpy(item['data']).float() for item in batch]
    time = [torch.from_numpy(item['time']) for item in batch]
    # labels = [torch.from_numpy(item['label']).long() for item in batch]
    labels = torch.tensor(np.array([item['label'] for item in batch])).long()
    paths = [item['path'] for item in batch]
    
    data_concatenated = torch.cat(data, dim=0)
    time_concatenated = torch.stack(time, dim=0)
    # labels_tensor = torch.cat(labels,dim=0)
    
    return {
        'data': data_concatenated,
        'time': time_concatenated,
        'label': labels,
        }
def preprocess_collate(batch):
    for sample in batch:
        data = sample['data']  # Shape (472, 15, 12)
        # Efficiently compute features for the entire sample's data
        new_data = compute_features_batch(data)
        
        # Replace the original 12-band data with the new computed 10-feature data
        sample['data'] = new_data
    return extract_to_batched(batch)

def compute_features_batch(data):
    """
    Function to compute the 10 features for the entire data array in one go.
    Input: data is of shape (p, 15, 12), where 12 corresponds to the 12 Sentinel-2 bands
    Output: features is of shape (p, 15, 10), corresponding to the 10 computed features
    """
    # Extract the necessary bands for computation
    B02 = data[:, :, 1]   # Blue
    B03 = data[:, :, 2]   # Green
    B04 = data[:, :, 3]   # Red
    B05 = data[:, :, 4]   # Red Edge 1
    B06 = data[:, :, 5]   # Red Edge 2
    B07 = data[:, :, 6]   # Red Edge 3
    B08 = data[:, :, 7]   # NIR
    B11 = data[:, :, 10]  # SWIR1
    B12 = data[:, :, 11]  # SWIR2

    # Compute the features for the entire data array
    NDVI = (B08 - B04) / (B08 + B04 + 1e-6)  # Add small value to avoid division by zero
    EVI = 2.5 * (B08 - B04) / (B08 + 6 * B04 - 7.5 * B02 + 1 + 1e-6)
    ChlRe = (B08 - B05) / (B08 + B05 + 1e-6)
    REPO = (B05 + B06 + B07) / 3  # Simplified Red Edge Position computation
    Ferrous = B11 / (B12 + 1e-6)  # Add small value to avoid division by zero
    Veg_Moisture = (B08 - B11) / (B08 + B11 + 1e-6)
    NDMI = (B08 - B11) / (B08 + B11 + 1e-6)
    NDWI = (B03 - B08) / (B03 + B08 + 1e-6)
    NDCI = (B05 - B04) / (B05 + B04 + 1e-6)
    MCARI = ((B05 - B04) / (B05 + B04 + 1e-6)) - 0.2 * ((B05 - B03) / (B05 + B03 + 1e-6))

    # Stack the computed features into a single array (472, 15, 10)
    features = np.stack([NDVI, EVI, ChlRe, REPO, Ferrous, Veg_Moisture, NDMI, NDWI, NDCI, MCARI], axis=-1)
    
    return features

def setup_parser():
        # Parse user arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true', default=False, required=False,
                             help='Run in train mode.')
    parser.add_argument('--resume', type=str, default=None, required=False,
                             help='Resume training from the given checkpoint, or the last checkpoint available.')
    parser.add_argument('--model', type=str, default='tsformer', required=False,
                             help='The model to train/test. Default tsformer')

    # parser.add_argument('--weighted_loss', action='store_true', default=True, required=False,
    #                         help='Use a weighted loss function with precalculated weights per class. Default False.')

    parser.add_argument('--prefix', type=str, default=None, required=False,
                             help='The prefix to use for dumping data files. If none, the current timestamp is used')

    parser.add_argument('--load_checkpoint', type=str, required=False,
                             help='The checkpoint path to load for model testing.')

    parser.add_argument('--num_epochs', type=int, default=100, required=False,
                             help='Number of epochs. Default 10')
    parser.add_argument('--batch_size', type=int, default=1, required=False,
                             help='The batch size. Default 1')
    parser.add_argument('--lr', type=float, default=1e-3, required=False,
                             help='Starting learning rate. Default 1e-1')
    # depth, heads, dim_head, mlp_dim
    parser.add_argument('--depth', type=int, default=3, required=False,
                             help='Number of transformer layers. Default 3')
    parser.add_argument('--heads', type=int, default=3, required=False,
                             help='Number of attention heads. Default 3')
    parser.add_argument('--dim_head', type=int, default=64, required=False,
                             help='Dimension of each attention head. Default 64')
    parser.add_argument('--mlp_dim', type=int, default=64, required=False,
                             help='Dimension of the MLP. Default 64')
    # dim = 10
    parser.add_argument('--dim', type=int, default=12, required=False,)
    return parser
if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()

    log_path     = Path('logs')
    loaders_path = log_path / 'loaders'
    log_path.mkdir(exist_ok=True, parents=True)
    loaders_path.mkdir(exist_ok=True, parents=True) 

    prefix        = determine_prefix(args)


    results_path = create_model_log_path(log_path, prefix, args.model, args.train)
    run_path, resume_from_checkpoint, max_epoch, init_epoch = resume_or_start(results_path, args.train, args.num_epochs, args.load_checkpoint)


    train_config = {
                    'linear_encoder'  : LINEAR_ENCODER,
                    'class_weights'   : CLASS_WEIGHT_PIXEL_SHUFFLED_CAP_2000,
                    'run_path'        : run_path,
                    'checkpoint_epoch': init_epoch,
                    'learning_rate'   : args.lr,
                    'method'           : 'one_pixel',
    }


    if args.model == 'cnn':
        model = CNNModel(train_config = train_config)
        if args.load_checkpoint:
            model = model.load_from_checkpoint(args.load_checkpoint, train_config=train_config)
    elif args.model == 'tfcnn':
        model_configs =  {'dim'     : args.dim, # The number of bands
                          'depth'   : args.depth,
                          'heads'   : args.heads,
                          'dim_head': args.dim_head,
                          'mlp_dim' : args.mlp_dim}
        model = TFCNN(**model_configs, train_config = train_config)
        if args.load_checkpoint:
            version = 0
            cfg_path = Path(args.load_checkpoint).parent.parent / f"tensorboard/lightning_logs/version_{version}" / "hparams.yaml"

            # Add a constructor for unknown or unsupported tags
            hparams = read_yaml(cfg_path)
            train_config.update(hparams['train_config'])
            train_config['run_path'] = Path(args.load_checkpoint).parent.parent
            # train_config['run_path'] = '/'.join(train_config['run_path'])
            train_config['checkpoint_epoch'] = init_epoch
            hparams.pop('train_config')
            model_configs.update(hparams)
            model = model.load_from_checkpoint(args.load_checkpoint, **model_configs, train_config=train_config)
    elif args.model == 'tsformer':
        model_configs =  {'dim': 12, 
                          'depth': args.depth, 
                          'heads': args.heads, 
                          'dim_head': args.dim_head, 
                          'mlp_dim': args.mlp_dim}
        model = Medianformer(**model_configs, train_config = train_config)
        if args.load_checkpoint:
            version = 0
            cfg_path = Path(args.load_checkpoint).parent.parent / f"tensorboard/lightning_logs/version_{version}" / "hparams.yaml"

            # Add a constructor for unknown or unsupported tags
            hparams = read_yaml(cfg_path)
            train_config.update(hparams['train_config'])
            train_config['run_path'] = '/'.join(train_config['run_path'])
            hparams.pop('train_config')
            model_configs.update(hparams)
            model = model.load_from_checkpoint(args.load_checkpoint, **model_configs, train_config=train_config)

    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    dm = get_data_module(args, preprocess_collate)
    # dm = get_data_module(args, None)

    if args.train:

        callbacks = get_callbacks(run_path)

        trainer = pl.Trainer(
                              gpus                      = 1,
                              progress_bar_refresh_rate = 10000,
                              min_epochs                = 1,
                              max_epochs                = max_epoch + 1,
                              check_val_every_n_epoch   = 1,
                              val_check_interval        = 1.0,
                              precision                 = 32,
                              callbacks                 = callbacks,
                              logger                    = pl_loggers.TensorBoardLogger(run_path / 'tensorboard'),
                              checkpoint_callback       = True,
                             )

        # Train model
        trainer.fit(model, datamodule=dm)
    else:
        # Load model from checkpoint
        
        print("loaded model from checkpoint")
        trainer = pl.Trainer(
                            gpus                      = 1,
                            # num_nodes                 = args.num_nodes,
                            progress_bar_refresh_rate = 1,
                            min_epochs                = 1,
                            max_epochs                = 2,
                            precision                 = 32,
                             )
        # Test model
        model.eval()

        trainer.test(model, datamodule=dm)
        
        print('Testing done!')