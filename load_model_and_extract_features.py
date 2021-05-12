import os
import pathlib
import argparse
import json

import pickle

from tqdm import tqdm

import numpy as np

import deepdish as dd

import torch
from torch.utils.data import DataLoader

import torch.nn.functional as F

from models.move_model import MOVEModel
from datasets.full_size_instance_dataset import FullSizeInstanceDataset
from utils.data_utils import handle_device


def enumerate_h5_files(data_dir: str):
    file_list = []
    for dirs, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".h5"):
                file_list.append(os.path.join(dirs, file))

    return file_list


def evaluate(exp_name,
             exp_type,
             main_path,
             emb_size,
             loss,
             data_dir):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print('Evaluating model {}.'.format(exp_name))

    file_list = enumerate_h5_files(data_dir)
    file_list.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])
    print("Number feature files: {}".format(len(file_list)))

    data = []
    name = list(map(lambda x: os.path.splitext(os.path.basename(x))[0], file_list))
    print("name: {}".format(name))
    image_with_index_list = dict(zip(name, range(len(name))))
    print("image_with_index_list: {}".format(image_with_index_list))

    for file in tqdm(file_list):
        temp_crema = dd.io.load(file)["crema"]
        #print("crema shape: {}".format(temp_crema.shape))
        idxs = np.arange(0, temp_crema.shape[0], 8)

        temp_tensor = torch.from_numpy(temp_crema[idxs].T)

        data.append(torch.cat((temp_tensor, temp_tensor))[:23].unsqueeze(0))
        #name.append(os.path.splitext(os.path.basename(file))[0])

    test_set = FullSizeInstanceDataset(data=data)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    print("Initializing model")

    # initializing the model
    model = MOVEModel(emb_size=emb_size)

    # loading a pre-trained model
    model_name = os.path.join(main_path, 'saved_models', '{}_models'.format(exp_type), 'model_{}.pt'.format(exp_name))
    model.load_state_dict(torch.load(model_name, map_location='cpu'))

    # sending the model to gpu, if available
    model.to(device)

    with torch.no_grad():  # disabling gradient tracking
        model.eval()  # setting the model to evaluation mode

        # initializing an empty tensor for storing the embeddings
        embed_all = torch.tensor([], device=device)

        # iterating through the data loader
        for batch_idx, item in tqdm(enumerate(test_loader)):
            # sending the items to the proper device
            item = handle_device(item, device)

            # forward pass of the model
            # obtaining the embeddings of each item in the batch
            emb = model(item)

            # appending the current embedding to the collection of embeddings
            embed_all = torch.cat((embed_all, emb))

        embed_all = F.normalize(embed_all, p=2, dim=1)

    return embed_all.cpu(), image_with_index_list


if __name__ == '__main__':

    path = pathlib.Path(__file__).parent.absolute()

    parser = argparse.ArgumentParser(description='Training and evaluation code for Re-MOVE experiments.')
    parser.add_argument('-rt',
                        '--run_type',
                        type=str,
                        default='train',
                        choices=['train', 'test'],
                        help='Whether to run the training or the evaluation script.')
    parser.add_argument('-mp',
                        '--main_path',
                        type=str,
                        default='{}'.format(path),
                        help='Path to the working directory.')
    parser.add_argument('--exp_type',
                        type=str,
                        default='lsr',
                        choices=['lsr', 'kd', 'pruning', 'baseline'],
                        help='Type of experiment to run.')
    parser.add_argument('-pri',
                        '--pruning_iterations',
                        type=int,
                        default=None,
                        help='Number of iterations for pruning.')
    parser.add_argument('-tf',
                        '--train_file',
                        type=str,
                        default=None,
                        help='Path for training file. If more than one file are used, '
                             'write only the common part.')
    parser.add_argument('-ch',
                        '--chunks',
                        type=int,
                        default=None,
                        help='Number of chunks for training set.')
    parser.add_argument('-vf',
                        '--val_file',
                        type=str,
                        default=None,
                        help='Path for validation data.')
    parser.add_argument('-noe',
                        '--num_of_epochs',
                        type=int,
                        default=None,
                        help='Number of epochs for training.')
    parser.add_argument('-emb',
                        '--emb_size',
                        type=int,
                        default=None,
                        help='Size of the final embeddings.')
    parser.add_argument('-bs',
                        '--batch_size',
                        type=int,
                        default=None,
                        help='Batch size for training iterations.')
    parser.add_argument('-l',
                        '--loss',
                        type=int,
                        default=None,
                        choices=[0, 1, 2, 3],
                        help='Which loss to use for training. 0 for Triplet, '
                             '1 for ProxyNCA, 2 for NormalizedSoftmax, and 3 for Group loss.')
    parser.add_argument('-kdl',
                        '--kd_loss',
                        type=str,
                        default=None,
                        choices=['distance', 'cluster'],
                        help='Which loss to use for Knowledge Distillation training.')
    parser.add_argument('-ms',
                        '--mining_strategy',
                        type=int,
                        default=None,
                        choices=[0, 1, 2, 3],
                        help='Mining strategy for Triplet loss. 0 for random, 1 for semi-hard, '
                             '2 for hard, 3 for semi-hard with using all positives.')
    parser.add_argument('-ma',
                        '--margin',
                        type=float,
                        default=None,
                        help='Margin for Triplet loss.')
    parser.add_argument('-o',
                        '--optimizer',
                        type=int,
                        default=None,
                        choices=[0, 1],
                        help='Optimizer for training. 0 for SGD, 1 for Ranger.')
    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        default=None,
                        help='Base learning rate for the optimizer.')
    parser.add_argument('-flr',
                        '--finetune_learning_rate',
                        type=float,
                        default=None,
                        help='Learning rate for finetuning the feature extractor for LSR training.')
    parser.add_argument('-mo',
                        '--momentum',
                        type=float,
                        default=None,
                        help='Value for momentum parameter for SGD.')
    parser.add_argument('-lrs',
                        '--lr_schedule',
                        type=int,
                        nargs='+',
                        default=None,
                        help='Epochs for reducing the learning rate. Multiple arguments are appended in a list.')
    parser.add_argument('-lrg',
                        '--lr_gamma',
                        type=float,
                        default=None,
                        help='Step size for learning rate scheduler.')
    parser.add_argument('-pl',
                        '--patch_len',
                        type=int,
                        default=None,
                        help='Number of frames for each input in time dimension (only for training).')
    parser.add_argument('-da',
                        '--data_aug',
                        type=int,
                        default=None,
                        choices=[0, 1],
                        help='Whether to use data augmentation while training.')
    parser.add_argument('-nw',
                        '--num_workers',
                        default=None,
                        type=int,
                        help='Num of workers for the data loader.')
    parser.add_argument('-ofb',
                        '--overfit_batch',
                        type=int,
                        default=None,
                        help='Whether to overfit a single batch. It may help with revealing problems with training.')
    parser.add_argument("--data_dir",
                        type=str,
                        default=None)
    parser.add_argument("--save_dir",
                        type=str,
                        default=None)

    args = parser.parse_args()

    experiment_name = 're-move_{}'.format(args.exp_type)

    with open(os.path.join(path, 'data/{}_defaults.json'.format(args.exp_type))) as f:
        cfg = json.load(f)

    for key in args.__dict__.keys():
        if getattr(args, key) is None:
            setattr(args, key, cfg[key])

    for key in cfg.keys():
        if key == 'abbr':
            pass
        else:
            if cfg[key] != getattr(args, key):
                val = '{}'.format(getattr(args, key))
                val = val.replace('.', '-')
                experiment_name = '{}_{}_{}'.format(experiment_name, cfg['abbr'][key], val)

    cfg = args.__dict__

    embed_all, image_with_index_list = evaluate(experiment_name, cfg['exp_type'], cfg['main_path'], cfg['emb_size'], cfg['loss'], cfg["data_dir"])

    embed_all = embed_all.numpy()
    print("Embedding shape: {}".format(embed_all.shape))

    image_with_index_list = {v: k for k, v in image_with_index_list.items()}

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    np.save(os.path.join(args.save_dir, "remove_features.npy"), embed_all)

    with open(os.path.join(args.save_dir, "remove_index_to_image.pkl"), "wb") as f:
        pickle.dump(image_with_index_list, f)
