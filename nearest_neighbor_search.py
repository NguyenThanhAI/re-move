import os
import pathlib
import argparse
import json

import pickle

from tqdm import tqdm

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import faiss

import acoss

#from acoss import extractors
#from acoss.extractors import AudioFeatures
from features import AudioFeatures

from models.move_model import MOVEModel
from datasets.full_size_instance_dataset import FullSizeInstanceDataset
from utils.data_utils import handle_device


def enumerate_mp3_files(data_dir: str, audio_format: str):
    file_list = []
    for dirs, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(audio_format):
                file_list.append(os.path.join(dirs, file))

    return file_list


def compute_features(audio_path: str, params: dict, feature: AudioFeatures):
    #feature = AudioFeatures(audio_file=audio_path, sample_rate=["sample_rate"])
    feature.read_audio(audio_path)
    if feature.audio_vector.shape[0] == 0:
        raise IOError("Empty or invalid audio recording file -%s-" % audio_path)

    if params["endtime"]:
        feature.audio_vector = feature.audio_slicer(endTime=params["endtime"])
    if params["downsample_audio"]:
        feature.audio_vector = feature.resample_audio(params["sample_rate"] / params["downsample_factor"])

    output_feature = getattr(feature, "crema")()

    return output_feature


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
    parser.add_argument("--audio_dir",
                        type=str,
                        default=None,
                        help="Directory save mp3 files of index songs")
    parser.add_argument("--feature_dir",
                        type=str,
                        default=None,
                        help="Directory save feature vector of index songs")
    parser.add_argument("--save_dir",
                        type=str,
                        default=None)
    parser.add_argument("--audio_format",
                        type=str,
                        default=None)
    parser.add_argument("--num_nearest",
                        type=int,
                        default=9)
    parser.add_argument("--num_query",
                        type=int,
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

    audio_files = enumerate_mp3_files(data_dir=args.audio_dir, audio_format=args.audio_format)
    audio_files.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])
    print("Number audio files: {}".format(len(audio_files)))

    num_audios = len(audio_files)
    if num_audios > args.num_query:
        audio_files = audio_files[:args.num_query]

    params = {"sample_rate": 44100,
              "input_audio_format": "." + args.audio_format,
              "downsample_audio": False,
              "downsample_factor": 2,
              "endtime": None}

    feature = AudioFeatures(sample_rate=params["sample_rate"])

    crema_feature_list = []
    for audio_file in tqdm(audio_files):
        crema_feature = compute_features(audio_path=audio_file, params=params, feature=feature)
        idxs = np.arange(0, crema_feature.shape[0], 8)
        temp_tensor = torch.from_numpy(crema_feature[idxs].T)
        crema_feature_list.append(torch.cat((temp_tensor, temp_tensor))[:23].unsqueeze(0))

    test_set = FullSizeInstanceDataset(data=crema_feature_list)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    print("Initializing model")

    # initializing the model
    model = MOVEModel(emb_size=args.emb_size)

    # loading a pre-trained model
    model_name = os.path.join(args.main_path, 'saved_models', '{}_models'.format(args.exp_type), 'model_{}.pt'.format(experiment_name))
    model.load_state_dict(torch.load(model_name, map_location='cpu'))

    # sending the model to gpu, if available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    remove_items = []

    with torch.no_grad():  # disabling gradient tracking
        model.eval()  # setting the model to evaluation mode

        # initializing an empty tensor for storing the embeddings
        embed_all = torch.tensor([], device=device)

        # iterating through the data loader
        for batch_idx, item in tqdm(enumerate(test_loader)):
            try:
                # sending the items to the proper device
                item = handle_device(item, device)

                # forward pass of the model
                # obtaining the embeddings of each item in the batch
                emb = model(item)

                # appending the current embedding to the collection of embeddings
                embed_all = torch.cat((embed_all, emb))
            except Exception as e:
                print("Error: {}, input shape: {}, index".format(e, item.shape, batch_idx))
                remove_items.append(audio_files[batch_idx])
                continue
        for re_item in remove_items:
            audio_files.remove(re_item)
            print("Number of audios: {}".format(len(audio_files)))

        embed_all = F.normalize(embed_all, p=2, dim=1).cpu()

        embed_all = embed_all.numpy()

    print("Load index feature")

    index_feature = np.load(os.path.join(args.feature_dir, "remove_features.npy"))

    with open(os.path.join(args.feature_dir, "remove_index_to_image.pkl"), "rb") as f:
        image_with_index_list = pickle.load(f)

    print("Initialize faiss")

    d = index_feature.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(index_feature)
    print(index.is_trained, index.ntotal)

    distances, indices = index.search(embed_all, args.num_nearest)

    audio_to_nearest_neighbors = {}

    for i in range(indices.shape[0]):
        query_audio = audio_files[i]
        print("song: {}, {}, i: {}, index: {}".format(audio_files[i], query_audio, i, indices[i]))
        nearest_neigbors = dict(zip(range(1, indices.shape[1] + 1), map(lambda x: image_with_index_list[x], indices[i])))
        audio_to_nearest_neighbors[query_audio] = nearest_neigbors

    print("audio_to_nearest_neighbors: {}".format(audio_to_nearest_neighbors))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    with open(os.path.join(args.save_dir, "nearest_neighbors.pkl"), "wb") as f:
        pickle.dump(audio_to_nearest_neighbors, f)
