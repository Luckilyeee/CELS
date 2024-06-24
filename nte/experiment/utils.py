import numpy as np
from torch.autograd import Variable
import torch
import cv2
import wandb
import argparse
import io
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from fcn_pytorch_model import FCN as fcn

from nte.data.real.univariate.binary.Coffee import CoffeeDataset
from nte.data.real.univariate.binary.GunPoint import GunPointDataset
from nte.data.real.univariate.binary.ECG200 import ECG200Dataset
from nte.data.real.univariate.binary.TwoLeadECG import TwoLeadECGDataset
from nte.data.real.univariate.multi_class.CBF import CBFDataset
import random
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.colors import ListedColormap

SNS_CMAP = ListedColormap(sns.light_palette('red').as_hex())


def dual_min_max_norm(data, fixed_min=None, fixed_max=None):
    pos_indices = np.argwhere(data > 0)
    pos_features = data[pos_indices]
    neg_indices = np.argwhere(data < 0)
    neg_features = data[neg_indices]

    pos_features_min = np.min(pos_features) if fixed_min is None else fixed_min
    pos_features_max = np.max(pos_features) if fixed_max is None else fixed_max
    pos_features = (pos_features - pos_features_min) / (pos_features_max - pos_features_min)

    neg_features = np.abs(neg_features)
    neg_features_min = np.min(pos_features) if fixed_min is None else fixed_min
    neg_features_max = np.max(pos_features) if fixed_max is None else fixed_max
    neg_features = (neg_features - neg_features_min) / (neg_features_max - neg_features_min)

    data[pos_indices] = pos_features
    data[neg_indices] = -neg_features
    return data


def print_var_stats(var):
    print(f"Min: {var.min()} ({np.argmin(var)}) | Max: {var.max()} ({np.argmax(var)}) | Var: {var.var()}")

def distance_metrics(sample_a, sample_b, def_key=''):
    dist_metrics = {}
    dist_metrics['euc'] = euclidean(sample_a, sample_b)
    dist_metrics['dtw'] = fastdtw(sample_a, sample_b)[0]
    dist_metrics['cs'] = cosine_similarity([sample_a], [sample_b])[0][0]
    return {def_key + k: v for k, v in dist_metrics.items()}

def set_global_seed(seed_value):
    print(f"Setting seed ({seed_value})  . . .")
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    cv2.setRNGSeed(seed_value)
    random.seed(seed_value)


def get_run_configuration(args, dataset, TASK_ID):
    if args.dataset_type == 'train':
        data = dataset.train_data
        label = dataset.train_label
    elif args.dataset_type == 'test':  #.....
        data = dataset.test_data
        label = dataset.test_label
    elif args.dataset_type == 'valid':
        data = dataset.valid_data
        label = dataset.valid_label
    else:
        raise Exception(f"Unknown dataset_type : {args.dataset_type}. Supported - [train, test, representative]")
    print(f"Running on {args.dataset_type} data")

    if args.run_mode == 'single':
        ds = enumerate(zip([data[args.single_sample_id]], [label[args.single_sample_id]]))
        print(f"Running a single sample: idx {args.single_sample_id} . . .")
    elif args.run_mode == 'local':
        ds = enumerate(zip(data, label))
        print(f"Running in local mode on complete data . . .")
    else: #......
        # if args.jobs_per_task > 0:
        #     args.samprun_evaluation_metricsles_per_task = math.ceil(len(data) / args.jobs_per_task)
        print(f"Running in turing mode using slurm tasks jobs{args.jobs_per_task}.samples{args.samples_per_task} . data len{len(data)} TASKID{TASK_ID}..start{int(TASK_ID) * args.samples_per_task}..end{int(TASK_ID) * args.samples_per_task + (args.samples_per_task)}")
        ds = enumerate(
            zip(data[
                int(TASK_ID) * args.samples_per_task: int(TASK_ID) * args.samples_per_task + (args.samples_per_task)],
                label[
                int(TASK_ID) * args.samples_per_task: int(TASK_ID) * args.samples_per_task + (args.samples_per_task)]))
    return ds


def backgroud_data_configuration(BACKGROUND_DATA, BACKGROUND_DATA_PERC, dataset):
    # Ensure the dataset data is loaded
    if dataset.train_data is None or dataset.train_label is None:
        dataset.load_train_data()
    if dataset.test_data is None or dataset.test_label is None:
        dataset.load_test_data()

    # Background Data Configuration
    if BACKGROUND_DATA == 'train':
        print("Using TRAIN data as background data")
        bg_data = dataset.train_data
        bg_label = dataset.train_label
        bg_len = int(len(bg_data) * BACKGROUND_DATA_PERC / 100)
    elif BACKGROUND_DATA == 'test':
        print("Using TEST data as background data")
        bg_data = dataset.test_data
        bg_label = dataset.test_label
        bg_len = int(len(bg_data) * BACKGROUND_DATA_PERC / 100)
    else:
        print("Using Instance as background data (No BG Data)")
        bg_data = dataset.test_data
        bg_label = dataset.test_label
        bg_len = 0

    return bg_data, bg_label, bg_len



number_to_dataset = {
    "1": "GunPoint",
    "2": "Coffee",
    "3": "ECG200",
    "4": "TwoLeadECG",
    "5": "CBF",
}


def dataset_mapper(DATASET):
    # Dataset Mapper
    if DATASET in ['GunPoint', "1"]:
        dataset = GunPointDataset()
    elif DATASET in ['Coffee', "2"]:
        dataset = CoffeeDataset()
    elif DATASET in ['ECG200', "3"]:
        dataset = ECG200Dataset()
    elif DATASET in ["TwoLeadECG", "4"]:
        dataset = TwoLeadECGDataset()
    elif DATASET in ['CBF', "5"]:
        dataset = CBFDataset()
    else:
        raise Exception(f"Unknown Dataset: {DATASET}")
    return dataset


def send_plt_to_wandb(plt, title):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return wandb.Image(Image.open(buf), caption=title)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_image(img):
    if len(img.shape) == 4:
        img = np.transpose(img[0], (1, 2, 0))
        return np.uint8(255 * img)
    else:
        return np.uint8(255 * img)


def tv_norm(signal, tv_beta):
    signal = signal.flatten()
    signal_grad = torch.mean(torch.abs(signal[:-1] - signal[1:]).pow(tv_beta))
    return signal_grad


def preprocess_image(img, use_cuda):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    if use_cuda:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.unsqueeze_(0)
    return Variable(preprocessed_img_tensor, requires_grad=False)


def save_timeseries(mask, time_series, perturbated_output, save_dir, dataset, algo,blurred=None , enable_wandb=False, raw_mask=None, category=None):
    mask = mask
    uplt = plot_cmap(time_series, perturbated_output, mask)

    uplt.xlabel("Timesteps")
    ax=uplt.gca()
    ax.tick_params(tick1On=False)
    ax.tick_params(tick2On=False)
    uplt.xticks([])
    uplt.yticks([])
    ax.yaxis.set_label_coords(0.2,-1)
    ax2 = ax.twinx()
    ax2.tick_params(tick1On=False)
    ax2.tick_params(tick2On=False)
    ax2.yaxis.set_label_coords(0.85,-0.4)
    for pos in ['right', 'top', 'bottom', 'left']:
        uplt.gca().spines[pos].set_visible(True)
        uplt.gca().spines[pos].set_color('k')
    ax.axes.get_xaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.set_yticks([])
    if enable_wandb:
        wandb.log({"Result": [send_plt_to_wandb(uplt, 'Saliency Visualization')]})
    uplt.savefig("/tmp/kdd.png", dpi=1200)
    uplt.savefig("/tmp/kdd.pdf", dpi=1200)


def save(mask, img, blurred, save_dir, enable_wandb):
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))

    mask = (mask - np.min(mask)) / np.max(mask)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = 1.0 * heatmap + np.float32(img) / 255
    cam = cam / np.max(cam)

    img = np.float32(img) / 255
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)

    cv2.imwrite(f"{save_dir}/res-perturbated.png", np.uint8(255 * perturbated))

    if enable_wandb:
        wandb.log({"Result": [wandb.Image(np.uint8(255 * perturbated), caption="Perurbation"),
                              wandb.Image(np.uint8(255 * heatmap), caption="Heatmap"),
                              wandb.Image(np.uint8(255 * mask), caption='Mask'),
                              wandb.Image(np.uint8(255 * cam), caption='CAM')]})
    cv2.imwrite(f"{save_dir}/res-heatmap.png", np.uint8(255 * heatmap))
    cv2.imwrite(f"{save_dir}/res-mask.png", np.uint8(255 * mask))
    cv2.imwrite(f"{save_dir}/res-cam.png", np.uint8(255 * cam))


def numpy_to_torch(img, use_cuda, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        output = output.cuda()

    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v

def get_model(dataset, input_size = 1, num_classes = 2):

    path = "models/" + dataset + "_best_model.pth"
    model = fcn(input_size, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    return model

def plot_cmap(data, pertuebated_data, saliency):

    data = data
    pertuebated_data = pertuebated_data.cpu().detach().numpy().flatten().tolist()

    plt.clf()
    fig = plt.gcf()
    im = plt.imshow(saliency.reshape([1, -1]), cmap=SNS_CMAP, aspect="auto", alpha=0.8,
                        extent=[0, len(saliency) - 1, float(np.min([np.min(data), np.min(pertuebated_data),  np.min(saliency)])) - 1e-1,
                                float(np.max([np.max(data), np.max(pertuebated_data), np.max(saliency)])) + 1e-1]
                        )
    plt.plot(data, lw=1, label = "original", color = 'blue')
    plt.plot(pertuebated_data, lw=1, label = 'Perturbated', color = 'green')
    plt.grid(False)
    plt.xlabel("Timesteps")
    plt.ylabel("Values")
    plt.legend()
    cax = fig.add_axes([0.27, 0.05, 0.5, 0.05])
    fig.colorbar(im, cax=cax, orientation="horizontal")
    plt.tight_layout(pad=4)
    return plt
