__all__ = ['parse_arguments']

import argparse
from nte.experiment.utils import str2bool


def parse_arguments(standalone=False):

    parser = argparse.ArgumentParser(description='NTE Pipeline')

    # General Configuration
    parser.add_argument('--pname', type=str, help='Project name - [project_name]', default="tsmule_mc")
    parser.add_argument('--task_id', type=int, help='Task ID', default=0)
    parser.add_argument('--run_id', type=str, help='Run ID', default=0)
    parser.add_argument('--save_perturbations', type=str2bool, nargs='?', const=False, help='Save Perturbations',
                        default=False)
    parser.add_argument('--conf_thres', type=float, help="Confidence threshold of prediction", default=0.0)

    # Run Configuration
    parser.add_argument('--run_mode', type=str, help='Run Mode - ["single", "local", "turing"]',default='single', choices=["single", "local", "turing"])
    parser.add_argument('--dataset_type', type=str, help='Run Mode - ["train", "test", "valid"]',default='test', choices=["train", "test", "valid"])
    parser.add_argument('--samples_per_task', type=int, help='Number of samples to run per task in turing mode',default=1)
    parser.add_argument('--jobs_per_task', type=int, help='Max number of jobs to run all samples',default=1)
    #KDD Case study do not change Meat-mc default sample  = 1, 37 and  39 for red or 19 for LMUX CASE study
    parser.add_argument('--single_sample_id', type=int, help='Single Sample',default=0)

    # Seed Configuration
    parser.add_argument('--enable_seed', type=str2bool, nargs='?', const=True, help='Enable Seed',default=True)
    parser.add_argument('--enable_seed_per_instance', type=str2bool, nargs='?', const=True,help='Enable Seed Per Instance',default=False)
    parser.add_argument('--seed_value', type=int, help='Seed Value',default=1)

    # Mask Normalization
    parser.add_argument('--mask_norm', type=str, help='Mask Normalization - ["clamp", "softmax", "sigmoid"]',default='clamp',choices=["clamp", "softmax", "sigmoid", "none"])

    # Algorithm
    parser.add_argument('--algo', type=str, help='Algorithm type required',
                        default='cf',
                        choices=["cf"])

    # Dataset and Background Dataset configuration
    parser.add_argument('--dataset', type=str, default='Coffee',
                        help='Dataset name required',
                        choices=["GunPoint", "Coffee", "ECG200", "TwoLeadECG",  "CBF",
                                 "1", "2", "3", "4", "5"])
    parser.add_argument('--background_data', type=str, help='[train|test|none]', default='test',
                        choices=["train", "test", "none"])
    parser.add_argument('--background_data_perc', type=float, help='%% of Background Dataset', default=100)

    # Black-box model configuration
    # parser.add_argument('--bbmg', type=str, help='Grouped Time steps [yes|no]', default='no',choices=["yes", "no"])
    parser.add_argument('--bbm', type=str, help='Black box model type - [dnn|rnn|cnn]', default='dnn',choices=["dnn", "rnn", "cnn"])
    parser.add_argument('--bbm_path', type=str, help='Black box model path - [dnn|rnn]', default="default")

    # Gradient Based Algo configurations
    parser.add_argument('--enable_blur', type=str2bool, nargs='?', const=True, help='Enable blur', default=False)
    parser.add_argument('--enable_tvnorm', type=str2bool, nargs='?', const=True, help='Enable TV Norm', default=True)
    parser.add_argument('--enable_budget', type=str2bool, nargs='?', const=True, help='Enable budget', default=True)
    parser.add_argument('--enable_noise', type=str2bool, nargs='?', const=True, help='Enable Noise', default=False)
    parser.add_argument('--enable_dist', type=str2bool, nargs='?', const=True, help='Enable Dist Loss', default=False)

    parser.add_argument('--enable_lr_decay', type=str2bool, nargs='?', const=True, help='LR Decay', default=False)

    parser.add_argument('--dist_loss', type=str, help='Distance Loss Type - ["euc", "dtw", "w_euc", "w_dtw"]',
                        default='no_dist',
                        # default='dtw',
                        choices=["euc", "dtw", "w_euc", "w_dtw", "n_dtw", "n_w_dtw", "no_dist"])

    parser.add_argument('--early_stop_criteria_perc', type=float, help='Early Stop Criteria Percentage',default=0.80)

    # Evaluation Metric Configuration
    parser.add_argument('--run_eval', type=str2bool, nargs='?', const=True, help='Run Evaluation Metrics',
                        default=True)
    parser.add_argument('--run_eval_every_epoch', type=str2bool, nargs='?', const=True,
                        help='Run Evaluation Metrics for every epoch',
                        default=False)

    # Hyper Param Configuration
    parser.add_argument('--lr', type=float, help='Learning Rate', default=0.1) #0.01
    parser.add_argument('--lr_decay', type=float, help='LR Decay', default=0.999)

    parser.add_argument('--l_budget_coeff', type=float, help='L Budget Coefficient', default=0.6)  # 0.05
    parser.add_argument('--l_tv_norm_coeff', type=float, help='L TV Norm Coefficient', default=0.5)
    parser.add_argument('--tv_beta', type=float, help='TV Norm Beta', default=3)
    parser.add_argument('--l_max_coeff', type=float, help='L Minimize Coefficient', default=0.7)
    parser.add_argument('--noise_mean', type=float, help='Noise Mean', default=0)
    parser.add_argument('--noise_std', type=float, help='Noise Std', default=0.1)

    # mask repo config
    parser.add_argument('--enable_mask_repo', type=str2bool, nargs='?', const=True, help='Enable Mask Repo', default=True)
    parser.add_argument('--mask_repo_type', type=str, help='mask repo type', default="last_n_cond",
                        choices=["last_n_cond", "last_n", "min_l_total", "min_l_prev", "max_conf"])
    parser.add_argument('--mask_repo_rec', type=int, help='Fetch N masks', default=10)


    parser.add_argument('--dist_coeff', type=float, help='Dist Loss Coeff', default=0)
    parser.add_argument('--w_decay', type=float, help='Weight Decay', default=0.0)

    parser.add_argument('--bwin', type=int, help='bwin ', default=1)
    parser.add_argument('--run', type=str, help='Run ', default=501 )
    parser.add_argument('--bsigma', type=float, help='bsigma ', default=0.9)
    parser.add_argument('--sample', type=float, help='sample', default=6.25)

    parser.add_argument('--tsne', type=str,help='TSNE plots',default='no', choices=["yes", "no"])
    # Early Stopping Criteria
    parser.add_argument('--early_stopping', type=str2bool, nargs='?', const=False, help='Enable or Disable Early Stop',default=False)
    parser.add_argument('--early_stop_min_epochs', type=float, help='Early Stop Minimum Epochs',default=0)
    parser.add_argument('--max_itr', type=int, help='Maximum Iterations', default=5000)
    parser.add_argument('--early_stop_prob_range', type=float, help='Early Stop probability range',default=0.10)
    parser.add_argument('--early_stop_diff', type=float, help='Early Stop Difference',default=0)
    parser.add_argument('--prob_upto', type=float,help='Probability upto',default=0.00)
    parser.add_argument('--sample_masks', type=int,help='Number of Sample masks ',default=1)
    parser.add_argument('--window_size', type=int,help='Which window size to use ',default=5)
    parser.add_argument('--class_prob', type=float,help='class_prob',default=0.0)
    parser.add_argument('--ce', type=str,help='CrossEntropy', default='no', choices=["yes", "no"])


    if standalone:
        return parser.parse_known_args()
    else:
        return parser.parse_args()
