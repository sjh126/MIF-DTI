import argparse
import torch
from RunModel import run_MIF_model,ensemble_run_MIF_model
from model import MIFDTI

parser = argparse.ArgumentParser(
    prog='MIF-DTI',
    description='MIF-DTI is model in paper: \"multimodal information fusion method for drug-target interaction prediction\"',
    epilog='Model config set by config.py')

parser.add_argument('dataSetName', choices=[
                    "DrugBank", "Davis", "BIOSNAP"], help='Enter which dataset to use for the experiment')
parser.add_argument('-m', '--model', choices=['MIF-DTI', 'MIF-DTI-B'],
                    default='MIF-DTI', help='Which model to use, \"MIF-DTI\" is used by default')
parser.add_argument('-s', '--seed', type=int, default=114514,
                    help='Set the random seed, the default is 114514')
parser.add_argument('-f', '--fold', type=int, default=5,
                    help='Set the K-Fold number, the default is 5')
parser.add_argument('-g', '--gpu', type=int, default=0,
                    help='cuda number, the default is 0')

args = parser.parse_args()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

if args.model == 'MIF-DTI':
    run_MIF_model(SEED=args.seed, DATASET=args.dataSetName,
              MODEL=MIFDTI, K_Fold=args.fold, LOSS='PolyLoss', device=device)
if args.model == 'MIF-DTI-B':
    ensemble_run_MIF_model(SEED=args.seed, DATASET=args.dataSetName, K_Fold=args.fold, device=device)