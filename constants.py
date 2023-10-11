from pathlib import Path
import args as args
import torch


def get_project_root() -> Path:
    return Path(__file__).parent


parser = args.get_parser()
args = parser.parse_args()

TORCH_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ANS_TOKEN = '[ANS]'
SEP_TOKEN = '<SEP>'
