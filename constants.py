from pathlib import Path
import args as args

def get_project_root() -> Path:
    return Path(__file__).parent


parser = args.get_parser()
args = parser.parse_args()