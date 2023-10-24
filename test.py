from constants import *
from utils import *
import logging

root_path = get_project_root()
#  Add logger and formatting
logger = logging.getLogger()
logger.setLevel(logging.INFO)

con_handler = logging.StreamHandler()
file_handler = logging.FileHandler("""{path}/results/{dataset}_{model}_test.log""".format(
    path=root_path, dataset=args.dataset, model=args.model_name))
con_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)
con_handler.setFormatter(logging.Formatter(
    "%(asctime)-7s %(name)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S"))
file_handler.setFormatter(logging.Formatter(
    "%(asctime)-7s %(name)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S"))
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.addHandler(con_handler)


if __name__ == '__main__':
    filepath = """{path}/data/{dataset}/preprocessed_{dataset}_{name}.json""".format(
        path=root_path, dataset=args.dataset, name=args.name)
    data = make_df(path=str(filepath))
    logger.info(
        """Length of Test Dataset: {length}""".format(length=len(data)))

    path = args.checkpoint_path
    torch_device = TORCH_DEVICE
    logger.info(
        """-- Loading {name} dataset {dataset}""".format(name=args.name, dataset=args.dataset))
    logger.info("""-- Loading Model: {model} from {path} --""".format(
        model=args.model_name, path=args.checkpoint_path))
    model = set_model(model_name=args.model_name,
                      path=path, device=torch_device)
    logger.info(
        """-- Loading Tokenizer: {model} --""".format(model=args.model_name))
    tokenizer = set_tokenizer(model_name=args.model_name, path=path)

    # Calculate score
    scorer = Score()
    scorer.data_scorer(
        data, model=model, tokenizer=tokenizer, torch_device=torch_device)
    scorer.save_to_file()

    # Get average score of all metrics
    logger.info("""Meteor Score Average: {score}""".format(
        score=scorer.meteor_avg.avg))
    logger.info("""Bleu Score Average: {score}""".format(
        score=scorer.bleu_avg.avg))
    logger.info("""Sacre Bleu Score Average: {score}""".format(
        score=scorer.sacrebleu_avg.avg))
    logger.info("""Rouge Score Average: {score}""".format(
        score=scorer.rouge_avg.avg))
    logger.info("""Rouge L Score Average: {score}""".format(
        score=scorer.rouge_L_avg.avg))
