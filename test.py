from constants import *
from utils import *
import logging

root_path = get_project_root()
logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(
    """{path}/results/{dataset}_test.log""".format(path=root_path, dataset=args.dataset)), logging.StreamHandler()])
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    filepath = """{path}/data/{dataset}/preprocessed_{dataset}_{name}.json""".format(
        path=root_path, dataset=args.dataset, name=args.name)
    data = make_df(path=str(filepath))
    logging.info(
        """Length of Test Dataset: {length}""".format(length=len(data)))

    path = args.checkpoint_path
    torch_device = TORCH_DEVICE
    logger.info("""--Loading Model: {model}--""".format(model=args.model_name))
    model = set_model(model_name=args.model_name,
                      path=path, device=torch_device)
    logger.info("""--Loading Tokenizer: {model}--""".format(model=args.name))
    tokenizer = set_tokenizer(model_name=args.model_name, path=path)

    scorer = Score()
    scorer.data_scorer(
        data, model=model, tokenizer=tokenizer, torch_device=torch_device)
    scorer.save_to_file()

    logging.info("""Meteor Score Average: {score}""".format(
        score=scorer.meteor_avg.avg))
    logging.info("""Bleu Score Average: {score}""".format(
        score=scorer.bleu_avg.avg))
    logging.info("""Sacre Bleu Score Average: {score}""".format(
        score=scorer.sacrebleu_avg.avg))

