import sys
sys.path.append("./")
from utils import *
from constants import get_project_root
from constants import args
from constants import *
from QAAVData import prepare_data, train_val_split
from transformers import Trainer, TrainingArguments


class FineTuningTrainer:
    def __init__(self, root_path, model, train_data, eval_data, tokenizer, args):
        self.root_path = root_path
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.tokenizer = tokenizer
        self.args = args
        self.training_args = self.get_training_args()
        self.trainer = self.get_trainer()

    def get_training_args(self):
        output_directory = """{path}/output/{dataset}/{model}""".format(
            path=root_path, dataset=args.dataset, model=args.model_name)
        return TrainingArguments(
            num_train_epochs=args.train_epochs,
            output_dir=output_directory,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            save_steps=args.save_steps,
            save_strategy='steps',
            save_total_limit=4,
            evaluation_strategy='steps',
            eval_steps=args.eval_steps,
            warmup_steps=100,
            weight_decay=0.01,
            load_best_model_at_end=True,
        )

    def get_trainer(self):
        return Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_data,
            eval_dataset=self.eval_data,
            tokenizer=self.tokenizer,)

    def train(self):
        self.trainer.train()


if __name__ == '__main__':
    root_path = get_project_root()
    filepath = """{path}/data/{dataset}/preprocessed_{dataset}_{name}.json""".format(
        path=root_path, dataset=args.dataset, name=args.name)

    data = make_df(path=str(filepath))
    train_data, val_data = train_val_split(data)

    model = set_model(model_name=args.model_name,
                      path=args.model_path, device=TORCH_DEVICE)
    tokenizer = set_tokenizer(
        model_name=args.model_name, path=args.tokenizer_path)

    tr_data, val_data = prepare_data(tokenizer, train_data, val_data)
    tr = FineTuningTrainer(root_path=root_path, model=model, train_data=tr_data,
                           eval_data=val_data, tokenizer=tokenizer, args=args)
    tr.train()
