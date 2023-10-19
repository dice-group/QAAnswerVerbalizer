import sys
sys.path.append("./")
from utils import *
from constants import get_project_root
from constants import args
from constants import *
from QAAVData import prepare_data, train_val_split
import transformers


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

    training_args = transformers.TrainingArguments(
        output_dir="""{path}/output/{dataset}/{model}""".format(
            path=root_path, dataset=args.dataset, model=args.model_name),
        num_train_epochs=args.train_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        save_steps=args.save_steps,
        save_strategy='steps',
        save_total_limit=4,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="""{path}/output/{dataset}/{model}/logs""".format(
            path=root_path, dataset=args.dataset, model=args.model_name),
        logging_steps=10,
        load_best_model_at_end=True,
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tr_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    )
    trainer.train()
