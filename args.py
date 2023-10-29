import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description='QAAnswerVerbalizer: Generating Natural Language Answers for KBQA Systems')

    parser.add_argument('--dataset', default='vquanda',
                        type=str, help='choice between vquanda, qald, grailQA, paraQA')
    parser.add_argument('--num', default=100, type=int)
    parser.add_argument('--name', default='train', type=str,
                        help='choice between train and test data')
    parser.add_argument('--lang', default='en', type=str,
                        help='Language choice only available for Quald-9 plus. en for English, de for German')
    parser.add_argument('--mask_ans', default=True, type=bool,
                        help='Choice to mask answer with answer token')
    parser.add_argument('--ans_limit', default=5, type=bool,
                        help='Choice to mask answer with answer token')

    parser.add_argument('--model_name', default='pegasus', type=str)
    parser.add_argument(
        '--checkpoint_path', default='./output/quald/pegasus/checkpoint-5000', type=str, help='Checkpoint path')
    parser.add_argument('--model_path', default='google/pegasus-xsum',
                        type=str, help='Path of the model')
    parser.add_argument('--tokenizer_path', default='google/pegasus-xsum',
                        type=str, help='Path of the tokenizer')
    parser.add_argument('--train_epochs', default=3,
                        type=int, help='Number of training epochs')
    parser.add_argument('--save_steps', default=1000, type=int,
                        help='Number of steps after which checkpoint will be created')
    parser.add_argument('--eval_steps', default=1000, type=int,
                        help='Number of steps for evaluation')
    parser.add_argument('--mode', default='triples', type=str,
                        help='If Triples or Query is to be used.')
    parser.add_argument('--save_strategy', default='steps', type=str)
    parser.add_argument('--eval_strategy', default='steps', type=str)
    parser.add_argument('--device_train_batch_size', default=32,
                        type=int, help='per device training batch size')
    parser.add_argument('--device_eval_batch_size', default=32,
                        type=int, help='per device evaluation batch size')
    parser.add_argument('--warmup_steps', default=100,
                        type=int, help='warmup steps')
    parser.add_argument('--weight_decay', default=0.01,
                        type=float, help='weight decay')
    parser.add_argument('--load_best_model_at_end', default=True,
                        type=bool, help='Choice to load best model at the end')
    parser.add_argument('--save_limit', default=5, type=int,
                        help='The limit of total checkpoints to be saved')

    return parser
