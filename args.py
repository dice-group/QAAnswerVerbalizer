import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description='QAAnswerVerbalizer: Generating Natural Language Answers for KBQA Systems')

    parser.add_argument('--dataset', default='vquanda',
                        type=str, help='choice between vquanda, quald')
    parser.add_argument('--num', default=3999, type=int)
    parser.add_argument('--name', default='test', type=str,
                        help='choice between train and test data')
    parser.add_argument('--lang', default='de', type=str,
                        help='Language choice only available for Quald-9 plus. en for English, de for German')

    parser.add_argument('--model_name', default='pegasus', type=str)
    parser.add_argument(
        '--checkpoint_path', default='./output/checkpoint-800', type=str, help='Checkpoint path')
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
    return parser
