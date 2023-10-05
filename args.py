import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description="QAAnswerVerbalizer: Generating Natural Language Answers for KBQA Systems")

    parser.add_argument('--dataset', default='vquanda',
                        type=str, help="choice between vquanda, quald")
    parser.add_argument('--num', default=3999, type=int)
    parser.add_argument('--name', default='train', type=str,
                        help="choice between train and test data")
    parser.add_argument('--lang', default='de', type=str,
                        help="Language choice only available for Quald-9 plus. en' for English, 'de' for German ")

    parser.add_argument('--model_name', default='pegasus', type=str)

    return parser
