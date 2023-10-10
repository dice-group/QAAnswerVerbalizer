from constants import *
import json
import pandas as pd
from tqdm import tqdm
import evaluate
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

root_path = get_project_root()


def make_df(path=str):
    with open(path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame({"questions": [], "queries": [], 'labels': []}, index=[])
    for i in range(len(data)):
        d = data[str(i)]
        # add Answer token and Separator token along with query
        ans_query = ANS_TOKEN + ' ' + SEP_TOKEN + ' ' + d['query']
        df.loc[d['index']] = [d['question'], ans_query, d['verbalization']]

    return df


def set_model(model_name, path, device):
    if model_name == 'pegasus':
        model = PegasusForConditionalGeneration.from_pretrained(
            path).to(device)

    if model_name == 't5':
        model = T5ForConditionalGeneration.from_pretrained(path).to(device)

    return model


def set_tokenizer(model_name, path):
    if model_name == 'pegasus':
        tokenizer = PegasusTokenizer.from_pretrained(path)

    if model_name == 't5':
        tokenizer = T5Tokenizer.from_pretrained(path)

    return tokenizer


def predict(model, tokenizer, question, query, torch_device):
    input = tokenizer(question, query, return_tensors="pt").to(torch_device)
    outputs = model.generate(
        input["input_ids"], max_length=100).to(torch_device)
    verbalization = tokenizer.batch_decode(
        outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    return verbalization


class AverageScore():
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def calc_avg(self, score):
        self.sum += score
        self.count += 1
        self.avg = self.sum / self.count


class Score(object):
    def __init__(self):
        self.results = []
        self.bleu = evaluate.load('bleu')
        self.meteor = evaluate.load('meteor')
        self.sacre_bleu = evaluate.load('sacrebleu')
        self.bleu_avg = AverageScore()
        self.sacrebleu_avg = AverageScore()
        self.meteor_avg = AverageScore()

    def _bleu_score(self, pred, ref):
        return self.bleu.compute(predictions=[pred], references=[ref])

    def _meteor_score(self, pred, ref):
        return self.meteor.compute(predictions=[pred], references=[ref])

    def _sacrebleu(self, pred, ref):
        return self.sacre_bleu.compute(predictions=[pred], references=[ref])

    def data_scorer(self, data, model, tokenizer, torch_device):
        for i in tqdm(range(len(data[:10]))):
            pred = predict(model, tokenizer,
                           data.iloc[i, 0], data.iloc[i, 1], torch_device)
            ref = data.iloc[i, 2]

            bleu_score = self._bleu_score(pred, ref)
            meteor_score = self._meteor_score(pred, ref)
            sacrebleu_score = self._sacrebleu(pred, ref)

            self.bleu_avg.calc_avg(bleu_score['bleu'])
            self.meteor_avg.calc_avg(meteor_score['meteor'])
            self.sacrebleu_avg.calc_avg(sacrebleu_score['score'])

            self.results.append({'hyp': pred, 'reference': ref, 'bleu': bleu_score,
                                'meteor': meteor_score['meteor'], 'sacre_bleu': sacrebleu_score})

            print(bleu_score, meteor_score, sacrebleu_score)
        return self.results

    def save_to_file(self):
        result = json.dumps(self.results)
        with open("""{path}/output/{dataset}_result.json""".format(path=root_path, dataset=args.dataset), 'w', encoding='utf-8') as f:
            f.write(result)
