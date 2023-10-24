from constants import *
import json
import pandas as pd
from tqdm import tqdm
import evaluate
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer

root_path = get_project_root()


def make_df(path=str):
    with open(path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame({"questions": [], "queries": [], 'labels': []}, index=[])
    df = df.astype('object')

    for _, value in data.items():
        index = value['index']
        question = value['question']
        verbalization = value['verbalization']
        answers = value['answers']
        query = value['query']
        if args.dataset == 'paraQA' and args.name == 'train':
            # In-case of ParaQA. only use the fist verbalization
            verbalization = value['verbalization'][0]

        if args.dataset == 'grailQA':
            if args.mode == 'triples':
                triple = query['graph']
                triple = get_triples_string(triple)
                query = triple
            else:
                query = query['sparql']
        if args.mask_ans:
            ans = ANS_TOKEN
        else:
            ans = answers

        # add Answer token or Answer and Separator token along with query
        ans_query = ans + ' ' + SEP_TOKEN + ' ' + query
        df.loc[index] = [question, ans_query, verbalization]

    return df


def set_model(model_name, path, device):
    if model_name == 'pegasus':
        model = PegasusForConditionalGeneration.from_pretrained(
            path).to(device)

    if model_name == 't5':
        model = T5ForConditionalGeneration.from_pretrained(path).to(device)

    if model_name == 'bart':
        model = BartForConditionalGeneration.from_pretrained(path).to(device)

    return model


def set_tokenizer(model_name, path):
    if model_name == 'pegasus':
        tokenizer = PegasusTokenizer.from_pretrained(path)

    if model_name == 't5':
        tokenizer = T5Tokenizer.from_pretrained(path)

    if model_name == 'bart':
        tokenizer = BartTokenizer.from_pretrained(path)

    return tokenizer


def predict(model, tokenizer, question, query, torch_device):
    input = tokenizer(question, query, return_tensors="pt").to(torch_device)
    outputs = model.generate(
        input["input_ids"], max_length=100).to(torch_device)
    verbalization = tokenizer.batch_decode(
        outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    return verbalization


def get_triples_string(triples):
    """Join triples in a string"""
    for t in triples:
        triple_string = ""
        all_triples = ""
        for i in t:
            triple_string += ' ' + i['name']
        if all_triples:
            # Add SEP_TOKEN between multiple triples
            all_triples += SEP_TOKEN + triple_string
        else:
            all_triples += triple_string
    return all_triples


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
    """
    Calculates scores and average on different metrics after prediction. 
    The Bleu, Meteor and Rouge are normalized. 
    --------
    Metrics
    --------
    1. Bleu
    2. Meteor
    3. Sacre Bleu
    4. Rouge

    -------------------------
    returns: dict

    """
    def __init__(self):
        self.results = []
        self.bleu = evaluate.load('bleu')
        self.meteor = evaluate.load('meteor')
        self.sacre_bleu = evaluate.load('sacrebleu')
        self.rouge = evaluate.load('rouge')
        self.bleu_avg = AverageScore()
        self.sacrebleu_avg = AverageScore()
        self.meteor_avg = AverageScore()
        self.rouge_avg = AverageScore()
        self.rouge_L_avg = AverageScore()

    def _bleu_score(self, pred, ref):
        return self.bleu.compute(predictions=[pred], references=[ref])

    def _meteor_score(self, pred, ref):
        return self.meteor.compute(predictions=[pred], references=[ref])

    def _sacrebleu(self, pred, ref):
        return self.sacre_bleu.compute(predictions=[pred], references=[ref])

    def _rouge_score(self, pred, ref):
        return self.rouge.compute(predictions=[pred], references=[ref])

    def _normalize(self, score):
        return 100*score

    def data_scorer(self, data, model, tokenizer, torch_device):
        for i in tqdm(range(len(data))):
            pred = predict(model, tokenizer,
                           data.iloc[i, 0], data.iloc[i, 1], torch_device)
            ref = data.iloc[i, 2]

            bleu_score = self._bleu_score(pred, ref)
            meteor_score = self._meteor_score(pred, ref)
            sacrebleu_score = self._sacrebleu(pred, ref)
            rouge_score = self._rouge_score(pred, ref)

            n_bleu_score = self._normalize(bleu_score['bleu'])
            n_meteor_score = self._normalize(meteor_score['meteor'])
            n_rouge_score = self._normalize(rouge_score['rouge2'])
            n_rouge_L_score = self._normalize(rouge_score['rougeL'])

            self.bleu_avg.calc_avg(n_bleu_score)
            self.meteor_avg.calc_avg(n_meteor_score)
            self.rouge_avg.calc_avg(n_rouge_score)
            self.rouge_L_avg.calc_avg(n_rouge_L_score)
            self.sacrebleu_avg.calc_avg(sacrebleu_score['score'])

            self.results.append({'hyp': pred, 'reference': ref, 'bleu': n_bleu_score,
                                'meteor': n_meteor_score, 'sacre_bleu': sacrebleu_score, 'rouge': n_rouge_score, 'rougeL': n_rouge_L_score})

        self.results.append({'bleu_avg': self.bleu_avg.avg, 'meteor_avg': self.meteor_avg.avg,
                            'sacrebleu_avg': self.sacrebleu_avg.avg, 'rouge_avg': self.rouge_avg.avg, 'rouge_L_avg': self.rouge_L_avg.avg})

        return self.results

    def save_to_file(self):
        result = json.dumps(self.results)
        with open("""{path}/output/{dataset}_{model}_output.json""".format(path=root_path, dataset=args.dataset, model=args.model_name), 'w', encoding='utf-8') as f:
            f.write(result)
