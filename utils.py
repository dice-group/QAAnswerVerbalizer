from constants import *
import json
import pandas as pd
from tqdm import tqdm
import evaluate
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
import nltk

root_path = get_project_root()


def make_df(path=str):
    with open(path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame({"questions": [], "queries": [], 'labels': []}, index=[])
    df = df.astype('object')

    index = 0
    for _, value in data.items():
        question = value['question']
        verbalization = value['verbalization']
        answers = value['answers']
        if args.dataset != 'vanilla':
            query = value['query']

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

        if args.dataset == 'vanilla':
            ans_query = ans
        else:
            # add Answer token or Answer and Separator token along with query
            ans_query = ans + ' ' + SEP_TOKEN + ' ' + query

        if args.dataset == 'paraQA' and args.name == 'train':
            # In-case of ParaQA. make different pairs from all verbalizations
            for v in verbalization:
                df.loc[index] = [question, ans_query, v]
                index += 1

        else:
            df.loc[index] = [question, ans_query, verbalization]
            index += 1

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
        **input, max_length=100).to(torch_device)
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
        self.chrf = evaluate.load('chrf')
        self.ter = evaluate.load('ter')

        self.bleu_avg = AverageScore()
        self.sacrebleu_avg = AverageScore()
        self.meteor_avg = AverageScore()
        self.rouge_avg = AverageScore()
        self.rouge_L_avg = AverageScore()
        self.chrf_avg = AverageScore()
        self.ter_avg = AverageScore()

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
            chrf_score = self.chrf.compute(
                predictions=[pred], references=[[ref]], word_order=2)
            ter_score = self.ter.compute(predictions=[pred], references=[
                                         [ref]], case_sensitive=False)

            n_bleu_score = self._normalize(bleu_score['bleu'])
            n_meteor_score = self._normalize(meteor_score['meteor'])
            n_rouge_score = self._normalize(rouge_score['rouge2'])
            n_rouge_L_score = self._normalize(rouge_score['rougeL'])

            self.bleu_avg.calc_avg(n_bleu_score)
            self.meteor_avg.calc_avg(n_meteor_score)
            self.rouge_avg.calc_avg(n_rouge_score)
            self.rouge_L_avg.calc_avg(n_rouge_L_score)
            self.sacrebleu_avg.calc_avg(sacrebleu_score['score'])
            self.chrf_avg.calc_avg(chrf_score['score'])
            self.ter_avg.calc_avg(ter_score['score'])

            self.results.append({'hyp': pred, 'reference': ref, 'bleu': n_bleu_score,
                                'meteor': n_meteor_score, 'sacre_bleu': sacrebleu_score, 'rouge': n_rouge_score, 'rougeL': n_rouge_L_score, 'chrf': chrf_score, 'ter': ter_score})

        self.results.append({'bleu_avg': self.bleu_avg.avg, 'meteor_avg': self.meteor_avg.avg,
                            'sacrebleu_avg': self.sacrebleu_avg.avg, 'rouge_avg': self.rouge_avg.avg, 'rouge_L_avg': self.rouge_L_avg.avg, 'chrf_avg': self.chrf_avg.avg, 'ter_avg': self.ter_avg.avg})

        return self.results

    def save_to_file(self):
        result = json.dumps(self.results)
        with open("""{path}/output/{dataset}_{model}_output.json""".format(path=root_path, dataset=args.dataset, model=args.model_name), 'w', encoding='utf-8') as f:
            f.write(result)


class NltkAverageScore():
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def calc_avg(self, score):
        self.sum += score
        self.count += 1
        self.avg = self.sum / self.count


class NltkScore(object):
    """
    Calculates scores (nltk) and average on different metrics after prediction. 
    The Bleu, Meteor normalized. 
    --------
    Metrics
    --------
    1. Bleu
    2. Meteor

    -------------------------
    returns: dict

    """

    def __init__(self):
        self.results = []
        self.bleu_avg = {
            '1': NltkAverageScore(),
            '2': NltkAverageScore(),
            '3': NltkAverageScore(),
            '4': NltkAverageScore()
        }
        self.meteor_avg = NltkAverageScore()

    def tokenize_sentence(self, sentence):
        """Tokenize and add spaces with punctuations to be treated separately."""
        punct = {
            ",": " ,",
            "'": " '",
            "?": " ?",
            ".": " ."
        }
        for k, v in punct.items():
            sentence = sentence.replace(k, v)
        return sentence.split()

    def _bleu_score(self, pred, ref):
        return {
            '1': nltk.translate.bleu_score.sentence_bleu([ref], pred, weights=(1.0, 0.0, 0.0, 0.0)),
            '2': nltk.translate.bleu_score.sentence_bleu([ref], pred, weights=(0.5, 0.5, 0.0, 0.0)),
            '3': nltk.translate.bleu_score.sentence_bleu([ref], pred, weights=(0.33, 0.33, 0.33, 0.0)),
            '4': nltk.translate.bleu_score.sentence_bleu([ref], pred, weights=(0.25, 0.25, 0.25, 0.25)),
        }

    def _meteor_score(self, pred, ref):
        return nltk.translate.meteor_score.single_meteor_score(' '.join(ref), ' '.join(pred))

    def _normalize(self, score):
        return 100*score

    def data_scorer(self, data, model, tokenizer, torch_device):
        for i in tqdm(range(len(data))):
            pred = predict(model, tokenizer,
                           data.iloc[i, 0], data.iloc[i, 1], torch_device)
            ref = data.iloc[i, 2]

            pred = self.tokenize_sentence(pred)
            ref = self.tokenize_sentence(ref)
            bleu_score = self._bleu_score(pred, ref)
            meteor_score = self._meteor_score(pred, ref)

            n_meteor_score = self._normalize(meteor_score)

            self.bleu_avg['1'].calc_avg(self._normalize(bleu_score['1']))
            self.bleu_avg['2'].calc_avg(self._normalize(bleu_score['2']))
            self.bleu_avg['3'].calc_avg(self._normalize(bleu_score['3']))
            self.bleu_avg['4'].calc_avg(self._normalize(bleu_score['4']))
            self.meteor_avg.calc_avg(n_meteor_score)

            self.results.append({
                'hyp': pred,
                'reference': ref,
                'bleu': {
                    '1': self._normalize(bleu_score['1']),
                    '2': self._normalize(bleu_score['2']),
                    '3': self._normalize(bleu_score['3']),
                    '4': self._normalize(bleu_score['4'])
                },
                'meteor': n_meteor_score
            })

        self.results.append({
            'bleu_avg': {
                '1': self.bleu_avg['1'].avg,
                '2': self.bleu_avg['2'].avg,
                '3': self.bleu_avg['3'].avg,
                '4': self.bleu_avg['4'].avg
            },
            'meteor_avg': self.meteor_avg.avg
        })

        return self.results

    def save_to_file(self):
        result = json.dumps(self.results)
        with open("""{path}/output/{dataset}_{model}_nltk_output.json""".format(path=root_path, dataset=args.dataset, model=args.model_name), 'w', encoding='utf-8') as f:
            f.write(result)
