from constants import *
import json
import pandas as pd
from tqdm import tqdm
import evaluate
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
import nltk
from nltk.translate import chrf_score
import hashlib

root_path = get_project_root()


class PrepareInput():
    """
    Prepares input from different preprocessed datasets.
    Datasets:
    - VQuAnDa
    - ParaQA
    - QALD-9-plus
    - GrailQA
    - Vanilla

    Parameters
    -------------------------
    path: string 
    path of the preprocessed file

    Returns
    -------------------------
    make_df(): pandas Dataframe

    get_data(): list of lists

    """

    def __init__(self, path):
        self.args = args
        self.path = path
        self.data = None
        self.func_map = {
            'vquanda': self.prep_vquanda,
            'vanilla': self.prep_vanilla,
            'grailQA': self.prep_grailQA,
            'paraQA': self.prep_paraQA,
            'quald': self.prep_quald
        }
        self.all_data = []
        self.load_data()

    def load_data(self):
        with open(self.path, 'r') as f:
            d = json.load(f)
        for _, value in tqdm(d.items()):
            self.data = value
            if self.args.dataset in self.func_map:
                self.func_map[self.args.dataset]()
            else:
                print(f"Dataset {self.args.dataset} is not in the map")

    def prep_conditional_field(self):
        self.cond_input = self.answers + ' ' + SEP_TOKEN + ' ' + self.cond_input

    def prep_vquanda(self):
        self.question = self.data['question']
        if self.args.mask_ans:
            self.answers = ANS_TOKEN
        else:
            self.answers = self.data['answers']

        verbalization = self.data['verbalization']
        if self.args.mode == 'query':
            self.cond_input = self.data['query']
        elif self.args.mode == 'triples':
            self.cond_input = self.data['triples']

        self.prep_conditional_field()
        self.all_data.append([self.question, self.cond_input, verbalization])

    def prep_vanilla(self):
        self.question = self.data['question']
        if self.args.mask_ans:
            self.answers = ANS_TOKEN
        else:
            self.answers = self.data['answers']
        verbalization = self.data['verbalization']
        # No query present in Vanilla
        self.cond_input = self.answers
        self.all_data.append([self.question, self.cond_input, verbalization])

    def prep_grailQA(self):
        self.question = self.data['question']
        if self.args.mask_ans:
            self.answers = ANS_TOKEN
        else:
            self.answers = self.data['answers']
        verbalization = self.data['verbalization']
        if self.args.mode == 'query':
            self.cond_input = self.data['query']['sparql']
        elif self.args.mode == 'triples':

            self.cond_input = get_triples_string(self.data['query']['graph'])

        self.prep_conditional_field()
        self.all_data.append([self.question, self.cond_input, verbalization])

    def prep_paraQA(self):
        self.question = self.data['question']
        if self.args.mask_ans:
            self.answers = ANS_TOKEN
        else:
            self.answers = self.data['answers']
        verbalization = self.data['verbalization']

        if self.args.mode == 'query':
            self.cond_input = self.data['query']
        elif self.args.mode == 'triples':
            self.cond_input = self.data['triples']
        self.prep_conditional_field()
        for v in verbalization:
            self.all_data.append([self.question, self.cond_input, v])

    def prep_quald(self):
        self.question = self.data['question']
        if self.args.mask_ans:
            self.answers = ANS_TOKEN
        else:
            self.answers = self.data['answers']

        verbalization = self.data['verbalization']
        if self.args.mode == 'query':
            self.cond_input = self.data['query']
        elif self.args.mode == 'triples':
            self.cond_input = self.data['triples']

        self.prep_conditional_field()
        self.all_data.append([self.question, self.cond_input, verbalization])

    def make_df(self):
        return pd.DataFrame(self.all_data, columns=["questions", "cond_input", "labels"])

    def get_data(self):
        return self.all_data


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

    def _ter_score(self, pred, ref):
        return self.ter.compute(predictions=[pred], references=[
            [ref]], case_sensitive=False)

    def _chrf_score(self, pred, ref):
        return self.chrf.compute(
            predictions=[pred], references=[[ref]], word_order=2)

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
            chrf_score = self._chrf_score(pred, ref)
            ter_score = self._ter_score(pred, ref)

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
    3. Chrf

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
        self.chrf_avg = NltkAverageScore()

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

    def _chrf_score(self, pred, ref):
        return nltk.translate.chrf_score.sentence_chrf(pred, ref, min_len=1, max_len=6)

    def _normalize(self, score):
        return 100*score

    def _hash_text(self, text):
        return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16) % (10 ** 8)

    def data_scorer(self, data, model, tokenizer, torch_device):
        obj = Score()
        seen_questions = {}
        for i in tqdm(range(len(data))):
            pred = predict(model, tokenizer,
                           data.iloc[i, 0], data.iloc[i, 1], torch_device)
            ref = data.iloc[i, 2]

            pred = self.tokenize_sentence(pred)
            ref = self.tokenize_sentence(ref)
            bleu_score = self._bleu_score(pred, ref)
            meteor_score = self._meteor_score(pred, ref)
            chrf_score = self._chrf_score(pred, ref)
            sacrebleu_score = obj._sacrebleu(' '.join(pred), ' '.join(ref))
            ter_score = obj._ter_score(' '.join(pred), ' '.join(ref))
            chrf_score_hf = obj._chrf_score(' '.join(pred), ' '.join(ref))
            rougeL_score = obj._rouge_score(
                ' '.join(pred), ' '.join(ref))['rougeL']
            n_meteor_score = self._normalize(meteor_score)

            self.results.append({
                'hyp': ' '.join(pred),
                'reference': ' '.join(ref),
                'bleu': {
                    '1': self._normalize(bleu_score['1']),
                    '2': self._normalize(bleu_score['2']),
                    '3': self._normalize(bleu_score['3']),
                    '4': self._normalize(bleu_score['4'])
                },
                'chrf++': self._normalize(chrf_score),
                'meteor': n_meteor_score,
                'sacrebleu': sacrebleu_score,
                'ter': ter_score,
                'chrf_hf': chrf_score_hf,
                'rougeL': rougeL_score
            })

            if args.dataset == 'paraQA':
                question_hash = self._hash_text(' '.join(data.iloc[i, 1]))
                if question_hash in seen_questions:
                    seen_questions[question_hash]['bleu']['1'].append(
                        self._normalize(bleu_score['1']))
                    seen_questions[question_hash]['bleu']['2'].append(
                        self._normalize(bleu_score['2']))
                    seen_questions[question_hash]['bleu']['3'].append(
                        self._normalize(bleu_score['3']))
                    seen_questions[question_hash]['bleu']['4'].append(
                        self._normalize(bleu_score['4']))
                    seen_questions[question_hash]['meteor'].append(
                        n_meteor_score)
                    seen_questions[question_hash]['sacrebleu'].append(
                        sacrebleu_score['score'])
                    seen_questions[question_hash]['ter'].append(
                        ter_score['score'])
                    seen_questions[question_hash]['chrf'].append(chrf_score)

                else:
                    seen_questions[question_hash] = {
                        'bleu': {
                            '1': self._normalize(bleu_score['1']),
                            '2': self._normalize(bleu_score['2']),
                            '3': self._normalize(bleu_score['3']),
                            '4': self._normalize(bleu_score['4'])
                        },
                        'meteor': self._normalize(n_meteor_score),
                        'sacrebleu': self._normalize(sacrebleu_score),
                        'chrf': [self._normalize(chrf_score)],
                        'ter': [ter_score['score']]

                    }
            else:
                self.bleu_avg['1'].calc_avg(self._normalize(bleu_score['1']))
                self.bleu_avg['2'].calc_avg(self._normalize(bleu_score['2']))
                self.bleu_avg['3'].calc_avg(self._normalize(bleu_score['3']))
                self.bleu_avg['4'].calc_avg(self._normalize(bleu_score['4']))
                self.meteor_avg.calc_avg(n_meteor_score)
                obj.sacrebleu_avg.calc_avg(sacrebleu_score['score'])
                self.chrf_avg.calc_avg(self._normalize(chrf_score))
                obj.chrf_avg.calc_avg(self._normalize(chrf_score_hf['score']))
                obj.ter_avg.calc_avg(ter_score['score'])
                obj.rouge_L_avg.calc_avg(rougeL_score)

        if args.dataset == 'paraQA':
            for scores in seen_questions.values():
                self.bleu_avg['1'].calc_avg(max(scores['bleu_score']['1']))
                self.bleu_avg['2'].calc_avg(max(scores['bleu_score']['2']))
                self.bleu_avg['3'].calc_avg(max(scores['bleu_score']['3']))
                self.bleu_avg['4'].calc_avg(max(scores['bleu_score']['4']))
                self.meteor_avg.calc_avg(max(scores['meteor']))
                obj.sacrebleu_avg.calc_avg(max(scores['sacrebleu']))
                self.chrf_avg.calc_avg(max(scores['chrf']))
                self.obj.ter_avg.calc_avg(min(scores['ter']))

        self.results.append({
            'bleu_avg': {
                '1': self.bleu_avg['1'].avg,
                '2': self.bleu_avg['2'].avg,
                '3': self.bleu_avg['3'].avg,
                '4': self.bleu_avg['4'].avg
            },
            'meteor_avg': self.meteor_avg.avg,
            'chrf_avg': self.chrf_avg.avg,
            'sacrebleu_avg': obj.sacrebleu_avg.avg,
            'ter_avg': obj.ter_avg.avg,
            'chrf_hf_avg': obj.chrf_avg.avg,
            'rougeL_avg': obj.rouge_L_avg.avg
        })

        return self.results

    def save_to_file(self):
        result = json.dumps(self.results)
        with open("""{path}/output/{dataset}_{model}_nltk_output.json""".format(path=root_path, dataset=args.dataset, model=args.model_name), 'w', encoding='utf-8') as f:
            f.write(result)
