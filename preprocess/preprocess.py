"""DATASETS: QUALD-9 PLUS, VQUANDA, GRAILQA
"""

import sys
sys.path.append("./")
from constants import get_project_root
from constants import args
from tqdm import tqdm
import re
from question_query_answer import question_query_answer as qqa
import json

root_path = get_project_root()
ANS_TOKEN = '[ANS]'
SYMBOLS = {'{': 'brack_open ', '}': ' brack_close', '.': ' sep_dot ',
           '!=': 'not_equal', '=': 'equal',  "'": ''
           }


def vquanda(path, num, verbose=True):
    """
    preprocess the the Vquanda dataset. 

    Parameter
    ---------

    path: str
    path of the json file of the dataset.

    num: int
    number of entries to be processed.

    verbose: bool
    if the query has to be made more natural language readable

    Returns: dict
    -------------

    dict of pre-processed index, question, query , verbalization, answers
    """
    with open(path, 'r') as f:
        data = json.loads(f.read())
    final_dict = {}
    for i in tqdm(range(num)):
        d = data[i]
        qa = qqa(d)
        index = qa.get_id()
        question = qa.get_question()
        verbalization = qa.get_verbalized()
        query = qa.get_query()
        answers = qa.get_ans_label()

        if query and question and verbalization is not (None and ""):
            if verbose:
                query = make_verbose_vquanda(query)
            query = query.lower()
        if args.mask_ans:
            verbalization = mask_answer(verbalization)

        info = {'index': index, 'question': question,
                'query': query, 'verbalization': verbalization, 'answers': answers}
        final_dict[i] = info
    return final_dict


def quald(path, num, lang='en', verbose=True):
    """
    pre-process the quald-9 plus dataset.

    Parameter
    ---------

    path: str
    path of the json file of the dataset.

    num: int
    number of entries to be processed.

    lang: str
    language of the data: 'en' for English, 'de' for German

    verbose: bool
    if the query has to be made more natural language readable

    Returns: dict
    -------------

    dict of pre-processed index, question, query , verbalization, answers

    """
    with open(path, 'r') as f:
        data = json.loads(f.read())
    final_dict = {}
    for i in tqdm(range(num)):
        d = data[str(i)]
        qa = qqa(d)
        index = qa.get_id()
        question = qa.get_question(lang)
        query = qa.get_query()
        verbalization = qa.get_verbalized()
        answers = qa.get_ans_label()

        if query is not (None and "") and question is not (None and "") and verbalization is not (None and ""):
            if verbose:
                query = make_verbose(query, lang)
            if args.mask_ans:
                verbalization = mask_answer(verbalization)
            query = query.lower()

            info = {'index': index, 'question': question,
                    'query': query, 'verbalization': verbalization, 'answers': answers}
            final_dict[i] = info
    return final_dict


def grailQA(path, num=280):
    """
    pre-process the grailQA dataset.

    Parameter
    ---------

    path: str
    path of the json file of the dataset.

    num: int
    number of entries to be processed.

    Returns: dict
    -------------

    dict of pre-processed index, question, query, verbalization, answers

    """
    def extract_nodes_entities(query_graph):
        symbol_dict = {'<': 'less than', '>': 'greater_than',
                       '<=': 'less_than_equal', '>=': 'greater_than_equal'}

        edges_dict = {(edge['start'], edge['end']): edge['friendly_name']
                      for edge in query_graph['edges']}
        nodes_dict = {}
        for node in query_graph['nodes']:
            func = node['function']
            if symbol_dict.get(func) is not None:
                func = symbol_dict.get(func)
            nodes_dict[node['nid']] = {
                'name': node['friendly_name'], 'function': func, 'type': node['node_type']}

        triples = []
        for e in edges_dict:
            s = nodes_dict[e[0]]
            o = nodes_dict[e[1]]
            p = {'name': edges_dict[(e[0], e[1])]}
            triple = [s, p, o]
            triples.append(triple)

        return triples

    with open(path, 'r') as f:
        data = json.loads(f.read())

    final_dict = {}
    for i in tqdm(range(num)):
        d = data[i]
        qa = qqa(d)
        index = qa.get_id()
        question = qa.get_question()
        query_sparql, query_graph = qa.get_query()
        triple = extract_nodes_entities(query_graph)
        verbalization = qa.get_verbalized()
        answers = qa.get_ans_label()

        if query_sparql and question and verbalization is not None:
            if args.mask_ans:
                verbalization = mask_answer(verbalization)
            query_sparql.lower()

        info = {'index': index, 'question': question, 'query': {'sparql': query_sparql,
                                                                'graph': triple}, 'verbalization': verbalization, 'answers': answers}
        final_dict[i] = info

    return final_dict


def paraQA(path, verbose=True):
    """
    pre-process the paraQA dataset.

    Parameter
    ---------

    path: str
    path of the json file of the dataset.

    verbose: bool
    if the query has to be made more natural language readable

    Returns: dict
    -------------

    dict of pre-processed index, question, query, verbalization, answers

    """

    with open(path, 'r') as f:
        data = json.loads(f.read())

    final_dict = {}
    for i in tqdm(range(len(data))):
        d = data[i]
        qa = qqa(d)
        index = qa.get_id()
        question = qa.get_question()
        verbalization = qa.get_verbalized()
        query = qa.get_query()
        answers = qa.get_ans_label()

        if query is not (None and "") and question is not (None and "") and verbalization is not (None and ""):
            if verbose:
                query = make_verbose_vquanda(query)
            if args.mask_ans:
                verbalization = mask_answer(verbalization)
            query = query.lower()

            info = {'index': index, 'question': question,
                    'query': query, 'verbalization': verbalization, 'answers': answers}
            final_dict[i] = info
    return final_dict


def mask_answer(verbalization):
    ans_pattern = r'\[.*?\]'
    ans = re.findall(ans_pattern, verbalization)
    if ans:
        verbalization = verbalization.replace(ans[0], ANS_TOKEN)

    return verbalization


def make_verbose(query, lang):
    """
    Makes QUALD (based on wikidata) query into a more readable format, by replacing URIs with the respective labels and 
    replacing Symbols and other operators with text descriptions.

    Parameter
    ---------

    query: str

    lang: str
    language of the data: 'en' for English, 'de' for German

    Returns:
    -------------

    query: str

    """
    # All Regular Expression Patterns to get entities, relations and prefix
    ent_pattern = re.compile(
        r'wd:(Q\d+)|<http://www\.wikidata\.org/entity/(Q\d+)>')
    rel_pattern = re.compile(
        r'(wdt|p|pq):(P\d+)\b|<http://www\.wikidata\.org/prop/direct/(P\d+)>')
    prefix_pattern = re.compile(r'PREFIX [^:]+: <[^>]+>\s*')

    ent = [e for sub in ent_pattern.findall(query) for e in sub if e]
    rel = [r for sub in rel_pattern.findall(query) for r in sub if r]
    ent = list(set(ent))
    rel = list(set(rel))

    for entity in ent:
        label = qqa.get_label_endpoint(entity)
        query = query.replace("""wd:{e}""".format(e=entity), label)
        query = query.replace(
            """<http://www.wikidata.org/entity/{e}>""".format(e=entity), label)

    for relation in rel:
        label = qqa.get_label_endpoint(relation)
        query = query.replace("""wdt:{r}""".format(r=relation), label)
        query = query.replace(
            """<http://www.wikidata.org/prop/direct/{r}>""".format(r=relation), label)

    query = prefix_pattern.sub('', query)
    query = replace_symbols_var(query)

    return query


def make_verbose_vquanda(query):
    """
    Makes Vquanda(based on DBPedia) query into a more readable format, by replacing URIs with the respective labels and 
    replacing Symbols and other operators with text descriptions.
    """
    pattern = re.compile(
        r'(<http://dbpedia.org/resource/[^>]+>) |(<http://dbpedia.org/property/[^>]+>) |(<http://dbpedia.org/ontology/[^>]+>)')
    ent = [e for sub in pattern.findall(query) for e in sub if e]
    ent = list(set(ent))
    for e in ent:
        # Get the labels of entities from the endpoint
        label = qqa.get_label_endpoint(e, endpoint='dbpedia')
        query = query.replace(e, label)

    type_url = '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'
    if type_url in query:
        query = query.replace(type_url, 'type')
    query = replace_symbols_var(query)

    return query


def replace_symbols_var(query):
    """Replace symbols in the query with more readable charaters in natural language"""
    for s in SYMBOLS:
        if s in query:
            query = query.replace(s, SYMBOLS[s])

    # Find and replace ? identifier in variables with var_
    var_pattern = re.compile(r'(?:\?\w+)')
    var = re.findall(var_pattern, query)
    for v in var:
        v_var = v.replace('?', 'var_')
        query = query.replace(v, v_var)

    return query


def write_to_file(dataset, data, name):
    with open("""{path}/data/{dataset}/preprocessed_{dataset}_{name}.json""".format(path=root_path, dataset=dataset, name=name), 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)


if __name__ == '__main__':

    filepath = """{path}/data/{dataset}/{name}.json""".format(
        path=root_path, dataset=args.dataset, name=args.name)
    if args.dataset == 'vquanda':
        pre_data = vquanda(filepath, num=args.num)

    if args.dataset == 'quald':
        pre_data = quald(filepath, num=args.num, lang=args.lang)

    if args.dataset == 'grailQA':
        pre_data = grailQA(filepath, num=args.num)

    if args.dataset == 'paraQA':
        pre_data = paraQA(filepath)

    write_to_file(args.dataset, pre_data, args.name)
