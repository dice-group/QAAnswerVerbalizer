"""DATASETS: QUALD-9 PLUS, VQUANDA, GRAILQA, VANILLA
"""

import sys
sys.path.append("./")
from constants import get_project_root
from constants import args
from constants import *
from tqdm import tqdm
import re
from question_query_answer import question_query_answer as qqa
import json
from flair.data import Sentence
from flair.models import SequenceTagger
from rdflib import Graph, URIRef, BNode, Literal, Variable, Namespace
from rdflib.namespace import RDF



root_path = get_project_root()
SYMBOLS = {'{': 'brack_open ', '}': ' brack_close', '.': ' sep_dot ',
           '!=': 'not_equal', '=': 'equal',  "'": ''
           }

tagger = SequenceTagger.load('ner')


def vquanda(path):
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
    for i in tqdm(range(len(data))):
        d = data[i]
        qa = qqa(d)
        index = qa.get_id()
        question = qa.get_question()
        verbalization = qa.get_verbalized()
        query = qa.get_query()
        answers = qa.get_ans_label()

        if query is not None and query != "" and question is not None and question != "" and verbalization is not None and verbalization != "":
            triples = query_to_rdf(query, endpoint='dbpedia')
            triples = make_verbose_vquanda(triples)
            query = make_verbose_vquanda(query)
            query = query.lower()
        if args.mask_ans:
            question, entities = mask_entities(question)
            verbalization = mask_answer(verbalization)
            verbalization, _ = mask_entities(verbalization)

        info = {'index': index, 'question': question,
                'query': query, 'verbalization': verbalization, 'answers': answers, 'triples': triples}
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

        if query is not None and query != "" and question is not None and question != "" and verbalization is not None and verbalization != "":
            if verbose:
                triples = query_to_rdf(query, endpoint='wikidata')
                query = make_verbose(query, lang)
            if args.mask_ans:
                question = mask_entities(question)
                verbalization = mask_answer(verbalization)
                verbalization = mask_entities(verbalization)
            query = query.lower()

            info = {'index': index, 'question': question,
                    'query': query, 'verbalization': verbalization, 'answers': answers, 'triples': triples}
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

        if query_sparql is not None and question is not None and verbalization is not None:
            if args.mask_ans:
                question = mask_entities(question)
                verbalization = mask_answer(verbalization)
                verbalization = mask_entities(verbalization)
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
        verbalizations = qa.get_verbalized()
        query = qa.get_query()
        answers = qa.get_ans_label()

        if query is not None and query != "" and question is not None and question != "" and verbalizations is not None and verbalizations != "":
            if verbose:
                triples = query_to_rdf(query, endpoint='dbpedia')
                query = make_verbose_vquanda(query)
            if args.mask_ans:
                question, q_ent = mask_entities(question)
                verbalizations = [mask_answer(v) for v in verbalizations]
                verbalizations, entities = mask_entities(verbalizations)
                query = query.lower()
                info = {'index': index, 'question': question,
                        'query': query, 'triples': triples, 'verbalization': verbalizations, 'entities': entities, 'q_ent': q_ent, 'answers': answers}
            else:
                info = {'index': index, 'question': question,
                        'query': query, 'triples': triples, 'verbalization': verbalizations, 'answers': answers}
            final_dict[i] = info
    return final_dict


def vanilla(path):
    """
    preprocess the the Vanilla dataset. 

    Parameter
    ---------

    path: str
    path of the json file of the dataset.

    Returns: dict
    -------------

    dict of pre-processed index, question, verbalization, answers
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
        answers = qa.get_ans_label()
        if question is not None and question != "" and verbalization is not None and verbalization != "":
            if args.mask_ans:
                question = mask_entities(question)
                masked_v = verbalization.replace(answers.lower(), ANS_TOKEN)
                verbalization = mask_entities(masked_v)

            info = {'index': index, 'question': question,
                    'verbalization': verbalization, 'answers': answers}
            final_dict[i] = info
    return final_dict


def mask_answer(verbalization):
    ans_pattern = re.compile(r'\[.*?\]')
    verbalization = ans_pattern.sub(ANS_TOKEN, verbalization)
    return verbalization


def mask_entities(verbalizations):
    is_list = True
    if isinstance(verbalizations, list):
        sentences = [Sentence(verbalization)
                     for verbalization in verbalizations]

    else:
        sentences = [Sentence(verbalizations)]
        is_list = False
    tagger.predict(sentences)

    m_verbalizations = []
    entities = []
    for sentence in sentences:
        verbalization = sentence.to_original_text()
        for ent in sentence.get_spans('ner'):
            if ent.text != 'ANS':
                entities.append(ent.text)
                verbalization = verbalization.replace(ent.text, ENT_TOKEN)
        m_verbalizations.append(verbalization)
    if is_list:
        return m_verbalizations, entities
    else:
        return m_verbalizations[0], entities


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
        label = qqa.get_label_endpoint(e=entity)
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
        # label = qqa.get_label_endpoint(e, endpoint='dbpedia')
        label = (e.rsplit('/', 1))[-1]
        label = label.replace('>', '')

        if args.mask_ans and 'resource' in e:
            query = query.replace(e, ENT_TOKEN)
        else:
            query = query.replace(e, label)

    query = query.replace('\n', '')
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


def query_to_rdf(query, endpoint):
    def make_rdf(input, result):
        if input.startswith("<") and input.endswith(">"):
            input = input[1:-1]
            if input == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
                return RDF.type
            else:
                return URIRef(input)

        elif any(input.startswith(k) for k, v in namespaces.items()):
            for k, v in namespaces.items():
                if input.startswith(k):
                    input = input[len(k)+1:-1]
                    return v[input]

        elif input.startswith('"') and input.endswith('"'):
            return Literal(input)

        elif input.startswith("?"):
            input = input.replace('?', '')
            if input == var:
                if result != 'answer':
                    return result
                else:
                    return URIRef(result)

            else:
                return Variable(input)
        else:
            return BNode()

    def get_namespace(query):
        namespace_pattern = re.compile(
            r'PREFIX\s+([a-zA-Z0-9_]+):\s*<([^>]+)>')
        prefix_matches = re.findall(namespace_pattern, query)
        namespace = {}
        for pref, uri in prefix_matches:
            namespace[pref] = Namespace(uri)
        return namespace

    def prep_query(query, endpoint='dbpedia'):
        where_pattern = re.compile(r"WHERE\s*{\s*(.*?)\s*}\s*", re.DOTALL)
        if 'WHERE' not in query:
            pos_of_select = query.find('SELECT')
            if pos_of_select != -1:
                end_pos_of_select = query.find('{', pos_of_select)
                if end_pos_of_select != -1:
                    query = query[:end_pos_of_select] + \
                        'WHERE' + query[end_pos_of_select:]

        where_clause = where_pattern.findall(query)[0]

        # Set Endpoints for parsing
        if endpoint == 'dbpedia':
            endpoint_uri = '<https://dbpedia.org/sparql>'
        elif endpoint == 'wikidata':
            endpoint_uri = '<https://query.wikidata.org/sparql>'

        # Add SERVICE to parse query
        new_where = f"SERVICE {endpoint_uri} {{ {where_clause} }}"
        query = query.replace(where_clause, new_where)

        # Fix count for vquanda
        if args.dataset == 'vquanda':
            if 'COUNT(' in query:
                query = query.replace('COUNT(', '(COUNT(?uri) AS ')

        # Get Filter Clause and exclude from triples
        filter_clause = None
        if 'FILTER' in where_clause:
            filter_pattern = re.compile(r'(FILTER\s+\(.*?\))')
            filter_clause = re.findall(filter_pattern, query)[0]
            where_clause = where_clause.replace(filter_clause, '')

        pattern_triples = re.compile(
            r'(\?[a-zA-Z0-9_]+|<[^>]+>|[\w:]+)\s+(<[^>]+>|[\w:]+)\s+(<[^>]+>|[\w:]+|\?\w+)')
        triple_matches = re.findall(pattern_triples, where_clause)

        return query, triple_matches, filter_clause

    wrap_query, matches, filter_clause = prep_query(query, endpoint)
    graph = Graph()

    if 'PREFIX' in query:
        namespaces = get_namespace(query)
        for k, v in namespaces.items():
            graph.bind(k, v)

    try:
        qresults = graph.query(wrap_query)
    except Exception as e:
        print(f"Error: {e}")
        return None

    isLit = False
    query_type = qresults.type
    if query_type == 'ASK':
        result = qresults.askAnswer
    else:
        var = str(qresults.vars[0])
        result = 'answer'
        for binding in qresults:
            if isinstance(binding, Literal):
                isLit = True
            result = binding[0]
    if args.mask_ans:
        if isLit:
            result = Literal(ANS_TOKEN)
        else:
            result = URIRef(ANS_TOKEN)

    if matches:
        for s, p, o in matches:
            print(s, p, o)
            s = make_rdf(s, result)
            p = make_rdf(p, result)
            o = make_rdf(o, result)
            graph.add((s, p, o))

    triples = graph.serialize(format='nt')
    if filter_clause:
        triples = triples + filter_clause

    return triples


def write_to_file(dataset, data, name):
    with open("""{path}/data/{dataset}/preprocessed_{dataset}_{name}.json""".format(path=root_path, dataset=dataset, name=name), 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)


if __name__ == '__main__':

    filepath = """{path}/data/{dataset}/{name}.json""".format(
        path=root_path, dataset=args.dataset, name=args.name)
    if args.dataset == 'vquanda':
        pre_data = vquanda(filepath)

    if args.dataset == 'quald':
        pre_data = quald(filepath, num=args.num, lang=args.lang)

    if args.dataset == 'grailQA':
        pre_data = grailQA(filepath, num=args.num)

    if args.dataset == 'paraQA':
        pre_data = paraQA(filepath)

    if args.dataset == 'vanilla':
        pre_data = vanilla(filepath)

    write_to_file(args.dataset, pre_data, args.name)
