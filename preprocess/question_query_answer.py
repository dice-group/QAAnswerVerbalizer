from constants import *
from linking import linking as link
import re


class question_query_answer:
    """Gives the questions, queries, answers and their id from the json data"""

    def __init__(self, data):
        self.data = data
        self.dataset = args.dataset

    def get_question(self, lang='en'):
        """Get the question based on the language"""
        if self.dataset == 'quald':
            for question in self.data['question']:
                if question['language'] == lang:
                    return question['string']
                else:
                    return None
        if self.dataset == 'vquanda' or self.dataset == 'paraQA' or self.dataset == 'vanilla':
            return self.data['question']
        if self.dataset == 'grailQA':
            return self.data['question']
        else:
            return None

    def get_query(self):
        """Get the query"""
        if self.dataset == 'quald':
            if 'query' in self.data:
                for _, v in self.data['query'].items():
                    return v
        if self.dataset == 'vquanda' or self.dataset == 'paraQA':
            return self.data['query']
        if self.dataset == 'grailQA':
            query_sparql = self.data['sparql_query']
            query_graph = self.data['graph_query']
            return query_sparql, query_graph
        else:
            return None

    def get_answers(self):
        """Get all the answers"""
        if self.dataset == 'quald':
            return self.data['answers']
        if self.dataset == 'grailQA':
            return self.data['answer']

    def get_id(self):
        """Get the id of the object. One id can be mapped to questions, query and answers """
        if self.dataset == 'quald':
            return self.data['id']
        if self.dataset == 'vquanda' or self.dataset == 'paraQA':
            return self.data['uid']
        if self.dataset == 'grailQA':
            return self.data['qid']
        if self.dataset == 'vanilla':
            return self.data['question_id']
        else:
            return None

    @staticmethod
    def get_label_endpoint(e, endpoint="wikidata"):
        label = link(entity=e, endpoint=endpoint)
        return label

    def search_entity(self, uri):
        """Search Entity in the uri and get label from  endpoint"""

        pattern = r'entity/([Q]\d+)'
        res = re.search(pattern, uri, re.IGNORECASE)
        if res is not None:
            entity = res.group(1)
            label = self.get_label_endpoint(entity)
            return {'entity': entity, 'label': label}

    def get_ans_label(self):
        if self.dataset == 'quald':
            # Get label of the answers
            if 'boolean' in self.data['answers'][0]:
                ans = self.data['answers'][0]['boolean']
                return str(ans)
            else:
                ans = self.data['answers'][0]['results']['bindings']

                answers = []
                for a in ans[:args.ans_limit]:
                    for _, value in a.items():
                        cat = value['type']
                        if cat == 'literal':
                            e = value['value']
                        elif self.search_entity(value['value']):
                            e = self.search_entity(value['value'])['label']
                        else:
                            # if not found or could not fetch then add dummy
                            e = "answer"
                    answers.append(e)
                ans_string = ", ".join(answers)
                return ans_string

        if self.dataset == 'vquanda':
            # Get answers between []
            pattern = r'\[(.*?)\]'
            answers = re.findall(pattern, self.get_verbalized())
            ans_string = ", ".join(answers)
            return ans_string

        if self.dataset == 'grailQA':
            ans = self.get_answers()

            answers = []
            for a in ans[:args.ans_limit]:
                if a['answer_type'] == 'Entity':
                    label = a['entity_name']
                else:
                    label = a['answer_argument']
                answers.append(label)
            ans_string = (', ').join(answers)
            return ans_string

        if self.dataset == 'paraQA':
            pattern = r'\[(.*?)\]'
            verbalized_list = self.get_verbalized()
            answers = re.findall(pattern, verbalized_list[0])
            ans_string = ", ".join(answers)
            return ans_string

        if self.dataset == 'vanilla':
            return self.data['answer']

        else:
            return None

    def get_verbalized(self):
        """
        Replace the verbalization answers or token withing the [] in Qald and GrailQA.
        ParaQA has multiple verbalizations and hence a list is returned.
        """
        pattern = r'\[.*?\]'
        if self.dataset == 'quald':
            verbalized_ans = self.data['verbalized']['en']
            answers = "[" + self.get_ans_label() + "]"
            verbalized = re.sub(pattern, answers, verbalized_ans)
            return verbalized

        if self.dataset == 'vquanda':
            verbalized = self.data['verbalized_answer']
            return verbalized

        if self.dataset == 'grailQA':
            verbalized_ans = self.data['label']
            answers = "[" + self.get_ans_label() + "]"
            verbalized = re.sub(pattern, answers, verbalized_ans)
            return verbalized

        if self.dataset == 'paraQA':
            verbalized = []
            verbalized.append(self.data['verbalized_answer'])
            for i in range(2, 9):
                key = """verbalized_answer_{i}""".format(i=i)
                verbalized_ans = self.data[key]
                if verbalized_ans != "":
                    verbalized.append(verbalized_ans)
            return verbalized

        if self.dataset == 'vanilla':
            return self.data['answer_sentence']
        else:
            return None
