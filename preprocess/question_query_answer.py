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
        if self.dataset == 'vquanda':
            return self.data['question']
        else:
            return None

    def get_query(self):
        """Get the query"""
        if self.dataset == 'quald':
            if 'query' in self.data:
                for _, v in self.data['query'].items():
                    return v
        if self.dataset == 'vquanda':
            return self.data['query']

    def get_answers(self):
        """Get all the answers"""
        return self.data['answers']

    def get_id(self):
        """Get the id of the object. One id can be mapped to questions, query and answers """
        if self.dataset == 'quald':
            return self.data['id']
        if self.dataset == 'vquanda':
            return self.data['uid']

    def get_label_endpoint(self, e):
        return link(e)

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
        else:
            return None

    def get_verbalized(self):
        """Replace the verbalization answers or token withing the []"""
        if self.dataset == 'quald':
            pattern = r'\[.*?\]'
            verbalized_ans = self.data['verbalized']['en']
            answers = "[" + self.get_ans_label() + "]"
            verbalized = re.sub(pattern, answers, verbalized_ans)
            return verbalized

        if self.dataset == 'vquanda':
            verbalized = self.data['verbalized_answer']
            return verbalized
        else:
            return None
