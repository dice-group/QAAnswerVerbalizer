
class question_query_answer:
    """Gives the questions, queries, answers and their id from the json data"""
    def __init__(self, data):
        self.data = data

    def get_question(self, lang='en'):
        """Get the question based on the language"""
        for question in self.data['question']:
            if question['language'] == lang:
                return question['string']

    def get_query(self):
        """Get the query"""
        if 'query' in self.data:
            for k,v in self.data['query'].items():
                return v

    def get_answers(self):
        """Get all the answers"""
        return self.data['answers']

    def get_id(self):
        """Get the id of the object. One id can be mapped to questions, query and answers """
        return self.data['id']
    
    def get_ans_label(self):
        list1= self.data['answers'][1]['label']

        if list1 is list :
            join_list= ', '.join(list1)
            string_list= str(join_list)
            return string_list
        else:
            return list1
    
    def get_verbalized(self):
        return self.data['verbalized']['en']