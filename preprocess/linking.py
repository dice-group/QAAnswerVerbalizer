import requests


def linking(entity, lang='en', endpoint='wikidata'):
    """Takes a wikidata entity and the language as input and queries it using the API. It returns the label of this entity"""
    label = ""
    if endpoint == 'wikidata':
        api_url = "https://query.wikidata.org/sparql"
        prefix = "wd:"
    else:
        api_url = "http://dbpedia.org/sparql"
        prefix = ""
    sparql_query = """
  SELECT ?label
  WHERE {{
    {prefix}{entity} rdfs:label ?label .
    FILTER (lang(?label) = '{lang}')
  }}""".format(prefix=prefix, entity=entity, lang=lang)

    headers = {
        "User-Agent": "QAAnswerVerbalizerApp/1.0",
        "Accept": "application/sparql-results+json"
    }
    response = requests.get(api_url, headers=headers,
                            params={"query": sparql_query})
    data = response.json()
    if "results" in data and isinstance(data["results"], dict) and len(data["results"]["bindings"]) > 0:
        first_result = data["results"]["bindings"][0]
        if isinstance(first_result, dict) and "label" in first_result:
            label = first_result["label"].get("value")
        else:
            label = None

    return label
