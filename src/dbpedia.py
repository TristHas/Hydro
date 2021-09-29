import pickle, os
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON

dico = {
    "堤高": "http://ja.dbpedia.org/property/堤高",
    "堤頂長": "http://ja.dbpedia.org/property/堤頂長",
    "堤体積": "http://ja.dbpedia.org/property/堤体積",
    "流域面積": "http://ja.dbpedia.org/property/流域面積",
    "湛水面積": "http://ja.dbpedia.org/property/湛水面積",
    "総貯水容量": "http://ja.dbpedia.org/property/総貯水容量",
    "有効貯水容量": "http://ja.dbpedia.org/property/有効貯水容量",
    "利用目的": "http://ja.dbpedia.org/property/利用目的",
    "事業主体": "http://ja.dbpedia.org/property/事業主体",
    "電気事業者": "http://ja.dbpedia.org/property/電気事業者",
    "発電所名": "http://ja.dbpedia.org/property/発電所名",
}

def query_db_dam(name):
    queryString = "SELECT ?p ?o  WHERE {<http://ja.dbpedia.org/resource/" + name + "> ?p ?o .} LIMIT 1000"

    sparql = SPARQLWrapper("http://ja.dbpedia.org/sparql")
    sparql.setQuery(queryString)
    sparql.setReturnFormat(JSON)
    return parse_query(sparql.query().convert()["results"]["bindings"])

def parse_query(results):
    if results:
        P,O = list(zip(*[[result["p"]["value"], result["o"]["value"]] for result in results]))
        df = pd.Series(O, index=P)

        wiki_details = {}
        for k,v in dico.items():
            try:
                res = df[v]
                if isinstance(res, pd.Series):
                    res = [x.split("/")[-1] for x in res.values.tolist()]
                wiki_details[k]=res
            except KeyError:
                pass
        return wiki_details
    else:
        return None