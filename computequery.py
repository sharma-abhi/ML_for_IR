__author__ = 'Abhijeet'

from elasticsearch import Elasticsearch
from elasticsearch import client
import string


class ComputeQuery(object):
    """
    Returns the query scores
    """
    def __init__(self):
        self.search_size = 15000
        self.es = Elasticsearch(timeout = 180)

        file_path = 'AP_DATA'
        query_filename = 'query_desc.51-100.short.txt'

        with open(file_path+"/"+query_filename) as f:
            self.query_data = f.readlines()

        stop_file_name = 'stoplist.txt'
        with open(file_path+"/"+stop_file_name) as f:
            self.stop_file_data = [x.replace("\n", '') for x in f]

        # calculated using Sense
        # total no. of documents in corpus
        self.maxSize = 84678
        # no. of unique terms in all documents
        self.vocabSize = 140292
        self.lamb = 0.5

    def find_average_doc_length(self):
        """
        Computes the average length of documents
        :return: float
        """
        avg_doc_length_result = self.es.search(index="ap_dataset", doc_type="document",
            body={"aggs": {"avg_doc_length": {"avg": {"script": "doc['doclength'].values"}}}})
        avg_doc_length = avg_doc_length_result['aggregations']['avg_doc_length']['value']
        return avg_doc_length

    def find_sum_doc_length(self):
        """
        Computes the total length of documents
        :return: float
        """
        sum_doc_length_result = self.es.search(index="ap_dataset",
                                               doc_type="document",
                                               body={"aggs":
                                                         {"sum_doc_length":
                                                              {"sum": {"script": "doc['doclength'].values"}}}})
        sum_doc_length = sum_doc_length_result['aggregations']['sum_doc_length']['value']
        return sum_doc_length

    def calc_frequencies(self, query, docno):
        ic = client.IndicesClient(self.es)
        sum_tf = 0
        sum_df = 0
        sum_ttf = 0

        query_array = []

        analyzed_result = ic.analyze(index="ap_dataset",analyzer="my_english",body=query)
        token_length = len(analyzed_result['tokens'])

        for i in range(token_length):
            query_array.append(str(analyzed_result['tokens'][i]['token']))

        res = self.es.termvector(index="ap_dataset", doc_type="document", id=docno, term_statistics=True)

        term_dict = res['term_vectors']['text']['terms']

        for term in query_array:
            if term in term_dict.keys():
                sum_tf += res['term_vectors']['text']['terms'][term]['term_freq']
                sum_df += res['term_vectors']['text']['terms'][term]['doc_freq']
                sum_ttf += res['term_vectors']['text']['terms'][term]['ttf']

        return sum_tf, sum_df, sum_ttf

    def calc_okapi_tf(self, query, query_no, avg_doc_length):
        """
        Calculates the OkapiTf scores
        :param query: str
        :param query_no: int
        :param avg_doc_length: float
        :return: okapi_tf scores: float
        """
        okapi_tf_scores = {}
        f_okapi_tf = open("Results/okapi_tf_output.txt",'a')
        query_array = []
        ic = client.IndicesClient(self.es)

        analyzed_result = ic.analyze(index="ap_dataset",analyzer="my_english",body=query)
        token_length = len(analyzed_result['tokens'])
        for i in range(token_length):
            query_array.append(str(analyzed_result['tokens'][i]['token']))

        query_body = {"query":
                          {"function_score": {"query": {"match": {"text": query}},
                                              "functions": [
                                                  {"script_score":
                                                      {"script": "getOkapiTF", "lang": "groovy",
                                                       "params": {"query": query_array, "field": "text",
                                                                  "avgLength": avg_doc_length}}}],
                                              "boost_mode": "replace"}}, "fields":["stream_id"]}

        okapi_result = self.es.search(index="ap_dataset", doc_type="document", size=self.search_size,
                                      analyzer="my_english", body=query_body)
        result_size = len(okapi_result['hits']['hits'])

        rank = 1
        for i in range(result_size):
            doc_id = str(okapi_result['hits']['hits'][i]['_id'])
            score = okapi_result['hits']['hits'][i]['_score']
            if score != 0:
                f_okapi_tf.write(query_no + " Q0 " + doc_id + " " + str(rank) + " " + str(score) + " Exp\n")
                okapi_tf_scores[doc_id] = score
                rank += 1
        f_okapi_tf.close()
        return okapi_tf_scores

    def calc_tf_idf(self, query, queryNo, avgDocLength, nDocs):

        tf_idf_scores = {}
        fTtIdf = open("Results/tf_idf_output.txt",'a')
        queryArray = []
        ic = client.IndicesClient(self.es)
        analyzedResult = ic.analyze(index="ap_dataset",analyzer="my_english",body=query)
        tokenLength = len(analyzedResult['tokens'])
        for i in range(tokenLength):
            queryArray.append(str(analyzedResult['tokens'][i]['token']))

        queryBody = {"query": {"function_score": {"query": {"match": {"text": query}},
            "functions":[{"script_score": {"script": "getTfIdf", "lang": "groovy",
                "params": {"query": queryArray, "field": "text", "avgLength": avgDocLength, "ndocs" : nDocs}}}],
            "boost_mode": "replace"}}, "fields":["stream_id"]}
        tfIdfResult = self.es.search(index="ap_dataset", doc_type="document", size=self.search_size,
            analyzer = "my_english", body = queryBody)

        resultSize = len(tfIdfResult['hits']['hits'] )
        rank = 1
        for i in range(resultSize):

            docId = str(tfIdfResult['hits']['hits'][i]['_id'])
            score = tfIdfResult['hits']['hits'][i]['_score']
            if score != 0:
                fTtIdf.write(queryNo + " Q0 " + docId + " " + str(rank) + " " + str(score) + " Exp\n")
                tf_idf_scores[docId] = score
                rank = rank + 1

        fTtIdf.close()
        return tf_idf_scores

    def calc_okapi_bm(self, query, queryNo, avgDocLength, nDocs):
        okapi_bm_scores = {}
        fokapiBM = open("Results/okapiBM_output.txt",'a')
        queryArray = []
        ic = client.IndicesClient(self.es)
        analyzedResult = ic.analyze(index="ap_dataset",analyzer="my_english",body=query)
        tokenLength = len(analyzedResult['tokens'])
        for i in range(tokenLength):
            queryArray.append(str(analyzedResult['tokens'][i]['token']))

        queryBody = {"query": {"function_score": {"query": {"match": {"text": query}},
            "functions":[{"script_score": {"script": "getOkapiBM", "lang": "groovy",
                "params": {"query": queryArray, "field":"text", "avgLength": avgDocLength, "ndocs" : nDocs}}}],
            "boost_mode": "replace"}}, "fields":["stream_id"]}
        okapiBMResult = self.es.search(index="ap_dataset", doc_type="document", size=self.search_size,
            analyzer = "my_english", body = queryBody)

        resultSize = len(okapiBMResult['hits']['hits'] )
        rank = 1
        for i in range(resultSize):

            docId = str(okapiBMResult['hits']['hits'][i]['_id'])
            score = okapiBMResult['hits']['hits'][i]['_score']
            if score != 0:
                fokapiBM.write(queryNo + " Q0 " + docId + " " + str(rank) + " " + str(score) + " Exp\n")
                okapi_bm_scores[docId] = score
                rank += 1
        fokapiBM.close()
        return okapi_bm_scores

    def calc_laplace(self, query, queryNo, vocabSize):
        laplace_scores = {}
        flaplace = open("Results/laplace_output.txt",'a')
        queryArray = []
        ic = client.IndicesClient(self.es)
        analyzedResult = ic.analyze(index="ap_dataset",analyzer="my_english",body=query)
        tokenLength = len(analyzedResult['tokens'])
        for i in range(tokenLength):
            queryArray.append(str(analyzedResult['tokens'][i]['token']))

        queryBody = {"query": {"function_score": {"query": {"match": {"text": query}},
            "functions":[{"script_score": {"script": "getLaplace", "lang": "groovy",
                "params": {"query": queryArray, "field":"text", "vocabSize": vocabSize}}}],
            "boost_mode": "replace"}}, "fields":["stream_id"]}
        laplaceResult = self.es.search(index="ap_dataset", doc_type="document", size=self.search_size,
            analyzer="my_english", body=queryBody)

        resultSize = len(laplaceResult['hits']['hits'] )
        rank = 1
        for i in range(resultSize):

            docId = str(laplaceResult['hits']['hits'][i]['_id'])
            score = laplaceResult['hits']['hits'][i]['_score']
            if score != 0:
                flaplace.write(queryNo + " Q0 " + docId + " " + str(rank) + " " + str(score) + " Exp\n")
                laplace_scores[docId] = score
                rank = rank + 1

        flaplace.close()
        return laplace_scores

    def calc_jm(self, query, queryNo, lamb, sumDocLength):
        jm_scores = {}
        fjm = open("Results/jm_output.txt",'a')
        queryArray = []
        ic = client.IndicesClient(self.es)
        analyzedResult = ic.analyze(index="ap_dataset",analyzer="my_english",body=query)
        tokenLength = len(analyzedResult['tokens'])
        for i in range(tokenLength):
            queryArray.append(str(analyzedResult['tokens'][i]['token']))

        queryBody = {"query": {"function_score": {"query": {"match": {"text": query}},"functions":[{"script_score": {"script": "getJM", "lang": "groovy", "params": {"query": queryArray, "field":"text", "lamb": lamb, "sumdoclength": sumDocLength }}}], "boost_mode": "replace"}}, "fields":["stream_id"]}
        jmResult = self.es.search(index="ap_dataset",
                                  doc_type="document",
                                  size=self.search_size,
                                  analyzer="my_english",
                                  body=queryBody)

        resultSize = len(jmResult['hits']['hits'] )
        rank = 1
        for i in range(resultSize):

            docId = str(jmResult['hits']['hits'][i]['_id'])
            score = jmResult['hits']['hits'][i]['_score']
            if score != 0:
                fjm.write(queryNo + " Q0 " + docId + " " + str(rank) + " " + str(score) + " Exp\n")
                jm_scores[docId] = score
                rank = rank + 1

        fjm.close()
        return jm_scores

    def compute_values(self):

        print "\nCalculating average Document Lengths..."
        avg_doc_length = self.find_average_doc_length()
        print "\naverage doc Length is ", avg_doc_length

        print "\nCalculating sum Document Lengths..."
        sum_doc_length = self.find_sum_doc_length()
        print "\nSum doc Length is ", sum_doc_length

        okapi_tf_results = {}
        tf_idf_results = {}
        okapi_bm_results = {}
        laplace_results = {}
        jm_results = {}
        sum_term_frequency = {}
        sum_doc_frequency = {}
        sum_ttf_frequency = {}

        query_count = 0

        for line in self.query_data:

            line_array = line.split(".")
            query_no = line_array[0]
            query = line_array[1]

            # Removing Ignored words from query
            query = query.replace(" Document will discuss ","")
            query = query.replace(" Document will report ","")
            query = query.replace(" Document will include ","")
            query = query.replace(" Document must describe ","")
            query = query.replace(" Document must identify ","")

            # Removing punctuations from query
            for p in string.punctuation:
                if p != '_' and p != '-':
                    query = query.replace(p, " ")

            tlist = query.split()
            slist = []
            for i in range(len(tlist)):
                if tlist[i] in self.stop_file_data:
                    slist.append('')
                else:
                    slist.append(tlist[i])
            query = ' '.join(slist)

            # Removing double spaces from query
            query = query.replace("  "," ")
            # Converting to lower text
            query = query.lower()

            print "Running Query no: ", query_no," query count", query_count
            query_count += 1

            print "Fetched results for Query no: ", query_no

            okapi_tf_results[query_no] = self.calc_okapi_tf(query, query_no, avg_doc_length)
            tf_idf_results[query_no] = self.calc_tf_idf(query, query_no, avg_doc_length, self.maxSize)
            okapi_bm_results[query_no] = self.calc_okapi_bm(query, query_no, avg_doc_length, self.maxSize)
            laplace_results[query_no] = self.calc_laplace(query, query_no, self.vocabSize)
            jm_results[query_no] = self.calc_jm(query, query_no, self.lamb, sum_doc_length)
        return okapi_tf_results, tf_idf_results, okapi_bm_results, laplace_results, jm_results

'''if __name__ == "__main__":
    cq = ComputeQuery()
    cq.compute_values()'''


