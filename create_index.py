__author__ = 'Abhijeet'

from elasticsearch import Elasticsearch
from os import listdir
import string
import cPickle as cp


class CreateIndex(object):
    """
    Creates an Index in Elastic Search
    """
    def __init__(self):

        self.es = Elasticsearch()
        self.file_corpus_path = 'AP_DATA/ap89_collection'

        stop_file_path = "AP_DATA"
        stop_file_name = 'stoplist.txt'
        with open(stop_file_path+"/"+stop_file_name) as f:
            self.stop_file_data = f.readlines()
        # little cleaning up the stop list data
        self.stop_file_data = [x.replace("\n",'') for x in self.stop_file_data]

        # fetch qrel data
        self.qrel_data = {}
        with open(stop_file_path+"/qrels.adhoc.51-100.AP89.txt") as f:
            for every_line in f:
                data = every_line.split()
                if self.qrel_data.get(data[2]) is None:
                    self.qrel_data[data[2]] = {data[0]: data[3]}
                else:
                    self.qrel_data[data[2]][data[0]] = data[3]

    def compute_index(self):
        """
        Creates the index in ES
        :return: Void
        """
        doc_length_dict = {}
        count = 0
        file_names = listdir(self.file_corpus_path)
        for apfile in file_names:
            f = open(self.file_corpus_path+"/"+apfile)
            extract_text = False
            text_string = ""
            for line in f:
                if line.startswith("<DOCNO>"):
                    doc_no = line[8:21]
                    # if doc_no not in qrel_data.keys():
                        # break
                    # else:
                    count += 1
                if line.startswith("</TEXT>"):
                    extract_text = False
                if extract_text:
                    text_string += line
                if line.startswith("<TEXT>"):
                    extract_text = True
                if line.startswith("</DOC>"):
                    text_string = text_string.replace("\n"," ")

                    tlist = text_string.split()
                    slist = []
                    for i in range(len(tlist)):
                        if tlist[i] in self.stop_file_data:
                            slist.append('')
                        else:
                            slist.append(tlist[i])
                    text_string = ' '.join(slist)

                    # Converting to lower text
                    text_string = text_string.lower()
                    # Removing punctuations from query
                    for p in string.punctuation:
                        if p != '_' and p != '-' and p != '\'':
                            text_string = text_string.replace(p, " ")
                    text_string = text_string.replace("  ", " ")

                    doc_length = len(text_string.split())
                    doc_length_dict[doc_no] = doc_length

                    doc = {'docno': doc_no, 'text': text_string, 'doclength': doc_length}

                    res = self.es.index(index="ap_dataset", doc_type='document', id=doc_no, body=doc)

                    extract_text = False
                    text_string = ""
            f.close()
            # print count
            with open("doc_length.txt", "w") as f:
                cp.dump(doc_length_dict, f)

if __name__ == "__main__":
    ci = CreateIndex()
    ci.compute_index()