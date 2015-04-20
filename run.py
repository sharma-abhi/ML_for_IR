__author__ = 'Abhijeet'

import sys
import create_index
import computequery
import random
import cPickle as cp
import string
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def clean_query(query_string):
    """
    cleans up the query text
    :param query_string: str
    :return: query_string: str
    """
    stop_file_name = 'stoplist.txt'
    with open(file_path+"/"+stop_file_name) as f:
        stop_file_data = [x.replace("\n", '') for x in f]

    # Removing Ignored words from query
    query_string = query_string.replace(" Document will discuss ", "")
    query_string = query_string.replace(" Document will report ", "")
    query_string = query_string.replace(" Document will include ", "")
    query_string = query_string.replace(" Document must describe ", "")
    query_string = query_string.replace(" Document must identify ", "")

    # Removing punctuations from query
    for p in string.punctuation:
        if p != '_' and p != '-':
            query_string = query_string.replace(p, " ")

    tlist = query_string.split()
    slist = []
    for i in range(len(tlist)):
        if tlist[i] in stop_file_data:
            slist.append('')
        else:
            slist.append(tlist[i])
    query_string = ' '.join(slist)

    # Removing double spaces from query
    query_string = query_string.replace("  "," ")
    # Converting to lower text
    query_string = query_string.lower()

    return query_string

args = sys.argv
CREATE_IX = False
print "input args: ", args
if len(args) == 2 and args[1] == '-ci':
    CREATE_IX = True
elif len(args) == 1:
    pass
else:
    print "Error! Usage:  run [-ci]"
    exit(1)

if CREATE_IX:
    ci = create_index.CreateIndex()
    ci.compute_index()
else:
    cq = computequery.ComputeQuery()
    file_path = 'AP_DATA'
    qrel_data = {}
    query_no_list = []
    query_dict = {}

    with open(file_path+"/qrels.adhoc.51-100.AP89.txt") as f:
        for every_line in f:
            data = every_line.split()
            if qrel_data.get(int(data[0])) is None:
                qrel_data[int(data[0])] = {data[2]: data[3]}
            else:
                qrel_data[int(data[0])][data[2]] = data[3]

    with open(file_path+"/query_desc.51-100.short.txt") as f:
        for line in f:
            line_array = line.split(".")
            query_no_list.append(int(line_array[0]))
            query = line_array[1]
            query_dict[int(line_array[0])] = clean_query(query)

    okapi_tf_val, tf_idf_val, okapi_bm_val, laplace_val, jm_val = cq.compute_values()

    # fetch doc lengths
    with open("doc_length.txt") as f:
        doc_length = cp.load(f)

    # dividing initial list into train and test lists randomly.
    train_query_no = []
    test_query_no = []

    while len(query_no_list) != 5:
        choice = random.choice(query_no_list)
        train_query_no.append(choice)
        query_no_list.remove(choice)

    test_query_no = query_no_list

    train_query_no = sorted(train_query_no)
    test_query_no = sorted(test_query_no)
    print "Training queries: ", train_query_no
    print "Test queries: ", test_query_no

    # creating Test data frame
    order_list = []

    # feature : (queryid-docid)
    feature1,feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10 = \
        ({} for i in range(10))
    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11 = ([] for i in range(11))

    for query_no in train_query_no:
        val = qrel_data[query_no]
        for k, v in val.items():
            order_list.append((query_no, k))
            feature1[(query_no, k)] = str((query_no, k))
            feature2[(query_no, k)] = doc_length[k]
            #print " before calc"
            feature3[(query_no, k)], feature4[(query_no, k)], feature5[(query_no, k)] = \
                cq.calc_frequencies(query_dict[query_no], k)
            #print " after calc"
            if okapi_tf_val[str(query_no)].get(k) is None:
                feature6[(query_no, k)] = 0
            else:
                feature6[(query_no, k)] = okapi_tf_val[str(query_no)][k]
            if tf_idf_val[str(query_no)].get(k) is None:
                feature7[(query_no, k)] = 0
            else:
                feature7[(query_no, k)] = tf_idf_val[str(query_no)][k]
            if okapi_bm_val[str(query_no)].get(k) is None:
                feature8[(query_no, k)] = 0
            else:
                feature8[(query_no, k)] = okapi_bm_val[str(query_no)][k]
            if laplace_val[str(query_no)].get(k) is None:
                feature9[(query_no, k)] = 0
            else:
                feature9[(query_no, k)] = laplace_val[str(query_no)][k]
            if jm_val[str(query_no)].get(k) is None:
                feature10[(query_no, k)] = 0
            else:
                feature10[(query_no, k)] = jm_val[str(query_no)][k]

    for item in order_list:
        f1.append(feature1[item])
        f2.append(feature2[item])
        f3.append(feature3[item])
        f4.append(feature4[item])
        f5.append(feature5[item])
        f6.append(feature6[item])
        f7.append(feature7[item])
        f8.append(feature8[item])
        f9.append(feature9[item])
        f10.append(feature10[item])
        query_no, docno = item
        f11.append(int(qrel_data[query_no][docno]))

    df = pd.DataFrame({'ID': pd.Series(f1),
                       'DL': pd.Series(f2),
                       'sum TF': pd.Series(f3),
                       'sum DF': pd.Series(f4),
                       'sum TTF': pd.Series(f5),
                       'okapi TF': pd.Series(f6),
                       'TF IDF': pd.Series(f7),
                       'okapi BM': pd.Series(f8),
                       'laplace': pd.Series(f9),
                       'JM': pd.Series(f10),
                       'label': pd.Series(f11)})

    train_ids = df['ID'].values
    df = df.drop(['ID'], axis=1)
    cols = df.columns.tolist()
    cols = cols[3:4] + cols[:3] + cols[4:]
    df = df[cols]
    train_data = df.values
    train_data2 = df.drop(['label'], axis=1).values

    # creating Test data frame
    order_list = []
    feature1,feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10 = \
        ({} for i in range(10))
    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10 = ([] for i in range(10))

    for query_no in test_query_no:
        val = qrel_data[query_no]
        for k, v in val.items():
            order_list.append((query_no, k))
            feature1[(query_no, k)] = (query_no, k)
            feature2[(query_no, k)] = doc_length[k]
            feature3[(query_no, k)], feature4[(query_no, k)], feature5[(query_no, k)] = cq.calc_frequencies(query_dict[query_no], k)
            if okapi_tf_val[str(query_no)].get(k) is None:
                feature6[(query_no, k)] = 0
            else:
                feature6[(query_no, k)] = okapi_tf_val[str(query_no)][k]
            if tf_idf_val[str(query_no)].get(k) is None:
                feature7[(query_no, k)] = 0
            else:
                feature7[(query_no, k)] = tf_idf_val[str(query_no)][k]
            if okapi_bm_val[str(query_no)].get(k) is None:
                feature8[(query_no, k)] = 0
            else:
                feature8[(query_no, k)] = okapi_bm_val[str(query_no)][k]
            if laplace_val[str(query_no)].get(k) is None:
                feature9[(query_no, k)] = 0
            else:
                feature9[(query_no, k)] = laplace_val[str(query_no)][k]
            if jm_val[str(query_no)].get(k) is None:
                feature10[(query_no, k)] = 0
            else:
                feature10[(query_no, k)] = jm_val[str(query_no)][k]

    for item in order_list:
        f1.append(feature1[item])
        f2.append(feature2[item])
        f3.append(feature3[item])
        f4.append(feature4[item])
        f5.append(feature5[item])
        f6.append(feature6[item])
        f7.append(feature7[item])
        f8.append(feature8[item])
        f9.append(feature9[item])
        f10.append(feature10[item])

    df2 = pd.DataFrame({'ID': pd.Series(f1),
                        'DL': pd.Series(f2),
                        'sum TF': pd.Series(f3),
                        'sum DF': pd.Series(f4),
                        'sum TTF': pd.Series(f5),
                        'okapi TF': pd.Series(f6),
                        'TF IDF': pd.Series(f7),
                        'okapi BM': pd.Series(f8),
                        'laplace': pd.Series(f9),
                        'JM': pd.Series(f10)})
    test_ids = df2['ID'].values
    df2 = df2.drop(['ID'], axis=1)
    test_data = df2.values

    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])
    output_forest = forest.predict(test_data)

    # Random Forest Classifier on Test Data
    forest_dict = {}
    for item in zip(test_ids, output_forest):
        (query_no, docno), score = item
        if forest_dict.get(query_no) is None:
            forest_dict[query_no] = {docno: score}
        else:
            forest_dict[query_no][docno] = score

    with open("random_forest_test_output.txt", "w") as f:
        for query_no, val in forest_dict.items():
            rank = 1
            sorted_keys = sorted(val, key=val.get, reverse=True)
            for docno in sorted_keys:
                f.write(str(query_no) + " Q0 " + str(docno) + " " + str(rank) + " " + str(val[docno]) + " Exp\n")
                rank += 1
    print "Random Forest on test data complete"

    # Random Forest Classifier on Train Data
    train_output_forest = forest.predict(train_data2)
    forest_dict = {}

    for item in zip(test_ids, train_output_forest):
        (query_no, docno), score = item
        if forest_dict.get(query_no) is None:
            forest_dict[query_no] = {docno: score}
        else:
            forest_dict[query_no][docno] = score

    with open("random_forest_train_output.txt", "w") as f:
        for query_no, val in forest_dict.items():
            rank = 1
            sorted_keys = sorted(val, key=val.get, reverse=True)
            for docno in sorted_keys:
                f.write(str(query_no) + " Q0 " + str(docno) + " " + str(rank) + " " + str(val[docno]) + " Exp\n")
                rank += 1
    print "Random Forest on train data complete"

    # Linear Regression on Test Data
    linear_reg = LinearRegression(normalize=True)
    linear_reg = linear_reg.fit(train_data[0::, 1::], train_data[0::, 0])
    output_linear_reg = linear_reg.predict(test_data)

    linear_reg_dict = {}

    for item in zip(test_ids, output_linear_reg):
        (query_no, docno), score = item
        if linear_reg_dict.get(query_no) is None:
            linear_reg_dict[query_no] = {docno: score}
        else:
            linear_reg_dict[query_no][docno] = score

    with open("linear_reg_test_output.txt", "w") as f:
        for query_no, val in linear_reg_dict.items():
            rank = 1
            sorted_keys = sorted(val, key=val.get, reverse=True)
            for docno in sorted_keys:
                f.write(str(query_no) + " Q0 " + str(docno) + " " + str(rank) + " " + str(val[docno]) + " Exp\n")
                rank += 1
    print "Linear Regression on test data complete"

    # Linear Regression on Train Data
    train_output_linear_reg = linear_reg.predict(train_data2)
    linear_reg_dict = {}

    for item in zip(test_ids, train_output_linear_reg):
        (query_no, docno), score = item
        if linear_reg_dict.get(query_no) is None:
            linear_reg_dict[query_no] = {docno: score}
        else:
            linear_reg_dict[query_no][docno] = score

    with open("linear_reg_train_output.txt", "w") as f:
        for query_no, val in linear_reg_dict.items():
            rank = 1
            sorted_keys = sorted(val, key=val.get, reverse=True)
            for docno in sorted_keys:
                f.write(str(query_no) + " Q0 " + str(docno) + " " + str(rank) + " " + str(val[docno]) + " Exp\n")
                rank += 1

    print "Linear Regression on train data complete"

    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])
    output_forest = forest.predict(test_data)

    # Random Forest Classifier on Test Data
    forest_dict = {}
    for item in zip(test_ids, output_forest):
        (query_no, docno), score = item
        if forest_dict.get(query_no) is None:
            forest_dict[query_no] = {docno: score}
        else:
            forest_dict[query_no][docno] = score

    with open("random_forest_test_output.txt", "w") as f:
        for query_no, val in forest_dict.items():
            rank = 1
            sorted_keys = sorted(val, key=val.get, reverse=True)
            for docno in sorted_keys:
                f.write(str(query_no) + " Q0 " + str(docno) + " " + str(rank) + " " + str(val[docno]) + " Exp\n")
                rank += 1
    print "Random Forest on test data complete"

    # Random Forest Regressor on Train Data
    reg_forest = RandomForestRegressor(n_estimators=100)
    reg_forest = reg_forest.fit(train_data[0::, 1::], train_data[0::, 0])
    output_reg_forest = reg_forest.predict(test_data)

    # Random Forest Regressor on Test Data
    forest_dict = {}
    for item in zip(test_ids, output_reg_forest):
        (query_no, docno), score = item
        if forest_dict.get(query_no) is None:
            forest_dict[query_no] = {docno: score}
        else:
            forest_dict[query_no][docno] = score

    with open("random_forest_reg_test_output.txt", "w") as f:
        for query_no, val in forest_dict.items():
            rank = 1
            sorted_keys = sorted(val, key=val.get, reverse=True)
            for docno in sorted_keys:
                f.write(str(query_no) + " Q0 " + str(docno) + " " + str(rank) + " " + str(val[docno]) + " Exp\n")
                rank += 1
    print "Random Forest Regresssor on test data complete"

    # Random Forest Regressor on Train Data
    train_output_reg_forest = reg_forest.predict(train_data2)
    forest_dict = {}

    for item in zip(test_ids, train_output_reg_forest):
        (query_no, docno), score = item
        if forest_dict.get(query_no) is None:
            forest_dict[query_no] = {docno: score}
        else:
            forest_dict[query_no][docno] = score

    with open("random_forest_reg_train_output.txt", "w") as f:
        for query_no, val in forest_dict.items():
            rank = 1
            sorted_keys = sorted(val, key=val.get, reverse=True)
            for docno in sorted_keys:
                f.write(str(query_no) + " Q0 " + str(docno) + " " + str(rank) + " " + str(val[docno]) + " Exp\n")
                rank += 1
    print "Random Forest Regresssor on train data complete"
