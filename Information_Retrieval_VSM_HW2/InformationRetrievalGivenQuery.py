import sys
import math
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from operator import itemgetter

q_dict_tf = {}
q_dict_tfidf = {}
q_dict_bm25 = {}
k = 4
b = 0.0001


def simplify_q_input():
    input = sys.argv[4].lower()
    stop_words = set(stopwords.words("english"))
    tokenizer = RegexpTokenizer(r'\w+')
    ps = PorterStemmer()
    question = tokenizer.tokenize(input)
    filtered_question = [word for word in question if not word in stop_words]
    for i in range(len(filtered_question)):
        filtered_question[i] = ps.stem(filtered_question[i])
    return filtered_question


def tf_idf(q, inverted_index, normal_IDF, all_docs_len):
    Wq = {}
    cossim = {}
    for w in inverted_index:
        if q_dict_tfidf.get(w) == None:
            q_dict_tfidf[w] = {}
            for key in inverted_index[w]:
                q_dict_tfidf[w][key] = inverted_index[w][key] * normal_IDF[w]

    ## max word in query ##
    max_word = 0
    for W_i in q:
        cnt = 0
        for W_j in q:
            if W_i == W_j:
                cnt += 1
        if max_word < cnt:
            max_word = cnt

    ## Wq ##
    for W_i in q:
        cnt = 0
        for W_j in q:
            if W_i == W_j:
                cnt += 1
        if q_dict_tfidf.get(W_i) == None:
            if normal_IDF.get(W_i) == None:
                Wq.update({W_i: 0})
        else:
            Wq.update({W_i: (cnt / max_word) * normal_IDF[W_i]})

    ## cossim ##
    for key in all_docs_len:
        numer = 0
        sum_1 = 0
        sum_2 = 0

        for W_i in inverted_index:
            if q_dict_tfidf[W_i].get(key) != None:
                sum_1 += q_dict_tfidf[W_i][key] * q_dict_tfidf[W_i][key]
            else:
                sum_1 += 0

        for W_i in q:
            if q_dict_tfidf.get(W_i) != None:
                if q_dict_tfidf[W_i].get(key) != None and Wq.get(W_i) != None:
                    numer += q_dict_tfidf[W_i][key] * Wq[W_i]
            else:
                numer += 0
            sum_2 += Wq[W_i] * Wq[W_i]
        prod = sum_1 * sum_2
        denom = math.sqrt(prod)
        cossim[key] = numer / denom

    return sorted(cossim.items(), key=itemgetter(1), reverse=True)


def q_bm25(q, inverted_index, all_docs_len, bm25_IDF, avgdl):
    for key in all_docs_len:
        q_dict_bm25[key] = 0
        for W_i in q:
            try:
                numer = inverted_index[W_i][key] * (k + 1)
                denom = inverted_index[W_i][key] + k * (1 - b + b * (all_docs_len[key] / avgdl))
                q_dict_bm25[key] += bm25_IDF[W_i] * (numer / denom)
            except:
                continue
    return sorted(q_dict_bm25.items(), key=itemgetter(1), reverse=True)


def compute_avgdl(all_docs_len):
    total = 0
    for doc in all_docs_len:
        total += all_docs_len[doc]
    return total / len(all_docs_len)


def ret_info(dict):
    inverted_index = dict["TF"]
    all_docs_len = dict["len_by_doc_name"]
    normal_IDF = dict["normal_IDF"]
    bm25_IDF = dict["BM25_IDF"]
    avgdl = compute_avgdl(all_docs_len)

    q = simplify_q_input()

    if q == False:
        print("Error in query question")
        return

    if sys.argv[2] == "tfidf":
        relevant_docs = tf_idf(q, inverted_index, normal_IDF, all_docs_len)
        threshold = 0.001
    if sys.argv[2] == "bm25":
        relevant_docs = q_bm25(q, inverted_index, all_docs_len, bm25_IDF, avgdl)
        threshold = 0.001

    f = open("ranked_query_docs.txt", "w")
    for i in range(0, len(relevant_docs)):
        if (relevant_docs[i][1] >= threshold):
            f.write(relevant_docs[i][0] + "\n")
    f.close()
