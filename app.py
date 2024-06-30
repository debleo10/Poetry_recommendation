from copyreg import pickle
import random
from hinglish_spellcheck import *
import warnings
warnings.filterwarnings('ignore')
from pyphonetics import Soundex, Metaphone, RefinedSoundex
from flask import Flask, render_template, url_for, request
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST', 'GET'])
def result():
    print("Result function called")
    import numpy as np
    import nltk
    import math
    from nltk.stem import PorterStemmer
    from nltk.tokenize import TweetTokenizer
    from nltk.corpus import stopwords
    from natsort import natsorted
    import string
    import pickle

    def loading_file(file):
        with open(file, 'r', errors="surrogateescape") as f:
            ele = f.read()
        return ele

    def rem_(strr):
        strr = strr.replace(".", "")
        strr = strr.replace("-", "")
        strr = strr.replace("'", "")
        strr = strr.replace('"', '')
        return strr

    def Processing(input_str):
        tokenizer = TweetTokenizer()
        lst_token = tokenizer.tokenize(input_str.lower())
        return lst_token

    root_path = os.getcwd()
    print(f"Root path: {root_path}")
    files = natsorted(os.listdir(root_path + "/Poems"))
    print(f"Files found: {files}")

    Tokenized_Word_list = []
    with open("Tokenized_Word_list.pkl", "rb") as f:
        Tokenized_Word_list = pickle.load(f)
    unique_tokens = set()
    for i in range(len(Tokenized_Word_list)):
        unique_tokens = unique_tokens.union(set(Tokenized_Word_list[i]))

    dict_lst = []
    for i in range(len(Tokenized_Word_list)):
        count_doci = dict.fromkeys(unique_tokens, 0)
        for token in set(Tokenized_Word_list[i]):
            count_doci[token] += 1
        dict_lst.append(count_doci)

    tf_lst_binary = dict_lst

    def calculate_idf(doc_counts):
        idf = dict.fromkeys(doc_counts[0].keys(), 0)
        for doc in doc_counts:
            for token, count in doc.items():
                if count != 0:
                    idf[token] += 1

        for token, count in idf.items():
            idf[token] = math.log(len(doc_counts) / float(count) + 1)

        return idf

    idf = calculate_idf(dict_lst)

    def calculate_tfidf(tf, idf):
        tfidf = dict()
        for token, count in tf.items():
            tfidf[token] = count * idf[token]
        return tfidf

    with open("tfidf_binary.pkl", "rb") as f:
        tfidf_binary = pickle.load(f)
    import pandas as pd

    tfidf_binary_df = pd.DataFrame(tfidf_binary)

    unique_words_corpus = set()
    for i in range(len(Tokenized_Word_list)):
        unique_words_corpus = unique_words_corpus.union(set(sorted(Tokenized_Word_list[i])))

    metaphone_map = dict()
    metaphone = Metaphone()
    for word in unique_words_corpus:
        meta_code = metaphone.phonetics(word)
        if meta_code not in metaphone_map.keys():
            metaphone_map[meta_code] = [word]
        else:
            temp_list = metaphone_map[meta_code]
            temp_list.append(word)
            metaphone_map[meta_code] = temp_list

    Query = request.form.get("name")
    print(f"Query received: {Query}")
    Query = rem_(Query)
    Query = correct_spelling_hinglish(Query, unique_words_corpus, metaphone_map)
    Query_lst = Processing(Query)

    unique_tokens_Query = set()
    for i in range(len(Query_lst)):
        unique_tokens_Query = unique_tokens_Query.union(set(Query_lst))
    count_dociQ_bin = dict.fromkeys(unique_tokens_Query, 0)
    for token in set(Query_lst):
        count_dociQ_bin[token] += 1
    idfQ = dict.fromkeys(count_dociQ_bin.keys(), 0)
    for token in idfQ.keys():
        idfQ[token] = idf[token]

    tfidfQ = calculate_tfidf(count_dociQ_bin, idfQ)

    def cosine_similarity(tfidf_dict_qry, df, tokens, doc_num):
        dot_product = 0
        qry_mod = 0
        doc_mod = 0

        for keyword in tokens:
            dot_product += tfidf_dict_qry[keyword] * df[keyword][doc_num]
            qry_mod += tfidf_dict_qry[keyword] * tfidf_dict_qry[keyword]
            doc_mod += df[keyword][doc_num] * df[keyword][doc_num]
        qry_mod = math.sqrt(qry_mod)
        doc_mod = math.sqrt(doc_mod)

        denominator = qry_mod * doc_mod
        if denominator == 0:
            return 0
        else:
            cos_sim = dot_product / denominator
            return cos_sim

    def rank_similarity_docs(data):
        cos_sim = []
        for doc_num in range(0, len(data)):
            cos_sim.append(cosine_similarity(tfidfQ, tfidf_binary_df, Query_lst, doc_num))
        return cos_sim

    similarity_docs = rank_similarity_docs(Tokenized_Word_list)
    result = {}
    for i in range(len(files)):
        result[files[i]] = similarity_docs[i]

    name = sorted(result.items(), key=lambda item: item[1], reverse=True)
    dictt = {}
    global matchingPoem
    matchingPoem = []
    for i in range(5):
        with open(root_path + "/Poems/" + name[:5][i][0], "r") as f:
            dictt[name[:5][i][0]] = f.read()
        matchingPoem.append({'name': name[:5][i][0], 'text': rem_(dictt[name[:5][i][0]])})

    matchingText = name[:5][0][0]

    candidate_titles = []
    for ff in matchingPoem:
        candidate_titles.append(ff["name"])

    def get_candidate_names(titles):
        names = []
        for t in titles:
            names.append(t[:-4].split("_")[1])
        return names

    candidate_names = get_candidate_names(candidate_titles)

    def get_suggestions(candidate_poets):
        poet_poems_dict = dict()
        for i in range(len(files)):
            poet = files[i][:-4].split("_")[1]
            poem_title = files[i][:-4].split("_")[0]
            if poet not in poet_poems_dict.keys():
                poet_poems_dict[poet] = [poem_title]
            elif poet in poet_poems_dict.keys():
                temp_list1 = poet_poems_dict[poet]
                temp_list1 += [poem_title]
                poet_poems_dict[poet] = temp_list1

        def append_poet_name(temp_list3, poet):
            tempp = []
            for x in temp_list3:
                xx = x + "_" + poet + ".txt"
                tempp += [xx]
            return tempp

        suggested_poems_names = []
        poems_per_poets = [3, 2, 2, 2, 1]
        for i in range(len(candidate_poets)):
            x = poems_per_poets[i]
            temp_list2 = poet_poems_dict[candidate_poets[i]]
            random.shuffle(temp_list2)
            suggested_poems_names += append_poet_name(temp_list2[:x], candidate_poets[i])
        suggested_poems = []
        for p in suggested_poems_names:
            temp_dict = dict()
            temp_dict['name'] = p
            with open("Poems/" + p) as f:
                temp_dict["text"] = f.read()
            suggested_poems.append(temp_dict)

        return suggested_poems

    global suggestions
    suggestions = get_suggestions(candidate_names)

    return render_template('result.html', matchingText=matchingText, len2=len(suggestions), len1=len(matchingPoem), suggestions=suggestions, matchingPoem=matchingPoem)


@app.route('/poem/<string:name>', methods=['POST', 'GET'])
def poem(name):
    text = []
    for i in matchingPoem:
        if i.get("name") == name:
            text.append(i.get("text"))

    return render_template('poem.html', text=text, len=len(text), name=name.split("_")[0].upper())


@app.route('/poem_suggested/<string:name>', methods=['POST', 'GET'])
def poem_sugg(name):
    text = []
    for i in suggestions:
        if i.get("name") == name:
            text.append(i.get("text"))
    return render_template('poem_sugg.html', text=text, len=len(text), name=name.split("_")[0].upper())

if __name__ == "__main__":
    app.run(debug=True)
