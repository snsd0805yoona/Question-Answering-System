import glob
import os
from typing import DefaultDict
from collections import defaultdict
import pickle
# os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import pysolr
from nltk.tokenize import sent_tokenize

import task1 as t1
path='articles'
solr = pysolr.Solr('http://localhost:8983/solr/gettingstarted', timeout=10)
fname=[]
docs=[]

#read file in folders
def readfiles():
    for filename in glob.glob(os.path.join(path, '*.txt')):
        with open(os.path.join(os.getcwd(), filename), 'r', encoding="utf-8") as f:
            content = f.read()
            s=""
            sentence = sent_tokenize(content)
            token=[]
            filter_t=[]
            lemma_word=[]
            entity=[]
            label = []
            tree=[]
            root=[]
            stemma=[]
            synonymns_list=[]
            hypernyms_list=[]
            hyponyms_list=[]
            meronyms_list=[]
            holonyms_list=[]
            sentence_feature_dict = [dict() for x in range(len(sentence))]
            for i in range(0, len(sentence)):
                t, filtered = t1.tokenize(sentence[i])
                tree.append(t1.dependency_tree(sentence[i]))
                token.append(t)
                filter_t.append(filtered)
                lemma_word.append(t1.lemmatize(filter_t[i]))
                stemma.append(t1.stemmatize(filter_t[i]))
                for word in lemma_word[i]:
                    s=s+word+" "
                tree, r = t1.dependency_tree(sentence[i])
                root.append(r)
                # POS_tags.append(t1.pos_tag(token[i]))
                syn, hyper, hypo, mero, holo = t1.synsets(t1.pos_tag(token[i]))
                synonymns_list.append(syn)
                hypernyms_list.append(hyper)
                hyponyms_list.append(hypo)
                meronyms_list.append(mero)
                holonyms_list.append(holo)
                e, l = t1.ner(sentence[i])
                entity.append(e)
                label.append(l)                

            fname.append(filename)
            docs.append(s)

            
    # return synonymns_list       
            add_in_solr(filename, sentence, filter_t, stemma, lemma_word, root, entity, label, sentence_feature_dict, synonymns_list, hypernyms_list, hyponyms_list, meronyms_list, holonyms_list)        


def add_in_solr(filename, sentence, filter_t, stemma, lemma_word, root, entity, label, sentence_feature_dict, synonymns_list, hypernyms_list, hyponyms_list, meronyms_list, holonyms_list):
    for i in range(0, len(sentence)):
        sentence_feature_dict[i]["filename"] = filename
        sentence_feature_dict[i]["sentence"] = sentence[i]
        sentence_feature_dict[i]["stemmatized"] = stemma[i]
        sentence_feature_dict[i]["filtered"] = filter_t[i]
        sentence_feature_dict[i]["lemmatized"] = lemma_word[i]
        sentence_feature_dict[i]["root"] = root[i]
        sentence_feature_dict[i]["entity"] = entity[i]
        sentence_feature_dict[i]["label"] = label[i]
        sentence_feature_dict[i]["synonym"] = synonymns_list[i]
        sentence_feature_dict[i]["hypernym"] = hypernyms_list[i]
        sentence_feature_dict[i]["hyponym"] = hyponyms_list[i]
        sentence_feature_dict[i]["meronym"] = meronyms_list[i]
        sentence_feature_dict[i]["holonym"] = holonyms_list[i]


    solr.add(sentence_feature_dict, commit = True)


def main():
    readfiles()
    with open('docs.pickle', 'wb') as f:
        pickle.dump(docs, f)
    with open('fname.pickle', 'wb')as f:
        pickle.dump(fname, f)

if __name__=="__main__":
    main()
# def get_top_3(question, tf_idf):
#     qtf_idf = vectorizer.transform([question])
#     res = cosine_similarity(tf_idf, qtf_idf)
#     res = res.ravel().argsort()[-3:]
#     result=[]
#     for num in res:
#         result.append(fname[num])

#     return result
# print(syn[0])
# print(hyper[0])
# print(hypo[0])

# print(n[0])

