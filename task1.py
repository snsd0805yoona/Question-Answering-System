from os import set_inheritable
import nltk
from nltk.corpus import wordnet as wn
from itertools import chain
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
import string
import spacy
sentences=["Tucson (/ˈtuːsɒn/ /tuːˈsɒn/) is a city and the county seat of Pima County, Arizona, United States, and home to the University of Arizona.", "The 2010 United States Census put the population at 520,116, while the 2013 estimated population of the entire Tucson metropolitan statistical area (MSA) was 996,544.",\
            "The Tucson MSA forms part of the larger Tucson-Nogales combined statistical area (CSA), with a total population of 980,263 as of the 2010 Census.", "Tucson is the second-largest populated city in Arizona behind Phoenix, both of which anchor the Arizona Sun Corridor.", "The city is  located 108 miles (174 km) southeast of Phoenix and 60 mi (97 km) north of the U.S.-Mexico border.", "Tucson is the 33rd largest city and the 59th largest metropolitan area in the United States.",\
              "Roughly 150 Tucson companies are involved in the design and manufacture of optics and optoelectronics systems, earning Tucson the nickname Optics Valley."]
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('words')
# nltk.download('maxent_ne_chunker')
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stop_words_and_dot = stop_words | set(string.punctuation)

lemmatizer = nltk.WordNetLemmatizer()
ps = PorterStemmer()
nlp = spacy.load("en_core_web_sm")


def tokenize(sentence):
    t=nltk.word_tokenize(sentence)
    filtered_t =[token for token in t if not token.lower() in stop_words_and_dot]
    return t, filtered_t

def lemmatize(tokens):
    l=[]
    for i in range(len(tokens)):
        l.append(lemmatizer.lemmatize(tokens[i]))
    return l

def pos_tag(tokens):
    p = nltk.pos_tag(tokens)
    return p

def ner(sentence):
    entity = []
    label = []
    doc = nlp(sentence)
    for ent in doc.ents:
        entity.append(ent.text)
        label.append(ent.label_)
    
    return entity, label

def stemmatize(tokens):
  s = []
  for i in range(len(tokens)):
    s.append(ps.stem(tokens[i]))
  return s

def dependency_tree(text):
  # text = "What Proto-Indo-European term means skillful assembler?"
  doc = nlp(text)
  tree = []
  #format is [word,dependency,head word]
  s = list(doc.sents)
  r = ""
  for t in s:
    r = t.root.text
  for token in doc:
    tree.append([token.text, token.dep_, token.head.text])
  # To visualize the the tree
  return tree, r


def synsets(pos):
 
  synonymns_list = []
  hypernyms_list =[]
  hyponyms_list = []
  meronyms_list = []
  holonyms_list = []
  for word in pos:
    # print(word[1])
    if(word[1] in ['NN', 'NNS','NNP','NNPS']):
      for synset in wn.synsets(word[0],pos = wn.NOUN):

        for hyper in synset.hypernyms():
            for lemma in hyper.lemmas():
              hypernyms_list.append(lemma.name())

        for hypo in synset.hyponyms():
            for lemma in hypo.lemmas():
              hyponyms_list.append(lemma.name())

        for mero in synset.part_meronyms():
          for lemma in mero.lemmas():
            meronyms_list.append(lemma.name())

        for holo in synset.part_holonyms():
          for lemma in holo.lemmas():
            holonyms_list.append(lemma.name())

        for lemma in synset.lemmas():
            synonymns_list.append(lemma.name())


    if(word[1] in ['VB', 'VBD','VBG','VBN','VBP','VBZ']):

      for synset in wn.synsets(word[0],pos = wn.VERB):

        for hyper in synset.hypernyms():
            for lemma in hyper.lemmas():
              hypernyms_list.append(lemma.name())

        for lemma in synset.lemmas():
            synonymns_list.append(lemma.name())

    if(word[1] in ['JJ', 'JJR','JJS']):

      for synset in wn.synsets(word[0],pos = wn.ADJ):


        for lemma in synset.lemmas():
            synonymns_list.append(lemma.name())

    if(word[1] in ['RB', 'RBR','RBS']):

      for synset in wn.synsets(word[0],pos = wn.ADV):


        for lemma in synset.lemmas():
            synonymns_list.append(lemma.name())



  return list(set(synonymns_list)),list(set(hypernyms_list)),list(set(hyponyms_list)),list(set(meronyms_list)),list(set(holonyms_list))
#   return synonymns_list, hypernyms_list, hyponyms_list, meronyms_list, holonyms_list 
# def synsets(tokens):
#     synonymns=[]
#     hypernyms=[]
#     hyponyms=[]
#     meronyms=[]
#     holonyms=[]
#     for word in tokens:

if __name__ == '__main__':
  with open("articles/6.txt", 'r', encoding="utf-8") as f1:
            content = f1.read()
            sentence = sent_tokenize(content)
            tokens, filtered_tokens = tokenize(sentence)
            lemma = lemmatize(filtered_tokens)
            stem = stemmatize(filtered_tokens)
            pos = pos_tag(filtered_tokens)
            tree, root = dependency_tree(sentence)
            name_entity, name_label = ner(sentence)
            syn, hyper, hypo, mero, holo = synsets(pos)
            with open("pipelines_example.txt", mode="w", encoding="utf-8") as f:
                f.write("sentence: "+str(sentence)+"\n")
                f.write("tokens: "+str(tokens)+"\n")
                f.write("filtered_tokens: "+str(filtered_tokens)+"\n")
                f.write("lemmas: "+str(lemma)+"\n")
                f.write("stemma: "+str(stem)+"\n")
                f.write("Pos_tag: "+str(pos)+"\n")
                f.write("Dependecy_tree: "+str(tree)+"\n")
                f.write("root: "+str(root)+"\n")
                f.write("name_entity: "+str(name_entity)+"\n")
                f.write("name_label: "+str(name_label)+"\n")
                f.write("Synonym: "+str(syn)+"\n")
                f.write("Hypernym: "+str(hyper)+"\n")
                f.write("Hyponym: "+str(hypo)+"\n")
                f.write("Meronym: "+str(mero)+"\n")
                f.write("Holonym: "+str(holo)+"\n")
                f.write("-------------------------------------------------------------\n")