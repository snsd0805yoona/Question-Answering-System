import pickle
import task1
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pysolr
import csv
import string 
import spacy
solr = pysolr.Solr('http://localhost:8983/solr/gettingstarted', timeout=10)
nlp = spacy.load("en_core_web_sm")
best=[]
questions=["Who mediated the truce with Khomeini?", "When did an empire collapse after Alexander's conquests?", 'What is the Leader of the Revolution also known as in Iran?', "What is the nickname for  Tucson?", "Who bought Arizona?", "When was Arizona purchased by Mexico?", "What type of fuel is used by Fajr-3 missile?", "Who succeeded Reza Shah?", \
 "What led to students capturing the US embassy?", "Who is the Supreme Leader?", "What distance can the Fajr-3 missile travel?", "Who found the Archaemenid Empire in 550 BC?", "What is the second largest city in Arizona?", "When did the Shahis switch from Buddhist to Hindu?", "Who did the Mongol princes ask to grant them titles?", "At around what age was Dominic ordained as a Priest?"\
     ,"What is Tucson's metro area population in 2015?", "What percentage of recovery of bitumen is by oil wells?"]
vectorizer = TfidfVectorizer()

with open('docs.pickle', 'rb') as f:
    docs = pickle.load(f)
with open('fname.pickle', 'rb') as f:
    fname = pickle.load(f)
tf_idf = vectorizer.fit_transform(docs)
def readfile(file):
    with open (file, "r", encoding="utf-8") as myfile:
        data = myfile.read()
        questions = []
        answers = []
        sentences = data.split('\n')
        for sentence in sentences:
            a = sentence.split('(')
            for i in a:
                b = i.split("', ")
                if(len(b) == 2):
                    questions.append(b[0][1:-1])
                    answers.append(b[1][1:-4])

        return questions, answers
def readfile2(file):
    with open(file, "r", encoding="utf-8") as f:
        data = f.read()
        questions=[]
        sentence = data.split('\n')
        for s in sentence:
            questions.append(s)


        return questions

def get_top_3(question):
    qtf_idf = vectorizer.transform([question])
    res = cosine_similarity(tf_idf, qtf_idf)
    res = res.ravel().argsort()[-3:]
    result=[fname[i] for i in res]
    # print(result)
    return result

def query_format(questions):
    for i in range(len(questions)):
        q = questions[i].lower()
        # get_top_3t, get_top_3_t=task1.tokenize(q)
        t, filtered_t = task1.tokenize(questions[i])
        # lemma_for_top3=task1.lemmatize(get_top_3_t)
        lemma = task1.lemmatize(filtered_t)
        # newq=" ".join(lemma_for_top3)
        newq=" ".join(lemma)
        relevant_articles=get_top_3(newq)
        for j in range(len(relevant_articles)):
            begin_sub = relevant_articles[j][:8]
            end_sub = relevant_articles[j][8:]
            s="\\"
            relevant_articles[j]=begin_sub+s+ end_sub
        relevant_article = " OR ".join(relevant_articles) if len(relevant_articles)!=0 else "*"
        stemma = task1.stemmatize(filtered_t)
        tree, root = task1.dependency_tree(questions[i])
        if len(root)==0:
            root="*"
        entity, label = task1.ner(questions[i])
        syn, hyper, hypo, mero, holo = task1.synsets(task1.pos_tag(t))
        s = " OR ".join(syn) if len(syn)!=0 else "*"
        hype = " OR ".join(hyper) if len(hyper)!=0 else "*"
        hypon = " OR ".join(hypo) if len(hypo)!=0 else "*"
        stem = " OR ".join(stemma)
        mero = " OR ".join(mero) if len(mero)!=0 else "*"
        holo = " OR".join(holo) if len(holo)!=0 else "*"
        ent = " OR ".join(entity) if len(entity)!=0 else "*"
        filtered_tq=" OR ".join(filtered_t)
        lemmatized = " OR ".join(lemma)
        query=""
        # query="filename:(articles\\56.txt OR articles\\400.txt OR articles\\304.txt)^20 AND filtered:(empire OR collapse OR Alexander OR 's OR conquests)^20 AND label:(TIME OR DATE)^20 OR lemmatized:(empire,collapse,Alexander,'s,conquest)^10"
        
        if "when" in q:
            entity_type="DATE OR TIME"
            query+="filename:("+relevant_article+")^5 AND (label:("+entity_type+")^20 OR filtered:("+filtered_tq+")^20 OR lemmatized:("+lemmatized+")^10 OR synonym:("+stem+"OR"+lemmatized+")^10 OR stemmatized:("+stem+")^10) OR entity:("+ent+")^20"
            # query+="filename:("+relevant_article+")^10 AND (label:("+entity_type+")^20 OR filtered:("+filtered_tq+")^20 OR lemmatized:("+lemmatized+")^10 OR stemmatized:("+stem+")^10) OR entity:("+ent+")^20"
        elif "who" in q:
            entity_type = "PERSON OR ORG"
            # query+="filename:("+relevant_article+")^10 AND (label:("+entity_type+")^20 OR filtered:("+filtered_tq+")^20 OR lemmatized:("+lemmatized+")^10 OR stemmatized:("+stem+")^10) OR entity:("+ent+")^20"
            query+="filename:("+relevant_article+")^5 AND (label:("+entity_type+")^20 OR filtered:("+filtered_tq+")^20 OR lemmatized:("+lemmatized+")^10 OR synonym:("+stem+"OR"+lemmatized+")^10 OR stemmatized:("+stem+")^10) OR entity:("+ent+")^20"
        else: 
            # query+="filename:("+relevant_article+")^10 AND (filtered:("+filtered_tq+")^20 OR lemmatized:("+lemmatized+")^10 OR stemmatized:("+stem+")^10) OR entity:("+ent+")^20"
            query+="filename:("+relevant_article+")^5 AND (filtered:("+filtered_tq+")^20 OR lemmatized:("+lemmatized+")^10 OR stemmatized:("+stem+")^10 OR synonym:("+stem+"OR"+lemmatized+")^10) OR entity:("+ent+")^10"
        
        result = solr.search(q=query, **{'q.op': 'AND', 'rows': 10, 'fl':'*, score', 'sort': 'score desc'})
        if "when" in q:
            answerWhen(result, filtered_t, entity, syn, lemma, root, questions[i], stemma, hyper, hypo)
        elif "who" in q:
            answerWho(result, filtered_t, entity, syn, lemma, root, questions[i], stemma, hyper, hypo)
        elif "what" in q:
            answerWhat(result, filtered_t, entity, syn, lemma, root, questions[i], stemma, hyper, hypo)
        else:
            print("The question type cannot be identified.")
    # return result


def query_format_demo(question):
    q = question.lower()
    # get_top_3t, get_top_3_t=task1.tokenize(q)
    t, filtered_t = task1.tokenize(question)
    # lemma_for_top3=task1.lemmatize(get_top_3_t)
    lemma = task1.lemmatize(filtered_t)
    # newq=" ".join(lemma_for_top3)
    newq=" ".join(lemma)
    relevant_articles=get_top_3(newq)
    for j in range(len(relevant_articles)):
        begin_sub = relevant_articles[j][:8]
        end_sub = relevant_articles[j][8:]
        s="\\"
        relevant_articles[j]=begin_sub+s+ end_sub
    relevant_article = " OR ".join(relevant_articles) if len(relevant_articles)!=0 else "*"
    stemma = task1.stemmatize(filtered_t)
    tree, root = task1.dependency_tree(question)
    if len(root)==0:
        root="*"
    entity, label = task1.ner(question)
    syn, hyper, hypo, mero, holo = task1.synsets(task1.pos_tag(t))
    s = " OR ".join(syn) if len(syn)!=0 else "*"
    hype = " OR ".join(hyper) if len(hyper)!=0 else "*"
    hypon = " OR ".join(hypo) if len(hypo)!=0 else "*"
    stem = " OR ".join(stemma)
    mero = " OR ".join(mero) if len(mero)!=0 else "*"
    holo = " OR".join(holo) if len(holo)!=0 else "*"
    ent = " OR ".join(entity) if len(entity)!=0 else "*"
    filtered_tq=" OR ".join(filtered_t)
    lemmatized = " OR ".join(lemma)
    query=""
        # query="filename:(articles\\56.txt OR articles\\400.txt OR articles\\304.txt)^20 AND filtered:(empire OR collapse OR Alexander OR 's OR conquests)^20 AND label:(TIME OR DATE)^20 OR lemmatized:(empire,collapse,Alexander,'s,conquest)^10"
        
    if "when" in q:
        entity_type="DATE OR TIME"
        # query+="filename:("+relevant_article+")^5 AND (filtered:("+filtered_tq+")^20 OR lemmatized:("+lemmatized+")^10 OR stemmatized:("+stem+")^10)"
        # query+="filename:("+relevant_article+")^5 AND (label:("+entity_type+")^20 OR filtered:("+filtered_tq+")^20 OR lemmatized:("+lemmatized+")^10 OR stemmatized:("+stem+")^10) OR entity:("+ent+")^20"
        query+="filename:("+relevant_article+")^5 AND (label:("+entity_type+")^20 OR filtered:("+filtered_tq+")^20 OR lemmatized:("+lemmatized+")^10 OR synonym:("+stem+"OR"+lemmatized+")^10 OR stemmatized:("+stem+")^10) OR entity:("+ent+")^20"
    elif "who" in q:
        entity_type = "PERSON OR ORG"
        # query+="filename:("+relevant_article+")^5 AND (filtered:("+filtered_tq+")^20 OR lemmatized:("+lemmatized+")^10 OR stemmatized:("+stem+")^10)"
        # query+="filename:("+relevant_article+")^5 AND (label:("+entity_type+")^20 OR filtered:("+filtered_tq+")^20 OR lemmatized:("+lemmatized+")^10 OR stemmatized:("+stem+")^10) OR entity:("+ent+")^20"
        query+="filename:("+relevant_article+")^5 AND (label:("+entity_type+")^20 OR filtered:("+filtered_tq+")^20 OR lemmatized:("+lemmatized+")^10 OR synonym:("+stem+"OR"+lemmatized+")^10 OR stemmatized:("+stem+")^10) OR entity:("+ent+")^20"
    else: 
        # query+="filename:("+relevant_article+")^5 AND (filtered:("+filtered_tq+")^20 OR lemmatized:("+lemmatized+")^10 OR stemmatized:("+stem+")^10)"
        # query+="filename:("+relevant_article+")^5 AND (filtered:("+filtered_tq+")^20 OR lemmatized:("+lemmatized+")^10 OR stemmatized:("+stem+")^10) OR entity:("+ent+")^20"
        query+="filename:("+relevant_article+")^5 AND (filtered:("+filtered_tq+")^20 OR lemmatized:("+lemmatized+")^10 OR stemmatized:("+stem+")^10 OR synonym:("+stem+"OR"+lemmatized+")^10) OR entity:("+ent+")^20"

    result = solr.search(q=query, **{'q.op': 'AND', 'rows': 10, 'fl':'*, score', 'sort': 'score desc'})
    if "when" in q:
        answerWhen(result, filtered_t, entity, syn, lemma, root, question, stemma, hyper, hypo)
    elif "who" in q:
        answerWho(result, filtered_t, entity, syn, lemma, root, question, stemma, hyper, hypo)
    elif "what" in q:
        answerWhat(result, filtered_t, entity, syn, lemma, root, question, stemma, hyper, hypo)
    else:
        print("The question type cannot be identified.")

def answerWho(ten_best, filtered_question, question_entity, question_synonym, lemma, root,question, stemma, hyper, hypo):
    max=0
    best_sentence=""
    best_doc=""
    doc = nlp(question)
    for sent in ten_best:
        cur_score = sent['score']/4
        
        for token in doc:
            if token.dep == "dobj" and token.text in sent["sentence"]:
                cur_score+=5
            if token.dep == "nsubj" and token.tag in "NNP, NN" and token.text in sent["sentence"]:
                cur_score+=5

        entity, label=task1.ner(sent["sentence"][0])
        for i in range(len(entity)):
            if label[i]=="PERSON" or label[i]=="ORG":
                cur_score+=5
                if entity[i] in question_entity:
                    cur_score+=5
            else:
                if entity[i] in question_entity:
                    cur_score+=3

        for token in sent["filtered"]:
            if token in filtered_question:
                if token not in "``''":
                    cur_score+=5
            elif token in lemma:
                if token not in "``''":
                    cur_score+=5
            elif token in stemma:
                if token not in "``''":
                    cur_score+=5
            elif token in question_synonym:
                if token not in "``''":
                    cur_score+=5
            elif token in hyper:
                if token not in "``''":
                    cur_score+=1
            elif token in hypo:
                if token not in "``''":
                    cur_score+=1


        

        # for token in sent["filtered"]:
        #     if token in filtered_question:
        #         if token not in "``''":
        #             cur_score+=5
        #     elif token in lemma:
        #         if token not in "``''":
        #             cur_score+=5
        #     elif token in stemma:
        #         if token not in "``''":
        #             cur_score+=5
        # print(sent["sentence"])
        # print(cur_score)
        if cur_score>max:
            max=cur_score
            best_sentence=sent["sentence"]
            best_doc=sent["filename"]
    if max==0:
        best_doc="N.A"
        best_sentence="N.A"
    best.append([best_doc, question, best_sentence])

def answerWhen(ten_best, filtered_question, question_entity, question_synonym, lemma, root, question, stemma, hyper, hypo):
    max=0
    best_sentence=""
    best_doc=""
    doc = nlp(question)
    for sent in ten_best:
        cur_score = sent['score']/4
        
        
        for token in doc:
            if token.dep == "dobj" and token.text in sent["sentence"]:
                cur_score+=5
            if token.dep == "nsubj" and token.tag in "NNP, NN" and token.text in sent["sentence"]:
                cur_score+=5

        entity, label=task1.ner(sent["sentence"][0])
        for i in range(len(entity)):
            if label[i]=="DATE" or label[i]=="TIME":
                cur_score+=5
                if entity[i] in question_entity:
                    cur_score+=5
            else:
                if entity[i] in question_entity:
                    cur_score+=3
        
        for token in sent["filtered"]:
            if token in filtered_question:
                if token not in "``''":
                    cur_score+=10
            elif token in lemma:
                if token not in "``''":
                    cur_score+=5
            elif token in stemma:
                if token not in "``''":
                    cur_score+=5
            elif token in question_synonym:
                if token not in "``''":
                    cur_score+=5
            elif token in hyper:
                if token not in "``''":
                    cur_score+=1
            elif token in hypo:
                if token not in "``''":
                    cur_score+=1

        # for token in sent["filtered"]:
        #     if token in filtered_question:
        #         if token not in "``''":
        #             cur_score+=10
        #     elif token in lemma:
        #         if token not in "``''":
        #             cur_score+=5
        #     elif token in stemma:
        #         if token not in "``''":
        #             cur_score+=5
        # print(sent["filename"])
        # print(sent["sentence"])
        # print(cur_score)
        if cur_score>max:
            max=cur_score
            best_sentence=sent["sentence"]
            best_doc=sent["filename"]
    if max==0:
        best_doc="N.A"
        best_sentence="N.A"
    # print(best_sentence)
    # print(best_doc)

    best.append([best_doc, question, best_sentence])

def answerWhat(ten_best, filtered_question, question_entity, question_synonym, lemma,root, question, stemma, hyper, hypo):
    max=0
    best_sentence=""
    best_doc=""
    # print(root)
    doc = nlp(question)
    for sent in ten_best:
        cur_score = sent['score']/4
        
        
        for token in doc:
            if token.dep == "dobj" and token.text in sent["sentence"]:
                cur_score+=5
            if token.dep == "nsubj" and token.tag in "NNP, NN" and token.text in sent["sentence"]:
                cur_score+=5
        
        entity, label=task1.ner(sent["sentence"][0])
        for i in range(len(entity)):
            if entity[i] in question_entity:
                cur_score+=5
            
        for token in sent["filtered"]:
            if token in filtered_question:
                if token not in "``''":
                    cur_score+=10
            elif token in lemma:
                if token not in "``''":
                    cur_score+=5
            elif token in stemma:
                if token not in "``''":
                    cur_score+=5
            elif token in question_synonym:
                if token not in "``''":
                    cur_score+=5
            elif token in hyper:
                if token not in "``''":
                    cur_score+=1
            elif token in hypo:
                if token not in "``''":
                    cur_score+=1

        # for token in sent["filtered"]:
        #     if token in filtered_question:
        #         if token not in "``''":
        #             cur_score+=10
        #     elif token in lemma:
        #         if token not in "``''":
        #             cur_score+=5
        #     elif token in stemma:
        #         if token not in "``''":
        #             cur_score+=5    
        # print(sent["filename"])
        # print(sent["sentence"])
        # print(cur_score)
        if cur_score>max:
            max=cur_score
            best_sentence=sent["sentence"]
            best_doc=sent["filename"]
    if max==0:
        best_doc="N.A"
        best_sentence="N.A"
    # print(best_sentence)
    # print(best_doc)

    best.append([best_doc,question, best_sentence])

if __name__ == '__main__':
    # questions, answers = readfile("test.txt")
    # print(len(answers))
    if len(sys.argv) !=2:
        print("Please give input text file with question as input")
    elif len(sys.argv)==2:
        file = sys.argv[1]
        questions = readfile2(file)
        query_format(questions)
    # query_format_demo("What is the goal of the textual critic?")
    # query_format_demo("Who was Marvel/Timely's first true full-time editor?")
    # print(best)
    # query_format(questions)
    # total = len(answers)
    # cur=0
    # for i in range(len(best)):
    #     if answers[i] in best[i][2][0]:
    #         cur+=1
    # print(cur)
    # print(cur/total)
    # with open('result2.csv', 'w', encoding="utf-8") as f:
    #         fieldnames = ['question', 'article','answer']
    #         writer = csv.DictWriter(f, fieldnames=fieldnames)
    #         writer.writeheader()
    #         for i in range(len(best)):
    #             writer.writerow({ 'question': best[i][1], 'article': best[i][0][0] ,'answer': best[i][2][0]})
        with open('result.csv', 'w', encoding="utf-8") as f:
            fieldnames = ['question', 'article','answer']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(best)):
                writer.writerow({'question': best[i][1], 'article': best[i][0][0] ,'answer': best[i][2][0]})
# query_format(questions[2])

# for doc in r:
#     print(doc["filename"])
#     print(doc["sentence"])

# print(symnoym_result)
