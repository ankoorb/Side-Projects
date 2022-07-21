import os
import re
import nltk
import spacy
import gensim

from nltk.corpus import stopwords
from collections import defaultdict

from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.models import AuthorTopicModel, CoherenceModel
                
# Download stopwords
if not os.path.exists('./nltk_data'):
    nltk.download('stopwords')

class TopicModeler(object):
    
    def __init__(self, data_path, use_lemma=False):
        self.data_path = data_path
        self.extract = ["discharge diagnosis", "chief complaint", "history of present illness"]
        self.use_lemma = use_lemma
        self.used_pos = ['NOUN', 'ADJ', 'VERB', 'ADV']
        self.nlp = spacy.load('en_core_web_md')#, disable=['parser', 'ner'])
        
    def check_line(self, line):
        for heading in self.extract:
            if heading in line.lower():
                return True
        return False
        
    def load_documents(self):
        docs = [] # [<text of documents in the data>]
        doc_ids = [] # {<document id from path>: <0 based index using docs>}
        condition2doc = {} # {<chief complaint>: [document ids from which complaint is extracted]}
        
        for file_path in os.listdir(self.data_path):
            
            if file_path.endswith(".txt"):
                
                with open(os.path.join(self.data_path, file_path)) as f:

                    docs.append(f.read())
                    doc_id = int(file_path.split('.')[0])
                    doc_ids.append(doc_id)

                    # Extract medical conditions
                    f.seek(0)
                    line = " "
                    conditions = None
                    while line:
                        line = f.readline()
                        if self.check_line(line): 
                            conditions = f.readline().split(',') # \n?
                            break

                    conditions = conditions if isinstance(conditions, list) else [""]
                    for condition in conditions:
                        condition = condition.lower().strip()
                        condition = re.sub(r'[.?!\'";:,]', "", condition)
                        doc_set = condition2doc.setdefault(condition, [])
                        doc_set.append(doc_id)
                        
        doc_id_map = dict(zip(doc_ids, range(len(doc_ids))))
        for condition, ids in condition2doc.items():
            for i, doc_id in enumerate(ids):
                condition2doc[condition][i] = doc_id_map[doc_id]
                
        return docs, condition2doc, doc_id_map
    
    def load_annotations(self):
        anns_data = {} # {<id from path>: {<T>: {keys: word, start, end, info}, <R>: {keys: word, arg1, arg2}}}
        anns2factors = defaultdict(set) # {<id from path>: {<set of factors>}}

        for file_path in os.listdir(self.data_path):
            
            if file_path.endswith(".ann"):
                
                with open(os.path.join(self.data_path, file_path)) as f:

                    lines = f.readlines()
                    data = defaultdict(dict)
                    ann_id = int(file_path.split('.')[0])

                    for line in lines:
                        split_line = line.split()
                        
                        if split_line[0].startswith('T'):
                            term = split_line[0]
                            word = split_line[1]
                            
                            if word.startswith("Reason"):
                                data[term]['word'] = word
                                data[term]['start'] = int(split_line[2])
                                end = split_line[3]
                                
                                if ";" in end: # Just extract the 1st start and end
                                    end = end.split(';')[0]
                                    data[term]['end'] = int(end)
                                else:
                                    data[term]['end'] = int(end)
                                    
                                data[term]['info'] = ' '.join([item for item in split_line[4:] if not item.isdigit()])
                                
                        elif split_line[0].startswith('R'):
                            relation = split_line[0]
                            word = split_line[1]
                            
                            if word.startswith("Reason"):
                                data[relation]['word'] = word
                                data[relation]['arg1'] = split_line[2].split(':')[1]
                                data[relation]['arg2'] = split_line[3].split(':')[1]
                        else:
                            pass

                    anns_data[ann_id] = data
            
        for key in anns_data:
            for x in anns_data[key]:
                if x.startswith('T'):
                    anns2factors[key].update(set(anns_data[key][x]['info'].lower().split()))
                    
        return anns_data, anns2factors
                    
    @staticmethod
    def doc_to_words(documents):
        words = [simple_preprocess(doc) for doc in documents]
        return words
    
    @staticmethod
    def remove_stopwords(docs, stop_words):
        tokens = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in docs]
        return tokens
    
    def lemmatize(self, docs):
        lemmas = [[token.lemma_ for token in self.nlp(' '.join(doc)) if token.pos_ in self.used_pos] for doc in docs]
        return lemmas
    
    def preprocess(self, no_below=5, no_above=0.5):
        stop_words = set(stopwords.words('english'))
        docs, condition2doc, doc_id_map = self.load_documents()
        
        # Remove punctuation, whitespace, PHI
        docs = [re.sub(r'\[\*\*.+?\*\*\]|[,.\'!?]', '', doc) for doc in docs]
        docs = [re.sub(r'\s+', r' ', doc) for doc in docs]
        
        # Tokenize documents and remove stop words
        docs = self.doc_to_words(docs)
        docs = self.remove_stopwords(docs, stop_words)
        
        # Lemmatize documents
        if self.use_lemma:
            docs = self.lemmatize(docs)
        
        idx2word = Dictionary(docs)
        idx2word.filter_extremes(no_below=no_below, no_above=no_above)
        
        corpus = [idx2word.doc2bow(doc) for doc in docs]
        
        return corpus, docs, idx2word, condition2doc, doc_id_map
    
    def get_topics(self, num_topics=100, chunk_size=50, passes=10, alpha='symmetric', eta='symmetric'):
        print("\nGenerating topics ...\n")
        corpus, docs, idx2word, condition2doc, doc_id_map = self.preprocess()
        self.cache = {'corpus': corpus, 'docs': docs, 'doc_id_map': doc_id_map, 'idx2word': idx2word, 
                      'condition2doc': condition2doc}
        
        # https://radimrehurek.com/gensim/models/atmodel.html
        model = AuthorTopicModel(corpus=corpus, num_topics=num_topics, id2word=idx2word, author2doc=condition2doc, 
                                 chunksize=chunk_size, passes=passes, alpha=alpha, eta=eta, random_state=7)
        
        # https://radimrehurek.com/gensim/models/coherencemodel.html
        coherence_model = CoherenceModel(model=model, texts=docs, dictionary=idx2word, coherence='c_v')
        coherence = coherence_model.get_coherence()
        
        topics = {topic: [word[0] for word in words] for topic, words in model.show_topics(num_topics, num_words=100, formatted=False)}
        conditions = {condition: model.get_author_topics(condition) for condition in model.id2author.values()}
        conditions = {condition: topics.get(max(scores, key=lambda x: x[1])[0]) for condition, scores in conditions.items()}
            
        self.cache['conditions'] = conditions
        self.cache['coherence'] = coherence
        
        return conditions
        
    def evaluate(self, topics):
        anns_data, anns2factors = self.load_annotations()
        
        idx_to_doc_id = {idx: doc_id for doc_id, idx in self.cache['doc_id_map'].items()}
        
        count = 0
        common_factors = defaultdict(set)
        for condition in self.cache['condition2doc']:
            if condition in topics:
                doc_idx = self.cache['condition2doc'][condition]
                ann_factors = set()
                for idx in doc_idx:
                    doc_id = idx_to_doc_id[idx]
                    ann_factors.update(anns2factors[doc_id])
                topic_factors = set(topics[condition])
                common = topic_factors.intersection(ann_factors)
                if common:
                    count += 1
                    common_factors[condition].update(common)

        self.cache['common_factors'] = common_factors
        return self.cache['coherence'], count / len(topics)
                

if __name__ == '__main__':
    
    data_dir = "./data/training_20180910/"
    topic_modeler = TopicModeler(data_dir)
    
    topics = topic_modeler.get_topics()
    
    print("Model Output:\n")
    for condition in topics:
        print(f"\nCondition: '{condition}' => top 10 words: {topics[condition][:10]}")
        
    print("\n\nEvaluate Model: ")
    coherence, fraction = topic_modeler.evaluate(topics)
    
    print(f"\n\tCoherence: {coherence:.4f}")
    
    print(f"\n\tFraction of documents where topic model output matched one or more words from documents annotation: {fraction:.4f}")
    
    
    freq = {}
    for key in topic_modeler.cache['common_factors']:
        size  = len(topic_modeler.cache['common_factors'][key])
        freq[size] = freq.get(size, 0) + 1
        
    size = len(topics)
    for key in freq:
        fraction = freq[key] / size
        print(f"\n\tFraction of documents where topic model output matched {key} words from documents annotation: {fraction:.4f}")
    print('\n\n')
    
    
