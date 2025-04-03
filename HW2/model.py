import math
from collections import Counter, defaultdict
from pydoc import ModuleScanner
from typing import List

import nltk
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm


class Ngram:
    def __init__(self, config, n=2):
        self.tokenizer = ToktokTokenizer()
        self.n = n
        self.model = None
        self.config = config

    def tokenize(self, sentence):
        '''
        E.g.,
            sentence: 'Here dog.'
            tokenized sentence: ['Here', 'dog', '.']
        '''
        return self.tokenizer.tokenize(sentence)

    def get_ngram(self, corpus_tokenize: List[List[str]]):
        '''
        Compute the co-occurrence of each pair.
        '''
        # begin your code (Part 1)
        #算model
        cnt={}
        # for i in range (len(corpus_tokenize)):
        #     for j in range (len(corpus_tokenize[i])):
        #         if corpus_tokenize[i][j] in cnt:
        #             cnt[corpus_tokenize[i][j]]+=1
        #         else:
        #             cnt.setdefault(corpus_tokenize[i][j],1)
        feature={}
        self.model={}
        for i in range (len(corpus_tokenize)):
            for j in range (len(corpus_tokenize[i])-1):
                if corpus_tokenize[i][j] in cnt:
                    cnt[corpus_tokenize[i][j]]+=1
                else:
                    cnt.setdefault(corpus_tokenize[i][j],1)

                self.model.setdefault(corpus_tokenize[i][j],dict())
                if corpus_tokenize[i][j+1] in self.model[corpus_tokenize[i][j]]:
                    self.model[ corpus_tokenize[i][j] ][ corpus_tokenize[i][j+1] ]+=1
                else:
                    self.model[ corpus_tokenize[i][j] ][corpus_tokenize[i][j+1]]=1

                if (corpus_tokenize[i][j],corpus_tokenize[i][j+1]) in feature:
                    feature[(corpus_tokenize[i][j],corpus_tokenize[i][j+1])]+=1
                else:
                    val={(corpus_tokenize[i][j],corpus_tokenize[i][j+1]):1}
                    feature.update(val)
        feature=sorted(feature.items(), key=lambda feat:feat[1],reverse=True)
        for i in self.model:
            for j in self.model[i]:
                self.model[i][j]/=cnt[i]
        #print(self.model)
        return self.model,feature                
        # end your code
    
    def train(self, df):
        '''
        Train n-gram model.
        '''
        corpus = [['[CLS]'] + self.tokenize(document) for document in df['review']]     # [CLS] represents start of sequence
        
        # You may need to change the outputs, but you need to keep self.model at least.

        self.model, self.features = self.get_ngram(corpus)
        #self.model = self.get_ngram(corpus)
    def compute_perplexity(self, df_test) -> float:
        '''
        Compute the perplexity of n-gram model.
        Perplexity = 2^(-entropy)
        '''
        if self.model is None:
            raise NotImplementedError("Train your model first")

        corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]
        perplexity=0
        # begin your code (Part 2)
        for i in tqdm(range (len(corpus))):
            tmp=0
            for j in range (len(corpus[i])-1):
                if corpus[i][j] in self.model:
                    if corpus[i][j+1] in self.model[corpus[i][j]]:
                        tmp+=math.log(self.model[corpus[i][j]][corpus[i][j+1]],2)
                    else:
                        if j == 0:
                            tmp+=math.log(0.01,2)
                        else:
                            if corpus[i][j-1] in self.model:
                                if corpus[i][j+1] in self.model[corpus[i][j-1]]:
                                    tmp+=math.log(self.model[corpus[i][j-1]][corpus[i][j+1]],2)
                                else:tmp+=math.log(0.01,2)
                            else:tmp+=math.log(0.01,2)
                else:tmp+=math.log(0.01,2)
            tmp/=(len(corpus[i])-1)
            perplexity+=math.pow(2,-tmp)
        perplexity/=len(corpus)
        # end your code
        return perplexity
        
    def train_sentiment(self, df_train, df_test):
        '''
        Use the most n patterns as features for training Naive Bayes.
        It is optional to follow the hint we provided, but need to name as the same.

        Parameters:
            train_corpus_embedding: array-like of shape (n_samples_train, n_features)
            test_corpus_embedding: array-like of shape (n_samples_train, n_features)
        
        E.g.,
            Assume the features are [(I saw), (saw a), (an apple)],
            the embedding of the tokenized sentence ['[CLS]', 'I', 'saw', 'a', 'saw', 'saw', 'a', 'saw', '.'] will be
            [1, 2, 0]
            since the bi-gram of the sentence contains
            [([CLS] I), (I saw), (saw a), (a saw), (saw saw), (saw a), (a saw), (saw .)]
            The number of (I saw) is 1, the number of (saw a) is 2, and the number of (an apple) is 0.
        '''
        # begin your code (Part 3)

        # step 1. select the most feature_num patterns as features, you can adjust feature_num for better score!
        feature_num = 1000
        train = [['[CLS]'] + self.tokenize(document) for document in df_train['review']]
        test = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]
        #算feature
        #train
        #trainfeatval={}
        #for i in tqdm(range (len(train))):
            #for j in tqdm(range (len(train[i])-1)):
                #if (train[i][j],train[i][j+1]) in self.features
                    # for tup in self.features:
                    #     if tup==(train[i][j],train[i][j+1]):
                    #         if tup in trainfeatval.keys() :
                    #             trainfeatval[(train[i][j],train[i][j+1])]+=1
                    #         else:
                    #             val={(train[i][j],train[i][j+1]):1}
                    #             trainfeatval.update(val)
                    #         break
        #trainfeatval=sorted(trainfeatval.items(), key=lambda feat:feat[1],reverse=True)
        
        trainfeat=[]
        x=0
        for i in self.features:
            if x < feature_num:
                x+=1
                trainfeat.append(i[0])
            else:
                break
        # step 2. convert each sentence in both training data and testing data to embedding.
        # Note that you should name "train_corpus_embedding" and "test_corpus_embedding" for feeding the model.
        train_corpus_embedding=[0]*len(train)
        test_corpus_embedding=[0]*len(test)
        # print(len(train))
        for i in range(len(train)):
            train_corpus_embedding[i]=[0]*len(trainfeat)
        for i in range(len(test)):
            test_corpus_embedding[i]=[0]*len(trainfeat)   

        for i in tqdm(range (len(train))):
            for j in range (len(train[i])-1):
                for k in range(len(trainfeat)):
                    if (train[i][j],train[i][j+1])==trainfeat[k]:
                        train_corpus_embedding[i][k]+=1
                        break
        for i in tqdm(range (len(test))):
            for j in range (len(test[i])-1):
                for k in range(len(trainfeat)):
                    if (test[i][j],test[i][j+1])==trainfeat[k]:
                        test_corpus_embedding[i][k]+=1
                        break
        # end your code
        # feed converted embeddings to Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(train_corpus_embedding, df_train['sentiment'])
        y_predicted = nb_model.predict(test_corpus_embedding)
        precision, recall, f1, support = precision_recall_fscore_support(df_test['sentiment'], y_predicted, average='macro', zero_division=1)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")


if __name__ == '__main__':
    '''
    Here is TA's answer of part 1 for reference only.
    {'a': 0.5, 'saw': 0.25, '.': 0.25}

    Explanation:
    (saw -> a): 2
    (saw -> saw): 1
    (saw -> .): 1
    So the probability of the following word of 'saw' should be 1 normalized by 2+1+1.

    P(I | [CLS]) = 1
    P(saw | I) = 1; count(saw | I) / count(I)
    P(a | saw) = 0.5
    P(saw | a) = 1.0
    P(saw | saw) = 0.25
    P(. | saw) = 0.25
    '''

    # unit test
    test_sentence = {'review': ['I saw a saw saw a saw.']}
    model = Ngram(2)
    model.train(test_sentence)
    print(model.model['saw'])
    print("Perplexity: {}".format(model.compute_perplexity(test_sentence)))
