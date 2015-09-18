
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from numpy.random import randint
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from numpy import power
from numpy import argmax


class Bunch(dict):
    """Container object for datasets: dictionary-like object that
       exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self
def k_randome_classifier(dataset,  n_max_feature_number, m_subspace_width , text_encoding):
    ''' Description 
    '''
    ### make subspaces 
    
    count_vectorizer = CountVectorizer( max_features=n_max_feature_number , encoding=text_encoding) 
    count_vectorizer.fit(dataset.data)
    wordlist = count_vectorizer.get_feature_names()
    if len(wordlist)< n_max_feature_number:
        raise ValueError('there a not enough features in the trainingset. Please be smart and try somthing else ')
    classifierList=[]
    vectorizerList =[]
   
    tfid_transformer= TfidfTransformer( use_idf=False)
    k =0
    while k < int(n_max_feature_number/m_subspace_width):
        singel_count_vectorizer = CountVectorizer( max_features=n_max_feature_number , encoding=text_encoding) 
       
        
        k = k+1
        wordSubspace =[]
        i = 0 # iterator
        while i < m_subspace_width:
            #wordSubspace is a list with the words for the new clasifier
            a_random_int = randint(n_max_feature_number)
            wordSubspace.append(wordlist[a_random_int])
            i=i+1
        singel_count_vectorizer.fit(wordSubspace)
        vectored_data = singel_count_vectorizer.transform(dataset.data)
        X_train_tfidf =tfid_transformer.fit_transform(vectored_data)
        
        clf = LinearDiscriminantAnalysis().fit( X_train_tfidf.toarray(), trainSet.target)
        
        classifierList.append(clf)
        vectorizerList.append(singel_count_vectorizer)
    
    
    return Bunch( classifier= classifierList, vectorizer= vectorizerList) ;
def exhausiv_disjoint_subspacing(dataset,  n_max_feature_number, m_subspace_width , text_encoding):
    ''' Description 
    '''
    
    
    ### make subspaces 
    
    count_vectorizer = CountVectorizer( max_features=n_max_feature_number , encoding=text_encoding) 
    count_vectorizer.fit(dataset.data)
    wordlist = count_vectorizer.get_feature_names()
    if (len(wordlist) < n_max_feature_number):
        raise ValueError('there a not enough features in the trainingset. Please be smart and try somthing else ')
    classifierList=[]
    vectorizerList =[]
    used_features = []

    tfid_transformer= TfidfTransformer( use_idf=False)
    k =0
    while k < int(n_max_feature_number/m_subspace_width):
        singel_count_vectorizer = CountVectorizer( max_features=n_max_feature_number , encoding=text_encoding) 
       
        
        k = k+1
        wordSubspace =[]
        i = 0 # iterator
        while i < m_subspace_width:
            #wordSubspace is a list with the words for the new clasifier
            a_random_int = randint(n_max_feature_number)
            ## until the Randome numer is no in list anymore 
            while a_random_int in used_features:
                a_random_int = randint(n_max_feature_number)
                
            used_features.append(a_random_int)
            wordSubspace.append(wordlist[a_random_int])
            i=i+1
            
        singel_count_vectorizer.fit(wordSubspace)
        vectored_data = singel_count_vectorizer.transform(dataset.data)
        X_train_tfidf =tfid_transformer.fit_transform(vectored_data)
        
        clf = LinearDiscriminantAnalysis().fit( X_train_tfidf.toarray(), trainSet.target)
        
        classifierList.append(clf)
        vectorizerList.append(singel_count_vectorizer)
    return Bunch( classifier= classifierList, vectorizer= vectorizerList) 
def getting_mean(n_max_feature_number, m_subspace_width, classifier_bunch, test_text):
    k= int (n_max_feature_number/m_subspace_width)
    i=0
    mean =[(0 , 0)]
    while i < k:
        test_vector = classifier_bunch.vectorizer[i].transform(test_text)
        mean = mean + classifier_bunch.classifier[i].predict_proba(test_vector)
        i += 1 
    
    return( mean * 1/k);
def getting_product(n_max_feature_number, m_subspace_width, classifier_bunch, test_text):
    k= int (n_max_feature_number/m_subspace_width)
    i=0
    test_vector = classifier_bunch.vectorizer[i].transform(test_text)
    mean =  classifier_bunch.classifier[i].predict_proba(test_vector)
    i +=1
    while i < k:
        test_vector = classifier_bunch.vectorizer[i].transform(test_text)
        mean = mean * classifier_bunch.classifier[i].predict_proba(test_vector)
        i += 1 
    
    return power(mean, (1/k));
def mp(n_max_feature_number, m_subspace_width, classifier_bunch, test_text):
        mean = getting_mean(n_max_feature_number, m_subspace_width, classifier_bunch, test_text)
        product= getting_product(n_max_feature_number, m_subspace_width, classifier_bunch, test_text)
        return (mean + product) /2
      
def predict_class( n_max_feature_number, m_subspace_width, trainingset, test_text, text_encoding,  mode = exhausiv_disjoint_subspacing ):
    # use k_randome_classifier or exhausiv_disjoint_subspacing for mode  exhausiv_disjoint_subspacing
    #works only for one text at the time 
    classifier_set = mode(trainingset,  n_max_feature_number , m_subspace_width, text_encoding)
    predict_result = mp(n_max_feature_number, m_subspace_width, classifier_set, test_text)
    return trainingset.target_names[argmax(predict_result)] 

n = 100
m = 2
encoding_setting = 'ISO 8859-2'

categories = None

#loading
# the folder must without .DS_store
# for this use 
#    find . -name '*.DS_Store' -type f -delete

trainSet = load_files('HitchHikersQuotes'  , categories= categories)
categories = ['alt.atheism', 'comp.graphics']

#trainSet = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'),  categories=categories)

test_text = ["OK. Leave this to me. I'm British. I know how to queue."]
#k_randome_classifier(trainSet, n, m, encoding_setting)

print predict_class(n, m, trainSet, test_text, encoding_setting)