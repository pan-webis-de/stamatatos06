
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from numpy.random import randint
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from numpy import power
from numpy import argmax
'''
this script implements the methods of E. Stamatatos' paper 'AUTORSHIP ATTRIBUTION ON FEATURE SET SUBSPACING ENSEMBLES'

to use the discriped lerning algorithems I helped me with the scikit Lern libray

So to run this script you need numpy and sklern


this script is written by Timo Sommer

'''

class Bunch(dict):
    """Container object for datasets: dictionary-like object that
       exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self
def k_randome_classifier(dataset,  n_max_feature_number, m_subspace_width , text_encoding):
    ''' Select m randomly chosen features from the data set and transfrom dem into a Classifier
    so you get N/M =k diffrent classifie
    '''
    ### make subspaces

    count_vectorizer = CountVectorizer( max_features=n_max_feature_number , encoding=text_encoding)
    count_vectorizer.fit(dataset.data)
    wordlist = count_vectorizer.get_feature_names()
    #when less features in the Feature list then max_feature raise Valueerror
    if len(wordlist)< n_max_feature_number:
        raise ValueError('there a not enough features in the trainingset. Please be smart and try somthing else ')
    #list for passing the objects
    classifierList=[]
    vectorizerList =[]

    tfid_transformer= TfidfTransformer( use_idf=False)
    k =0
    while k < int(n_max_feature_number/m_subspace_width):
        k = k+1
        #get new vectorizer trained on the features in the special subspace
        singel_count_vectorizer = CountVectorizer( max_features=n_max_feature_number , encoding=text_encoding)

        wordSubspace =[]#wordSubspace is a list with the words for the new clasifier
        i = 0 # iterator
        while i < m_subspace_width:

            a_random_int = randint(n_max_feature_number)
            wordSubspace.append(wordlist[a_random_int])
            i=i+1

        #train the vectorizer on the new subspace
        singel_count_vectorizer.fit(wordSubspace)
        #build the vectors from the dataset with the traind vectroizer
        vectored_data = singel_count_vectorizer.transform(dataset.data)
        #normize it
        X_train_tfidf =tfid_transformer.fit_transform(vectored_data)
        #train the Classifier with the normed vector set
        clf = LinearDiscriminantAnalysis().fit( X_train_tfidf.toarray(), trainSet.target)
        #add the classifier an the vectorizer to the list
        classifierList.append(clf)
        vectorizerList.append(singel_count_vectorizer)

    #return a struct with the lists
    return Bunch( classifier= classifierList, vectorizer= vectorizerList) ;
def exhausiv_disjoint_subspacing(dataset,  n_max_feature_number, m_subspace_width , text_encoding):
    ''' Select m randomly chosen features from the data set and erase them from it. so you use ever feature only once
    '''
    ########## create feature list
    count_vectorizer = CountVectorizer( max_features=n_max_feature_number , encoding=text_encoding)
    count_vectorizer.fit(dataset.data)
    wordlist = count_vectorizer.get_feature_names()
    #when less features in the Feature list then max_feature raise Valueerror
    if (len(wordlist) < n_max_feature_number):
        raise ValueError('there a not enough features in the trainingset. Please be smart and try somthing else ')

    #list for passing the objects
    classifierList=[]
    vectorizerList =[]
    used_features = []
    #nomilizer for the featuresets
    tfid_transformer= TfidfTransformer( use_idf=False)
    k =0
    while k < int(n_max_feature_number/m_subspace_width):
        k = k+1
        #get new vectorizer trained on the features in the special subspace
        singel_count_vectorizer = CountVectorizer( max_features=n_max_feature_number , encoding=text_encoding)
        wordSubspace =[] #wordSubspace is a list with the words for the new clasifier
        i = 0 # iterator
        while i < m_subspace_width:
            i=i+1

            a_random_int = randint(n_max_feature_number)
            ## until the Randome numer is no in list anymore
            while a_random_int in used_features:
                a_random_int = randint(n_max_feature_number)
            #add to list
            used_features.append(a_random_int)# used feature list
            wordSubspace.append(wordlist[a_random_int])


        #train the vectorizer on the new subspace
        singel_count_vectorizer.fit(wordSubspace)
        #build the vectors from the dataset with the traind vectroizer
        vectored_data = singel_count_vectorizer.transform(dataset.data)
        #normize it
        X_train_tfidf =tfid_transformer.fit_transform(vectored_data)
        #train the Classifier with the normed vector set
        clf = LinearDiscriminantAnalysis().fit( X_train_tfidf.toarray(), trainSet.target)
        #add the classifier an the vectorizer to the list
        classifierList.append(clf)
        vectorizerList.append(singel_count_vectorizer)
        #return a struct with the lists
    return Bunch( classifier= classifierList, vectorizer= vectorizerList)

def getting_mean(n_max_feature_number, m_subspace_width, classifier_bunch, test_text):
    '''
    getting the mean of the posterior probailities see formula 2 in the paper
    '''
    k= int (n_max_feature_number/m_subspace_width)
    i=0
    mean =[(0 , 0)]
    while i < k:
        test_vector = classifier_bunch.vectorizer[i].transform(test_text)
        mean = mean + classifier_bunch.classifier[i].predict_proba(test_vector)
        i += 1

    return( mean * 1/k);
def getting_product(n_max_feature_number, m_subspace_width, classifier_bunch, test_text):
    '''
    getting the mean of the posterior probailities see formula 3 in the paper
    '''
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
    '''Average of Mean and Prduct  '''
    mean = getting_mean(n_max_feature_number, m_subspace_width, classifier_bunch, test_text)
    product= getting_product(n_max_feature_number, m_subspace_width, classifier_bunch, test_text)
    return (mean + product)/2
def predict_class( n_max_feature_number, m_subspace_width, trainingset, test_text, text_encoding,  mode = exhausiv_disjoint_subspacing ):
    # use k_randome_classifier or exhausiv_disjoint_subspacing for mode  exhausiv_disjoint_subspacing
    #works only for one text at the time
    classifier_set = mode(trainingset,  n_max_feature_number , m_subspace_width, text_encoding)
    predict_result = mp(n_max_feature_number, m_subspace_width, classifier_set, test_text)
    return trainingset.target_names[argmax(predict_result)]


####################Settings##################
n = 1000
m = 5
encoding_setting = 'ISO 8859-2'

categories = None
##############################################


############# Loading Files ###################
# Load files in the follwing structur
# with categories you can select certain subfolders to load if none everything will included
# textfilenames are not relevant
#        /folder
#           /autor1
#                text1.txt text2.txt ....
#           /autor2
#                text1.txt text2.txt ....
#    for mac osX :
#     the folder must without .DS_store
#     for mac use this command in terminal 
#        find . -name '*.DS_Store' -type f -delete

#trainSet = load_files('HitchHikersQuotes'  , categories= categories)
categories = ['alt.atheism', 'comp.graphics']
#for a big data set use  fetch_20newsgroups
trainSet = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'),  categories=categories)

test_text = ["OK. Leave this to me. I'm British. I know how to queue."]
#k_randome_classifier(trainSet, n, m, encoding_setting)



print predict_class(n, m, trainSet, test_text, encoding_setting)
