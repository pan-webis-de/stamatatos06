
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from numpy.random import randint
from sklearn.lda import LDA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from numpy import power, float64
from numpy import argmax
from numpy import ndarray

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

    count_vectorizer = CountVectorizer( max_features=n_max_feature_number , encoding=text_encoding,token_pattern= '(?u)\\b\\w+\\b' )
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
    while k < int(2*n_max_feature_number/m_subspace_width):
        k = k+1
        #get new vectorizer trained on the features in the special subspace
        singel_count_vectorizer = CountVectorizer( max_features=n_max_feature_number , encoding=text_encoding)

        wordSubspace =[]#wordSubspace is a list with the words for the new clasifier
        used_features = []
        i = 0 # iterator
        while i < m_subspace_width:

            a_random_int = randint(n_max_feature_number)
            while a_random_int in used_features:
                a_random_int = randint(n_max_feature_number)
            wordSubspace.append(wordlist[a_random_int])
            used_features.append(a_random_int)
            i=i+1

        #train the vectorizer on the new subspace
        singel_count_vectorizer= CountVectorizer( max_features=m_subspace_width , encoding=text_encoding, vocabulary=wordSubspace).fit(wordSubspace)
        #build the vectors from the dataset with the traind vectroizer
        vectored_data = singel_count_vectorizer.transform(dataset.data)
        #normize it
        X_train_tfidf =tfid_transformer.fit_transform(vectored_data)
        #train the Classifier with the normed vector set
        clf = LDA(solver='svd').fit( X_train_tfidf.toarray(), trainSet.target)
        #add the classifier an the vectorizer to the list
        classifierList.append(clf)
        vectorizerList.append(singel_count_vectorizer)

    #return a struct with the lists
    return Bunch( classifier= classifierList, vectorizer= vectorizerList) ;
def exhaustiv_disjoint_subspacing(dataset,  n_max_feature_number, m_subspace_width , text_encoding):
    ''' Select m randomly chosen features from the data set and erase them from it. so you use ever feature only once
    '''
    ########## create feature list
    count_vectorizer = CountVectorizer( max_features=n_max_feature_number , encoding=text_encoding ,token_pattern= '(?u)\\b\\w+\\b')
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
    singel_count_vectorizer = CountVectorizer( max_features=m_subspace_width , encoding=text_encoding)
      
    tfid_transformer= TfidfTransformer( use_idf=False)
    k =0
    while k < int(n_max_feature_number/m_subspace_width):
        k = k+1
        #get new vectorizer trained on the features in the special subspace
        #singel_count_vectorizer = CountVectorizer( max_features=n_max_feature_number , encoding=text_encoding)
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
        singel_count_vectorizer= CountVectorizer( max_features=m_subspace_width , encoding=text_encoding, vocabulary=wordSubspace).fit(wordSubspace)
        #build the vectors from the dataset with the traind vectroizer
        vectored_data = singel_count_vectorizer.transform(dataset.data)
    
        
        #normize it
        X_train_tfidf =tfid_transformer.fit_transform(vectored_data)

        #train the Classifier with the normed vector set
        clf = LDA(solver='svd' ).fit( X_train_tfidf.toarray(), trainSet.target)
        
        #add the classifier an the vectorizer to the list
        classifierList.append(clf)
        vectorizerList.append(singel_count_vectorizer)
        #return a struct with the lists
    return Bunch( classifier= classifierList, vectorizer= vectorizerList)

def getting_mean(n_max_feature_number, m_subspace_width, classifier_bunch, testSet_data):
    
    k= int (n_max_feature_number/m_subspace_width)
    #while t in range(len(testSet_data)):
       
    i=0
    test_vector = classifier_bunch.vectorizer[i].transform(testSet_data)
        
    mean =  classifier_bunch.classifier[i].predict_proba(test_vector)
    #print i , mean
    i +=1
    while i < k:
        test_vector = classifier_bunch.vectorizer[i].transform(testSet_data)
        mean = mean + classifier_bunch.classifier[i].predict_proba(test_vector)
        print i , mean
        i += 1

    return (mean * (1./k))

def getting_product(n_max_feature_number, m_subspace_width, classifier_bunch, testSet_data):
    '''
    getting the mean of the posterior probailities see formula 3 in the paper
    '''
   
    k= int (n_max_feature_number/m_subspace_width)
    #while t in range(len(testSet_data)):
       
    i=0
    test_vector = classifier_bunch.vectorizer[i].transform(testSet_data)
        
    product =  classifier_bunch.classifier[i].predict_proba(test_vector)*10
    #print i , product
    i +=1
    while i < k:
        test_vector = classifier_bunch.vectorizer[i].transform(testSet_data)
        product = product * classifier_bunch.classifier[i].predict_proba(test_vector)*10
        #print i , product
        i += 1

    #productList.append( power(product, (1/k)));
    
    
    return (power(product, 1./k) * power(1./10 , 1./k))
def mp(n_max_feature_number, m_subspace_width, classifier_bunch, testSet_data):
    '''Average of Mean and Prduct  '''
    mean = getting_mean(n_max_feature_number, m_subspace_width, classifier_bunch, testSet_data)
   
    product= getting_product(n_max_feature_number, m_subspace_width, classifier_bunch, testSet_data)
   
    return (mean + product)/2
def predict_class( n_max_feature_number, m_subspace_width, trainingset, test_text, text_encoding,  mode = exhaustiv_disjoint_subspacing ):
    # use k_randome_classifier or exhaustiv_disjoint_subspacing for mode  exhaustiv_disjoint_subspacing
    #works only for one text at the time
    classifier_set = mode(trainingset,  n_max_feature_number , m_subspace_width, text_encoding)
    predict_result = mp(n_max_feature_number, m_subspace_width, classifier_set, test_text)
    return trainingset.target_names[argmax(predict_result)]
def perdict_with_trainset( n_max_feature_number, m_subspace_width, classifier_set,trainingset, test_text, text_encoding ):
    #use k_randome_classifier or exhaustiv_disjoint_subspacing for mode  exhaustiv_disjoint_subspacing
    #works only for one text at the time
    predict_result = mp(n_max_feature_number, m_subspace_width, classifier_set, test_text)
    print predict_result
    return trainingset.target_names[argmax(predict_result)]
####################Settings##################
n = 1000
m = 2
encoding_setting = 'ISO 8859-7'#'utf-8'

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

#trainSet = load_files('Vima-Authors/vima GB/training corpus' )
#testSet = load_files('Vima-Authors/vima GB/test corpus', categories=None )
trainSet = load_files('NEW CORPORA/C10', categories= ['candidate00001', 'candidate00002', 'candidate00003', 'candidate00004', 'candidate00005', 'candidate00006', 'candidate00007', 'candidate00008', 'candidate00009', 'candidate00010']  )
testSet = load_files('NEW CORPORA/C10', categories=['unknown'] )

classifier_set = exhaustiv_disjoint_subspacing(trainSet,  n , m, encoding_setting)



print '#########'


product = mp(n, m, classifier_set, testSet.data)

print ' normal',product

ergebnis = argmax(product ,1)
score =0
f=0
while f < len(ergebnis):
    if (ergebnis[f] == testSet.target[f]):
        score +=1
    else: 
        print testSet.target_names[ergebnis[f]] , testSet.target_names[testSet.target[f]]
    print 'f', f+1, 'score', score
    f +=1

percent = (float(score)/(f))*100

print ' percentage' , percent

print 'erg' ,ergebnis
print 'target ' , testSet.target
#print 'author 3 ', testSet.target_names[3]
### percentage #### 

'''
    print " vectorizer" 
    helper= classifier_set.vectorizer[t].transform(test_set)
    print helper
    
'''

