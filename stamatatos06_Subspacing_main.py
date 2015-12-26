
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from numpy.random import randint
from sklearn.lda import LDA
from sklearn.feature_extraction.text import TfidfTransformer
from numpy import power, float64
from numpy import argmax
from numpy import ndarray
import jsonhandler
import sys
import logging

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

    count_vectorizer = CountVectorizer( max_features=n_max_feature_number , encoding=text_encoding )
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
    count_vectorizer = CountVectorizer( max_features=n_max_feature_number , encoding=text_encoding )#,token_pattern= '(?u)\\b\\w+\\b'
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
        clf = LDA(solver='svd').fit( X_train_tfidf.toarray(), trainSet.target )
        
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
    i +=1
    while i < k:
        test_vector = classifier_bunch.vectorizer[i].transform(testSet_data)
        mean = mean + classifier_bunch.classifier[i].predict_proba(test_vector)
        i += 1

    return (mean * (1./k))

def getting_product(n_max_feature_number, m_subspace_width, classifier_bunch, testSet_data):
    '''
    getting the mean of the posterior probailities see formula 3 in the paper
    '''
   
    k= int (n_max_feature_number/m_subspace_width)
    #while t in range(len(testSet_data)):
       
    i=0
    j=0
    test_vector = classifier_bunch.vectorizer[i].transform(testSet_data)    
    productHelp =  classifier_bunch.classifier[i].predict_proba(test_vector)
    j += 1
    i += 1 
   
    
    product =  power(productHelp, (1./k))
    
    
    while i < k:
        j= 0
        
        
        while j< 100:
            if (j == 0):
                test_vector = classifier_bunch.vectorizer[i].transform(testSet_data)    
                productHelp =  classifier_bunch.classifier[i].predict_proba(test_vector)
                #print 'product help', productHelp
                j += 1
                i += 1 
                continue 
                #'''work here '''
                
            if (i<k):
                test_vector = classifier_bunch.vectorizer[i].transform(testSet_data)
                productHelp *=   (classifier_bunch.classifier[i].predict_proba(test_vector))
               #print productHelp
            i += 1
            j += 1
            #print i, j 
            
        
        product *= power(productHelp, (1./k))     
        #print ' i = ', i , product     
    return product
def mp(n_max_feature_number, m_subspace_width, classifier_bunch, testSet_data):
    #avaerage from the product and sum
    '''Average of Mean and Prduct  '''
    mean = getting_mean(n_max_feature_number, m_subspace_width, classifier_bunch, testSet_data)
    

    product= getting_product(n_max_feature_number, m_subspace_width, classifier_bunch, testSet_data)
    
    return ((mean + product)/2)
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
    return trainingset.target_names[argmax(predict_result)]
def getBunchOutRawTrain(texts, text_authors, authorNames):
    return Bunch( data= texts, target= text_authors, target_names= authorNames, DESCR=None)
def getBunchOutTest( test_text, file_names):
    return Bunch(data= test_text, file_names= file_names, DESCR=None)

def getProbabilities(ergebnis, resultMatrix ):
    probList =[]
    for i in range(len(ergebnis)):
        probList.append(resultMatrix[i][ergebnis[i]])
    return probList

def main(corpusdir, outputdir, n_max_feature_number=1000,m_subspace_width=2 ):
    
    jsonhandler.loadJson(corpusdir)
    jsonhandler.loadTraining()
    
    authors = jsonhandler.candidates
    tests = jsonhandler.unknowns
    
    encoding_setting= jsonhandler.encoding
    
    global trainSet,testSet
    texts = []
    text_authors =[]
    test_texts=[]
    ergebnisList=[]
    prob_list=[]
    for i in range(len(authors)):
        author = authors[i]
        for text in jsonhandler.trainings[author]:
            texts.append(jsonhandler.getTrainingText(author, text))
            text_authors.append(i)
   
    
    trainSet = getBunchOutRawTrain(texts, text_authors, authors)
    classifier_set = exhaustiv_disjoint_subspacing(trainSet,  n_max_feature_number, m_subspace_width, encoding_setting)
    for t_text in tests: 
        #run classifier for every test text 
        test_texts.append(jsonhandler.getUnknownText(t_text))
              
    proMatrx= mp(n_max_feature_number, m_subspace_width, classifier_set, test_texts)
    ergebnis = argmax(proMatrx, 1 ) 
    result_author_list= []
    for i in range(len(ergebnis)):
        result_author_list.append(authors[ergebnis[i]])
        prob_list.append(proMatrx[i][ergebnis[i]])
   
    
    jsonhandler.storeJson(outputdir,tests, result_author_list, prob_list)

#_____________READ ARGS______________

if (len(sys.argv )<3):
    logging.warning('MAaaaaaaN please youse more Arguments, atleast a path to a dataset and a path for the output')
    exit()

elif (len(sys.argv)==3):
    inputdir= sys.argv[1]
    outputdir= sys.argv[2]
    main(inputdir, outputdir)
    
elif (len(sys.argv)==5):
    inputdir= sys.argv[1]
    outputdir= sys.argv[2]
    n= int(sys.argv[3])
    m= int(sys.argv[4])
    main(inputdir, outputdir,n_max_feature_number=n, m_subspace_width=m)
else:
    logging.warning('use the rigth number of Arguments, 1st dataset, 2nd output, 3rd n, 4th m')
    exit()

