# stamatatos06
Stamatatos, E. (2006). Authorship Attribution Based on Feature Set Subspacing Ensembles, Int. Journal on Artificial Intelligence Tools, 15(5), pp. 823-838, World Scientific.


to run  this impementation of Stamatatos Feature Set Subspacing idea you need to have python 2.7 with the packages scipy and scikit-Learn installed 


sklearn:
http://scikit-learn.org/stable/



to get the best results the trainigset should be balanced for each aurthor. 

to exicute the script you need to go in folder with the script and enter the following line in the terminal 

"python stamatatos06_Subspacing_main.py <corpora-path> <output-path>" 
optional you can change the n_max_feature_number and the m_subspace_width over this command. 

to do so enter 
"python stamatatos06_Subspacing_main.py <corpora-path> <output-path> <n_max_feature_number> <m_subspace_width>" 