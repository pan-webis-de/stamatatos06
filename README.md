# stamatatos06
Stamatatos, E. (2006). Authorship Attribution Based on Feature Set Subspacing Ensembles, Int. Journal on Artificial Intelligence Tools, 15(5), pp. 823-838, World Scientific.


to run  this impementation of Stamatatos Feature Set Subspacing idea you need to have python 2.7 with the packages scipy and scikit-Learn installed 


sklearn:
http://scikit-learn.org/stable/

scipy:
http://www.scipy.org/



to get the best results the trainigset should be balanced for each aurthor. 

to exicute the script you need to go in folder with the script and enter the following line in the terminal 

"python <path>/stamatatos06_Subspacing_main.py <corpora-path> <output-path>" 
optional you can change the n_max_feature_number and the m_subspace_width over this command. 

to do so enter 
"python <path>/stamatatos06_Subspacing_main.py <corpora-path> <output-path> <n_max_feature_number> <m_subspace_width>" 

on the TIRA server it is for example " python ./run_Stamatatos06/stamatatos06_Subspacing_main.py ./C10 ./run_Stamatatos06/test"
where run_Stamatatos06 is the folder that includes the main script and the jsonhandler file. 