# CRF Model Improved

Improving upon NAACL-CLFL 2016 <br>
Title: Supervised Machine Learning for Hybrid Meter <br>
Authors: Alex Estes and Christopher Hench

Abstract: <br>
Following classical antiquity, European poetic meter was complicated by traditions negotiating between the prosodic stress of vernacular dialects and a classical system based on syllable length. Middle High German (MHG) epic poetry found a solution in a hybrid qualitative and quantitative meter. We develop a CRF model to predict the metrical values of syllables in MHG epic verse, achieving an F-score of .894 on 10-fold cross-validated development data (outperforming several baselines) and .904 on held-out testing data. The method used in this paper presents itself as a viable option for other literary traditions, and as a tool for subsequent genre or author analysis.


Data and source code for paper

Dependencies:

* sklearn (pip install scikit-learn)
* nltk (pip install nltk)
* pycrfsuite (pip install python-crfsuite)
