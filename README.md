# Skin-Segmentation-UCI-dataset-

This project mainly deals with the classification of skin and non-skin pixel category based on B, G, R pixel values. Skin segmentation is quite a challenging problem due to the fact that a lot depends on the lightning conditions, background etc. Hence in this project we implement different classifiers such as Random Forest, KNN, Logistic Regression, Decision Trees, LDA & QDA in order to obtain our best performing model capable to accurately predict whether the given combination of B, G, R values belong to skin or non-skin category. Additionally, we also come up with intuitive machine learning techniques such as k-fold cross validation to make our model robust to errors. 


In this project we have a total of 6 classifiers:
Logistic Regression
Nearest Neighbor Classifier and KNN
Random Forest Classifier
LDA
QDA
SVM
Decision Tree Classifier


The dataset consists of randomly sampled B, G, R pixel values from face images of various age groups (young, middle and old), race groups (white, black and Asian). The total sample size is 245057; out of which 50859 is the skin samples and 194198 is non-skin samples. Thus, the dataset is of dimension 245057*4 where the first three columns are B, G, R (x1, x2, x3 features) values while the fourth column is of class label i.e. the decision variable y. The class labels for this dataset have 2 values i.e. 1 and 2. Also the total size of the dataset i.e. the total number of elements is 980228. The maximum value for the B, G, R pixel is 255 while the minimum is zero. It is important to note that the data must be randomly shuffled later while data pre-processing steps so that our model can be robust. Also, the data must be split into corresponding training, test and validation parts based on our design considerations such that we could obtain the best performing model that is robust to both overfitting and underfitting.


1)	Since all of our 3 features i.e. the B, G, R pixel values use same scale hence normalization isn’t needed (at least for initial few classifiers which do not deal with distance between its neighbors)
2)	Data shuffling is carried out for our B, G, R row entities so as to ensure that our machine learning model is robust such that it has the data elements unevenly spread out.
3)	Since our dataset does not have any missing outliers hence, we do not have to worry about filling Nan’s. The process of filling or removing out the redundancies in our data is termed as Imputation. One other critical step during preprocessing is filling empty data entities with 0. However, this dataset does not have those redundancies hence we do not have to worry about it.
4) We utilized k-fold cross validation to ensure that we use the entire data for training and later evaluate our model for validation accuracy. This is one of the most critical steps before building an actual model. This ensures that the model is robust such that it can essentially use the entire data for training. There are various approaches to handle this such as K-fold CV, Leave One out CV and Validation set approach. We employ k-fold splits for our implementation. We used several ways for cross validation i.e. 80-20 train test split, 75-25 and 90-10. 
