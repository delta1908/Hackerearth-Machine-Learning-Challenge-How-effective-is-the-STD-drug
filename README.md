# Hackerearth-Machine-Learning-Challenge-How-effective-is-the-STD-drug
Using NLP(Natural Language Processing) for predicting the effectiveness of STD drugs.

The data contains 5 important attributes: 'review_by_patient','use_case_for_drug','name_of_drug','effectiveness_rating' and 'number_of_times_prescribed'. Out of these 3 of them are in text format.

'review_by_patient','use_case_for_drug','name_of_drug' are merged together and are preprocessed using the gensim preprocessing_documents
function in python. This removes stopwords, punctuations,etc and also stems the text.

For vectorizing the text I have used TfidfVectorizer(from sklearn.feature__extraction.text) which returns a sparse 31265X20904
Now I normalized this sparse marix by using numpy function linealg and out of 32165 values I got only 8 unique values: [1.0,
 0.9999999999999999,
 1.0000000000000002,
 0.9999999999999998,
 0.9999999999999997,
 1.0000000000000004,
 0.9999999999999996,
 0.9999999999999994]


So I decided to make this text attribute categorial with 8 categories.
Now that I have all my attributes I just have to find the best model to fit the data. For this I chose
GradientBoostingRegressor(sklearn.ensemble) with the following hyperparameters:


(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                          init=None, learning_rate=0.1, loss='ls', max_depth=8,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=1200,
                          n_iter_no_change=None, presort='deprecated',
                          random_state=None, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)


'effectiveness_rating','number_of_times_prescribed' and 'new_column'(new categorial attribute extracted from the text) have repective feature_importances: [4.90134688e-01, 5.09691269e-01, 1.74043035e-04]


r2_score on the training data: 0.999999960616965


Contest Score: 94.56311


no of submissions: 22


Contest Rank: 21
