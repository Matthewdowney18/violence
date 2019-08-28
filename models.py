import numpy as np

from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import  SVC, LinearSVC , NuSVC
from sklearn.neural_network import MLPClassifier

from bert_embedding import BertEmbedding
import mxnet as mx
#from sklearn.svm.sparse import SVC
from sklearn.linear_model import LogisticRegression

def scale(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

def get_tf_idf_features(X_train, X_test):
    # initialize tf-idf vecotrizer
    Tfidf_vect = TfidfVectorizer(analyzer='word', 
                             min_df = 0, 
                             stop_words = 'english', 
                             sublinear_tf=True,
                             max_features = 5000)
    Tfidf_vect.fit(X_train)
    X_train_features = Tfidf_vect.transform(X_train)
    X_test_features = Tfidf_vect.transform(X_test)
               
    return X_train_features, X_test_features

def get_BERT_features(X_train, X_test):
    ctx = mx.gpu(0)
    # initialize tf-idf vecotrizer
    bert_embedding = BertEmbedding()
    X_train_bert = bert_embedding(X_train)
    print('almost there!')
    X_test_bert = bert_embedding(X_test)
    print('done!')
 
    X_train_features = np.array([np.mean(ex[1], axis=0) for ex in X_train_bert])
    X_test_features = np.array([np.mean(ex[1], axis=0) for ex in X_test_bert])
               
    scale(X_train_features, X_test_features)
               
    return X_train_features, X_test_features

def LR(X_train_features, y_train, X_test_features):
    log_classifier = LogisticRegression(solver='liblinear',
                                        max_iter=200)
    log_classifier.fit(X_train_features,y_train)
    y_pred_log = log_classifier.predict(X_test_features)
    return y_pred_log


def svc_sigmoid(X_train_features, y_train, X_test_features):
    svc_classifer = SVC(kernel='sigmoid', 
                    gamma='scale',
                    coef0=0,
                    C=1,
                    tol=.01,
                    random_state=None,
                    max_iter=300)
    svc_classifer.fit(X_train_features,y_train)
    y_pred_log = svc_classifer.predict(X_test_features)
    return y_pred_log


def svc_rbf(X_train_features, y_train, X_test_features):
    svc_classifer = SVC(kernel='rbf', 
                    gamma='scale',
                    C=1,
                    tol=.01,
                    random_state=None,
                    max_iter=300)
    svc_classifer.fit(X_train_features,y_train)
    y_pred_log = svc_classifer.predict(X_test_features)
    return y_pred_log


def svc_linear(X_train_features, y_train, X_test_features):
    svc_classifer = SVC(kernel='linear', 
                    gamma='scale',
                    C=1,
                    tol=.01,
                    random_state=None,
                    max_iter=300)
    svc_classifer.fit(X_train_features,y_train)
    y_pred_log = svc_classifer.predict(X_test_features)
    return y_pred_log


def NN(X_train_features, y_train, X_test_features):
    MLP_classifer = MLPClassifier(hidden_layer_sizes=(500, 500, 100),
                                  activation='relu', 
                                  solver='adam', 
                                  alpha=0.0001, 
                                  batch_size='auto', 
                                  learning_rate='constant', 
                                  learning_rate_init=0.001, 
                                  power_t=0.5, 
                                  max_iter=200, 
                                  shuffle=True, 
                                  random_state=None,
                                  tol=0.0001, 
                                  verbose=False, 
                                  warm_start=False, 
                                  momentum=0.9, 
                                  nesterovs_momentum=True, 
                                  early_stopping=False, 
                                  validation_fraction=0.1,
                                  beta_1=0.9, 
                                  beta_2=0.999, 
                                  epsilon=1e-08,
                                  n_iter_no_change=10)
    MLP_classifer.fit(X_train_features,y_train)
    y_pred_log = MLP_classifer.predict(X_test_features)
    return y_pred_log
