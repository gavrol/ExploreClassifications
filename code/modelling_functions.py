# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:18:15 2014

@author: ogavril

PURPOSE: modelling functions

"""

import numpy as np
from sklearn import svm,tree
from sklearn import ensemble# import RandomForestClassifier, ExtraTreesClassifier
from sklearn import naive_bayes# GaussianNB
import plot_functions
import common_functions
import statsmodels.api as sm
#from class_definitions import *

##### data prep functions ########################
def make_dataMatrix_fromDF(columns,train_df,normalizeInput=True):
    nps = np.array([])
    for c in range(len(columns)):
        if normalizeInput:
            norm_vec = common_functions.normalize(train_df[columns[c]])
        else:
            norm_vec = train_df[columns[c]]
        nps = np.append(nps,norm_vec) #train_df[columns[c]])
    nps = nps.reshape((len(columns),len(train_df.index)))
    train_data = nps.transpose()
    return train_data
    
    
def make_data_4scikit_functions(columns,train_df,test_df,target_name,normalizeInput=True):
    nps = np.array([])
 
    for c in range(len(columns)):
        if normalizeInput:
            norm_vec = common_functions.normalize(train_df[columns[c]])
        else:
            norm_vec = train_df[columns[c]]
        nps = np.append(nps,norm_vec) #train_df[columns[c]])
    nps = nps.reshape((len(columns),len(train_df.index)))
    train_data = nps.transpose()
    train_target = np.array(train_df[target_name])
#    print train_data.shape
#    print train_target.shape
    
    nps = np.array([])
    for c in range(len(columns)):
        if normalizeInput:
            norm_vec = common_functions.normalize(test_df[columns[c]])
        else:
            norm_vec = test_df[columns[c]]
        nps = np.append(nps,norm_vec) #test_df[columns[c]])
    nps = nps.reshape((len(columns),len(test_df.index)))
    test_data = nps.transpose()   
    try: 
        test_target = np.array(test_df[target_name])     
    except:
        print "!!! CAUTION:",target_name,"does not exist for test data...okay for final prediction"
        test_target = None
        
    return train_data,train_target,test_data,test_target    
    

#######  model implentations  #########

def fit_LOGLOG(train_df,yVarN,xVarN):
    log_model = sm.GLM(train_df[yVarN],train_df[xVarN],family=sm.families.NegativeBinomial(link=sm.families.links.cloglog)).fit()
    print "\nLOGLOG using",len(xVarN),"variables"
    
    signif_xVarN = xVarN
    for sig in [0.5,0.1]:
        signif_xVarN = [var for var in signif_xVarN if log_model.pvalues[var] <= sig]
        log_model = sm.Logit(train_df[yVarN],train_df[signif_xVarN]).fit()    
    print "number of significant variables in the final model",len(signif_xVarN)
    return log_model,signif_xVarN
    
def eval_LOGLOG_results(log_model,train_df,test_df,yVarN,signif_xVarN,TH=0.5,pltname=None):
    y_pred = log_model.predict()    
    predicted_values_train = np.array(np.zeros(len(train_df.index)))
    for i in xrange(len(train_df.index)):
        if y_pred[i]>=TH:
            predicted_values_train[i] = 1
    
    predicted_values_test = np.array(np.zeros(len(test_df.index)))
    y_pred_test = log_model.predict(test_df[signif_xVarN])
    for i in xrange(len(test_df.index)):
        if y_pred_test[i]>=TH:
            predicted_values_test[i] = 1

    if pltname != None:
        #plot_functions.plot_true_vs_pred_logit(train_df[yVarN],y_pred,title="True vs. Predicted LogReg",figname=pltname)
        plot_functions.hist_plot(y_pred,figname=pltname+"_TR")
        plot_functions.hist_plot(y_pred_test,figname=pltname+"_TS")

    return predicted_values_train,predicted_values_test
    

def fit_Logit(train_df,yVarN,xVarN):
    log_model = sm.Logit(train_df[yVarN],train_df[xVarN]).fit()
    print "\nLogit using",len(xVarN),"variables"
    
    signif_xVarN = xVarN
    for sig in [0.5,0.1]:
        signif_xVarN = [var for var in signif_xVarN if log_model.pvalues[var] <= sig]
        log_model = sm.Logit(train_df[yVarN],train_df[signif_xVarN]).fit()            
    print "number of significant variables in the final model",len(signif_xVarN)
    return log_model, signif_xVarN
    
def eval_Logit_results(log_model,train_df,test_df,yVarN,signif_xVarN,TH=0.5,pltname=None):

    y_pred = log_model.predict()    
    predicted_values_train = np.array(np.zeros(len(train_df.index)))
    for i in xrange(len(train_df.index)):
        if y_pred[i]>=TH:
            predicted_values_train[i] = 1
    
    predicted_values_test = np.array(np.zeros(len(test_df.index)))
    y_pred_test = log_model.predict(test_df[signif_xVarN])
#    y_pred_test = calculate_yPred_logit(log_model,test_df)
#    
#    fn = open("t_prediction_Vecs1.csv","w")    
#    for i in range(len(y_pred_test)):
#        fn.write(str(y_pred_test[i])+","+str(y_pred_test1[i])+"\n")
#    fn.close()

    for i in xrange(len(test_df.index)):
        if y_pred_test[i]>=TH:
            predicted_values_test[i] = 1

    if pltname != None:
        #plot_functions.plot_true_vs_pred_logit(train_df[yVarN],y_pred,title="True vs. Predicted LogReg",figname=pltname)
        plot_functions.hist_plot(y_pred,figname=pltname+"_TR")
        plot_functions.hist_plot(y_pred_test,figname=pltname+"_TS")

    return predicted_values_train,predicted_values_test

def calculate_yPred_logit(log_model,df):
    """evaluating based on coefficients, make sure 'const' is included in df, else you need to add it manually"""
    cdf = df.copy()
    coeff = {}
    for xN in log_model.params.keys():
        coeff[xN] = log_model.params[xN]
    if 'const' in coeff.keys():
        if 'const' not in cdf.columns:
            cdf['const'] = 1
    y_pred_lin = np.array([0 for i in xrange(len(cdf.index))])
    for col in cdf.columns:
        if col in coeff.keys():
            print col,
            t_ = np.array(cdf[col]).reshape(y_pred_lin.shape[0])
            y_pred_lin = coeff[col]*t_ + y_pred_lin
    y_pred = 1.0/(1.0+np.exp(-y_pred_lin)) #this step is the sigmoid transformation
    return y_pred

                                                              

def perform_NaiveBayes(train_data,train_target,test_data,test_target,distribution=None):
    if distribution.lower() == "bernoulli":
        mod = naive_bayes.BernoulliNB()
    elif distribution.lower() == "base":
        mod =naive_bayes.BaseNB()
    elif distribution.lower() == "multinomial":
        mod =naive_bayes.MultinomialNB()
    else:
        mod = naive_bayes.GaussianNB()
    mod.fit(train_data,train_target)
    predicted_values_train = mod.predict(train_data)
    predicted_values_test = mod.predict(test_data)
    return predicted_values_train,predicted_values_test,mod


def perform_svm(train_data,train_target,test_data,test_target,kernel,polydeg=3):
    if kernel == 'poly':
        mod = svm.SVC(C=3.0,kernel=kernel,degree=polydeg)
    else:
        mod = svm.SVC(C=10.0,kernel=kernel,gamma=0.9)
    mod.fit(train_data,train_target)
    predicted_values_train = mod.predict(train_data)
    predicted_values_test = mod.predict(test_data)
    return predicted_values_train,predicted_values_test,mod

def perform_CART(train_data,train_target,test_data,test_target,min_samples_split_,min_samples_leaf_, DT_max_features=None,DT_random_state=None):
    mod = tree.DecisionTreeClassifier(min_samples_split=min_samples_split_,min_samples_leaf=min_samples_leaf_, max_features=DT_max_features,random_state=DT_random_state,n_jobs=-1)
    #print mod
    mod.fit(train_data, train_target)
    predicted_values_train = mod.predict(train_data)
    predicted_values_test = mod.predict(test_data)
    return predicted_values_train,predicted_values_test,mod


def perform_AdaBoost(train_data,train_target,test_data,test_target,min_samples_split_,min_samples_leaf_, DT_max_features=None,DT_random_state=None,num_estimators=50):
    mod = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(compute_importances=None, criterion='gini',
                                                        min_samples_split =min_samples_split_,min_samples_leaf=min_samples_leaf_,
                                                         max_features=DT_max_features,random_state=DT_random_state,splitter='best'), 
                                                         n_estimators=num_estimators, learning_rate=1.0, algorithm='SAMME.R')

    mod.fit(train_data, train_target)
    predicted_values_train = mod.predict(train_data)
    predicted_values_test = mod.predict(test_data)
    return predicted_values_train,predicted_values_test,mod

def perform_GBT(train_data,train_target,test_data,test_target,min_samples_split_,min_samples_leaf_,max_depth=1,learning_rate =0.1,num_estimators=100,max_features ='log2'):
    """Gradient Boosting Tree"""                  
    mod = ensemble.GradientBoostingClassifier(n_estimators=num_estimators, learning_rate=learning_rate, min_samples_split=min_samples_split_,min_samples_leaf=min_samples_leaf_,
                                              max_depth=max_depth, random_state=None, max_features = max_features)
    #print mod
    mod.fit(train_data, train_target)
    predicted_values_train = mod.predict(train_data)
    predicted_values_test = mod.predict(test_data)
    return predicted_values_train,predicted_values_test,mod



def perform_RandomForest(train_data,train_target,test_data,test_target,min_samples_split_,min_samples_leaf_, num_trees=10, RF_max_depth=None, RF_max_features=None,RF_random_state=None,classifier='RandomForest'):
    if classifier.lower()== 'randomforest':    
        mod = ensemble.RandomForestClassifier(n_estimators = num_trees,max_depth=RF_max_depth, max_features=RF_max_features,random_state=RF_random_state,
                                              min_samples_split=min_samples_split_,min_samples_leaf=min_samples_leaf_,n_jobs=-1)
    else:
        mod = ensemble.ExtraTreesClassifier(n_estimators = num_trees,max_depth=RF_max_depth, max_features=RF_max_features,random_state=RF_random_state,
                                            min_samples_split =min_samples_split_,min_samples_leaf=min_samples_leaf_,n_jobs=-1)
    #print mod n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
    mod.fit(train_data, train_target)
    predicted_values_train = mod.predict(train_data)
    predicted_values_test = mod.predict(test_data)
    return predicted_values_train,predicted_values_test,mod

def randomize_prediction_v1(target_vec,success_rate):
    random_target = np.array(np.zeros(len(target_vec)))
    t_ = np.random.uniform(low=0.0,high=1.0,size=len(target_vec))
    for i in xrange(len(target_vec)):
        if t_[i] <= success_rate:
            random_target[i] = 1
    return random_target

    
def get_model_coefficients(model):
    coeff_D = {}
    xVars = [key for key in model.params.keys()]
    for xN in xVars:
        coeff_D[xN] = {'mean': float(model.params[xN])}
    for n in xrange(len(xVars)):
        coeff_D[xVars[n]]['CI_lower'] = float(model.conf_int()[0][n])
        coeff_D[xVars[n]]['CI_upper'] = float(model.conf_int()[1][n])
        
    return coeff_D
    
##### evaluate models ##############

def calculate_accuracy(target_predicted, target):
    """function to use for SVM, CART, etc. where np arrays are passed to determine who different they are"""
    if len(target_predicted) == len(target):
        diff = np.abs(target_predicted - target)
        num_correctly_predicted = len(diff) - np.count_nonzero(diff)
        accuracy = float(num_correctly_predicted)/float(len(diff))
        return accuracy
    else:
        print "\n!!! ERROR in",calculate_accuracy.__name__
        return None    

def calculate_SensSpecifPrecAccur(target_predicted,target):
    
    if len(target_predicted) == len(target):
        numTP = len(target_predicted[(target_predicted==1) & (target==1)])
        numFP = len(target_predicted[(target_predicted==1) & (target==0)])
        numTN = len(target_predicted[(target_predicted==0) & (target==0)])
        numFN = len(target_predicted[(target_predicted==0) & (target==1)])
        sensitivity = float(numTP)/float(max(numTP+numFN,1))
        specificity = float(numTN)/float(max(numTN+numFP,1))
        precision = float(numTP)/float(max(numTP+numFP,1))
        accuracy = float(numTP +numTN)/float(numTP +numTN +numFP +numFN)
        return sensitivity,specificity,precision,accuracy
    else:
        return None

def cal_TP_FP_FN_TN(target_predicted,target):
    if len(target_predicted) == len(target):
        numTP = len(target_predicted[(target_predicted==1) & (target==1)])
        numFP = len(target_predicted[(target_predicted==1) & (target==0)])
        numTN = len(target_predicted[(target_predicted==0) & (target==0)])
        numFN = len(target_predicted[(target_predicted==0) & (target==1)])
    return numTP,numFP,numFN,numTN,

 #### things to keep for a while #######   
