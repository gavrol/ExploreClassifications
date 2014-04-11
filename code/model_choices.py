# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 14:57:28 2014

@author: ogavril
"""
import numpy as np
import datetime

import modelling_functions
import plot_functions
from class_definitions import *

from sklearn import metrics
from sklearn import grid_search,svm

def use_LOGLOG(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,LOGIT_THS):
    train_df = TRAIN_df.copy()
    test_df = TEST_df.copy()
    train_target = np.array(train_df[target_name])
    test_target = np.array(test_df[target_name])
    if 'const' not in train_df.columns:
        train_df['const'] = 1
    if 'const' not in test_df.columns:
        test_df['const'] = 1
    log_model,signif_xVarN = modelling_functions.fit_LOGLOG(train_df,target_name,xVarN)

    kernels = []
    for th in LOGIT_THS:
        LOGIT_TH = th*0.1
        kernel = "LOGLOG_TH"+str(th)
        kernels.append(kernel)

        if kernel not in TrainModel.keys():
            TrainModel[kernel] = {}
    
        predicted_values_train,predicted_values_test = modelling_functions.eval_LOGLOG_results(log_model,train_df,test_df,target_name,signif_xVarN,TH=LOGIT_TH,pltname=None)
        TrainModel[kernel][trs] = {}
        TrainModel[kernel][trs]['model'] = log_model
        TrainModel[kernel][trs]['model_variables'] = signif_xVarN
       
        tr_sensitivity,tr_specificity,tr_precision,tr_accuracy = modelling_functions.calculate_SensSpecifPrecAccur(predicted_values_train,train_target)
        #print "For the train set of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(tr_sensitivity,tr_specificity,tr_precision,tr_accuracy)
        TrainModel_Stats[trs][kernel] ={'sensitivity':tr_sensitivity,"specificity":tr_specificity,'precision':tr_precision,'accuracy': tr_accuracy}  
                      
        ts_sensitivity,ts_specificity,ts_precision,ts_accuracy = modelling_functions.calculate_SensSpecifPrecAccur(predicted_values_test,test_target)
        #print "For the test set of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(ts_sensitivity,ts_specificity,ts_precision,ts_accuracy)
        TestModel_Stats[trs][kernel] ={'sensitivity':ts_sensitivity,"specificity":ts_specificity,'precision':ts_precision,'accuracy': ts_accuracy}
        
    return kernels
 

def use_AdaBoost(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,NORMALIZE,Features,NumEstimators,min_samples_split,min_samples_leaf):
    kernels = []

    for max_features in Features: #xrange(10,250,10): 
        for num_estimator in NumEstimators:
            train_df = TRAIN_df.copy()
            test_df = TEST_df.copy()
            kernel = 'AdaBoost_mF_'+str(max_features)+"_nE_"+str(num_estimator)
            kernels.append(kernel)
            if kernel not in TrainModel.keys():
                TrainModel[kernel] = {}

            train_data,train_target,test_data,test_target = modelling_functions.make_data_4scikit_functions(xVarN,train_df,test_df,target_name,normalizeInput=NORMALIZE)

            predicted_values_train,predicted_values_test,model = modelling_functions.perform_AdaBoost(train_data,train_target,test_data,test_target,min_samples_split,min_samples_leaf,
                                                                                num_estimators=num_estimator, DT_max_features=max_features,
                                                                                   DT_random_state=None)
       
            TrainModel[kernel][trs] = {}
            TrainModel[kernel][trs]['model'] = model               
    
            tr_sensitivity,tr_specificity,tr_precision,tr_accuracy = modelling_functions.calculate_SensSpecifPrecAccur(predicted_values_train,train_target)
            #print "For the train set of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(tr_sensitivity,tr_specificity,tr_precision,tr_accuracy)
            TrainModel_Stats[trs][kernel] ={'sensitivity':tr_sensitivity,"specificity":tr_specificity,'precision':tr_precision,'accuracy': tr_accuracy}  
            TestModel_Stats[trs][kernel] ={'sensitivity':tr_sensitivity,"specificity":tr_specificity,'precision':tr_precision,'accuracy': tr_accuracy}  

            """TESTING is not needed for trees                             
            ts_sensitivity,ts_specificity,ts_precision,ts_accuracy = modelling_functions.calculate_SensSpecifPrecAccur(predicted_values_test,test_target)
            #print "For the test set of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(ts_sensitivity,ts_specificity,ts_precision,ts_accuracy)
            TestModel_Stats[trs][kernel] ={'sensitivity':ts_sensitivity,"specificity":ts_specificity,'precision':ts_precision,'accuracy': ts_accuracy}
            """ 
    return kernels



def use_Forest(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,NORMALIZE,Features,NumTrees,min_samples_split,min_samples_leaf):
    kernels = []

    for max_features in Features: #xrange(10,250,10): 
        for num_trees in NumTrees:
            train_df = TRAIN_df.copy()
            test_df = TEST_df.copy()
            kernel = 'Forest_mF_'+str(max_features)+"_nT_"+str(num_trees)
            kernels.append(kernel)
            if kernel not in TrainModel.keys():
                TrainModel[kernel] = {}

            train_data,train_target,test_data,test_target = modelling_functions.make_data_4scikit_functions(xVarN,train_df,test_df,target_name,normalizeInput=NORMALIZE)

            predicted_values_train,predicted_values_test,model = modelling_functions.perform_RandomForest(train_data,train_target,test_data,test_target,
                                                                                                        min_samples_split,min_samples_leaf, num_trees=num_trees, 
                                                                                                        RF_max_features=max_features,RF_random_state=1,classifier='RandomForest')
        
            TrainModel[kernel][trs] = {}
            TrainModel[kernel][trs]['model'] = model               
    
            tr_sensitivity,tr_specificity,tr_precision,tr_accuracy = modelling_functions.calculate_SensSpecifPrecAccur(predicted_values_train,train_target)
            #print "For the train set of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(tr_sensitivity,tr_specificity,tr_precision,tr_accuracy)
            TrainModel_Stats[trs][kernel] ={'sensitivity':tr_sensitivity,"specificity":tr_specificity,'precision':tr_precision,'accuracy': tr_accuracy}  
            TestModel_Stats[trs][kernel] ={'sensitivity':tr_sensitivity,"specificity":tr_specificity,'precision':tr_precision,'accuracy': tr_accuracy}  
 
            """TESTING is not needed for forest as cross-validation is implicit in fitting a forest                                       
            ts_sensitivity,ts_specificity,ts_precision,ts_accuracy = modelling_functions.calculate_SensSpecifPrecAccur(predicted_values_test,test_target)
            #print "For the test set of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(ts_sensitivity,ts_specificity,ts_precision,ts_accuracy)
            TestModel_Stats[trs][kernel] ={'sensitivity':ts_sensitivity,"specificity":ts_specificity,'precision':ts_precision,'accuracy': ts_accuracy}
            """
    return kernels

def use_GBT(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,NORMALIZE,NumInteractions,NumBoostingStages,Features,min_samples_split,min_samples_leaf,learning_rate_):
    kernels = []
    for max_num_inter in NumInteractions: #xrange(10,350,20):
        for num_estimator in NumBoostingStages: #xrange(50,250,50)
            for feature in Features:
                train_df = TRAIN_df.copy()
                test_df = TEST_df.copy()
                kernel = 'GBT_mI_'+str(max_num_inter)+"_mE_"+str(num_estimator)+"_F_"+str(feature)
                kernels.append(kernel)
                if kernel not in TrainModel.keys():
                    TrainModel[kernel] = {}
    
                train_data,train_target,test_data,test_target = modelling_functions.make_data_4scikit_functions(xVarN,train_df,test_df,target_name,normalizeInput=NORMALIZE)
                
                predicted_values_train,predicted_values_test,model = modelling_functions.perform_GBT(train_data,train_target,test_data,test_target,min_samples_split,min_samples_leaf,
                                                                                                     max_depth=max_num_inter,learning_rate =learning_rate_,
                                                                                                     num_estimators=num_estimator,max_features =feature)
                TrainModel[kernel][trs] = {}
                TrainModel[kernel][trs]['model'] = model
        
        
                tr_sensitivity,tr_specificity,tr_precision,tr_accuracy = modelling_functions.calculate_SensSpecifPrecAccur(predicted_values_train,train_target)
                #print "For the train set of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(tr_sensitivity,tr_specificity,tr_precision,tr_accuracy)
                TrainModel_Stats[trs][kernel] ={'sensitivity':tr_sensitivity,"specificity":tr_specificity,'precision':tr_precision,'accuracy': tr_accuracy}  
                TestModel_Stats[trs][kernel] ={'sensitivity':tr_sensitivity,"specificity":tr_specificity,'precision':tr_precision,'accuracy': tr_accuracy}  
                              
                """TESTING is not needed for GBT as cross-validation is implicit in fitting a GBT                                       
                ts_sensitivity,ts_specificity,ts_precision,ts_accuracy = modelling_functions.calculate_SensSpecifPrecAccur(predicted_values_test,test_target)
                #print "For the test set of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(ts_sensitivity,ts_specificity,ts_precision,ts_accuracy)
                TestModel_Stats[trs][kernel] ={'sensitivity':ts_sensitivity,"specificity":ts_specificity,'precision':ts_precision,'accuracy': ts_accuracy}
                """

    return kernels
    

def use_SVM(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,NORMALIZE,kernels,poly_degrees):
    train_df = TRAIN_df.copy()
    test_df = TEST_df.copy()
    train_data,train_target,test_data,test_target = modelling_functions.make_data_4scikit_functions(xVarN,train_df,test_df,target_name,normalizeInput=NORMALIZE)

    for kernel in kernels:
        if kernel == 'rbf':
            parameters = {'kernel': [kernel], 'C': [1, 10,100] , 'gamma': [1e-3, 1e-1, 1.0]} #'C': [1, 10,100]  'gamma': [1e-3, 1e-1, 1.0],
        elif kernel == 'linear':
            parameters = {'kernel': [kernel], 'C': [1, 10,100]  } #'C':[1, 10, 100]
        elif kernel in [ 'poly','sigmoid']:
            parameters = {'kernel': [kernel], 'degree':poly_degrees, 'C': [10], 'gamma': [1e-3, 1e-1, 1.0]}
        else:
            print "!!! unknown SVM kernel"
            return None
            
        kernel = "SVM_"+kernel
        print kernel
        if kernel not in TrainModel.keys():
            TrainModel[kernel] = {}
        classifier = svm.SVC()
        mod = grid_search.GridSearchCV(classifier, parameters) #, scoring=metrics.f1_score) #score_func)
        mod.fit(train_data, train_target, cv=3)
        predicted_values_train = mod.predict(train_data)
        print "Best parameters set found on development set: \n", mod.best_estimator_
        predicted_values_test = mod.predict(test_data)
#        print metrics.classification_report(test_target, predicted_values_test)

        TrainModel[kernel][trs] = {}
        TrainModel[kernel][trs]['model'] = mod
    
    
        tr_sensitivity,tr_specificity,tr_precision,tr_accuracy = modelling_functions.calculate_SensSpecifPrecAccur(predicted_values_train,train_target)
        #print "For the train set of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(tr_sensitivity,tr_specificity,tr_precision,tr_accuracy)
        TrainModel_Stats[trs][kernel] ={'sensitivity':tr_sensitivity,"specificity":tr_specificity,'precision':tr_precision,'accuracy': tr_accuracy}  
                      
        ts_sensitivity,ts_specificity,ts_precision,ts_accuracy = modelling_functions.calculate_SensSpecifPrecAccur(predicted_values_test,test_target)
        #print "For the test set of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(ts_sensitivity,ts_specificity,ts_precision,ts_accuracy)
        TestModel_Stats[trs][kernel] ={'sensitivity':ts_sensitivity,"specificity":ts_specificity,'precision':ts_precision,'accuracy': ts_accuracy}

    return kernels

       
def use_SVM_old(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,NORMALIZE,kernels,poly_degree):
   
    for model_name in kernels:
        train_df = TRAIN_df.copy()
        test_df = TEST_df.copy()

        kernel = "SVM_"+model_name
        if kernel not in TrainModel.keys():
            TrainModel[kernel] = {}

        train_data,train_target,test_data,test_target = modelling_functions.make_data_4scikit_functions(xVarN,train_df,test_df,target_name,normalizeInput=NORMALIZE)
        
        predicted_values_train,predicted_values_test,model = modelling_functions.perform_svm(train_data,train_target,test_data,test_target,model_name)        
        TrainModel[kernel][trs] = {}
        TrainModel[kernel][trs]['model'] = model


        tr_sensitivity,tr_specificity,tr_precision,tr_accuracy = modelling_functions.calculate_SensSpecifPrecAccur(predicted_values_train,train_target)
        #print "For the train set of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(tr_sensitivity,tr_specificity,tr_precision,tr_accuracy)
        TrainModel_Stats[trs][kernel] ={'sensitivity':tr_sensitivity,"specificity":tr_specificity,'precision':tr_precision,'accuracy': tr_accuracy}  
                      
        ts_sensitivity,ts_specificity,ts_precision,ts_accuracy = modelling_functions.calculate_SensSpecifPrecAccur(predicted_values_test,test_target)
        #print "For the test set of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(ts_sensitivity,ts_specificity,ts_precision,ts_accuracy)
        TestModel_Stats[trs][kernel] ={'sensitivity':ts_sensitivity,"specificity":ts_specificity,'precision':ts_precision,'accuracy': ts_accuracy}

    return kernels

def use_NaiveBayes(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,NORMALIZE,kernels):

    for model_name in kernels:
        train_df = TRAIN_df.copy()
        test_df = TEST_df.copy()
        
        kernel = "NaiveBayes_"+model_name
        if kernel not in TrainModel.keys():
            TrainModel[kernel] = {}

        train_data,train_target,test_data,test_target = modelling_functions.make_data_4scikit_functions(xVarN,train_df,test_df,target_name,normalizeInput=NORMALIZE) 
        
        predicted_values_train,predicted_values_test,model = modelling_functions.perform_NaiveBayes(train_data,train_target,test_data,test_target,model_name)
        TrainModel[kernel][trs] = {}
        TrainModel[kernel][trs]['model'] = model


        tr_sensitivity,tr_specificity,tr_precision,tr_accuracy = modelling_functions.calculate_SensSpecifPrecAccur(predicted_values_train,train_target)
        #print "For the train set of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(tr_sensitivity,tr_specificity,tr_precision,tr_accuracy)
        TrainModel_Stats[trs][kernel] ={'sensitivity':tr_sensitivity,"specificity":tr_specificity,'precision':tr_precision,'accuracy': tr_accuracy}  
                      
        ts_sensitivity,ts_specificity,ts_precision,ts_accuracy = modelling_functions.calculate_SensSpecifPrecAccur(predicted_values_test,test_target)
        #print "For the test set of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(ts_sensitivity,ts_specificity,ts_precision,ts_accuracy)
        TestModel_Stats[trs][kernel] ={'sensitivity':ts_sensitivity,"specificity":ts_specificity,'precision':ts_precision,'accuracy': ts_accuracy}

    return kernels

    
def use_Logit(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,LOGIT_THS,model_coeff=None):
    train_df = TRAIN_df.copy()
    test_df = TEST_df.copy()
    train_target = np.array(train_df[target_name])
    test_target = np.array(test_df[target_name])
    if 'const' not in train_df.columns:
        train_df['const'] = 1
    if 'const' not in test_df.columns:
        test_df['const'] = 1

    log_model,signif_xVarN = modelling_functions.fit_Logit(train_df,target_name,xVarN)

    kernels = []
    for th in LOGIT_THS:
        LOGIT_TH = th*0.1        
        kernel = "Logit_TH"+str(th)
        kernels.append(kernel)
            
        if kernel not in TrainModel.keys():
            TrainModel[kernel] = {}
    
        predicted_values_train,predicted_values_test = modelling_functions.eval_Logit_results(log_model,train_df,test_df,target_name,signif_xVarN,TH=LOGIT_TH,pltname=None)
        TrainModel[kernel][trs] = {}
        TrainModel[kernel][trs]['model'] = log_model
        TrainModel[kernel][trs]['model_variables'] = signif_xVarN

       
        tr_sensitivity,tr_specificity,tr_precision,tr_accuracy = modelling_functions.calculate_SensSpecifPrecAccur(predicted_values_train,train_target)
        #print "For the train set of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(tr_sensitivity,tr_specificity,tr_precision,tr_accuracy)
        TrainModel_Stats[trs][kernel] ={'sensitivity':tr_sensitivity,"specificity":tr_specificity,'precision':tr_precision,'accuracy': tr_accuracy}  
         
#        print "precision_score=",metrics.precision_score(train_target,predicted_values_train)  
#        print "recall/sensitivity=",metrics.recall_score(train_target,predicted_values_train)  
#        print metrics.classification_report(train_target,predicted_values_train)    
#        print "TP = %d, FP = %d \nFN = %d, TN = %d" %modelling_functions.cal_TP_FP_FN_TN(predicted_values_train,train_target)
        
        ts_sensitivity,ts_specificity,ts_precision,ts_accuracy = modelling_functions.calculate_SensSpecifPrecAccur(predicted_values_test,test_target)
        #print "For the test set of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(ts_sensitivity,ts_specificity,ts_precision,ts_accuracy)
        TestModel_Stats[trs][kernel] ={'sensitivity':ts_sensitivity,"specificity":ts_specificity,'precision':ts_precision,'accuracy': ts_accuracy}
        #print metrics.classification_report(test_target,predicted_values_test)         
       
    return kernels

       
####### model summaries ###########
       

 
def coefficient_tuning_via_CrossValidation(TrainModel,Model_Stats,MODEL_NAME):
    """
    This function is only for those model type for which scikit doesn't offer a cross-validation GridSearchCV,
    e.g., functions from statsmodels.
    This function is to be run AFTER summarize_model_performance() when models-vector (for one type of model_name, say Logit) 
    has been built (i.e., summarize_model_performance())"""
    print "\n Tuning coefficients in cross validation"
    
    kernels = []
    for trs in Model_Stats.keys():
        for kernel in  Model_Stats[trs].keys():
            if kernel not in kernels:
                kernels.append(kernel)
   
    coeff = {}
    variables = {}
    for kernel in kernels:
        coeff[kernel] = {}
        for trs in TrainModel[kernel].keys():
            model = TrainModel[kernel][trs]['model']
            coeff[kernel][trs] = modelling_functions.get_model_coefficients(model)
            for var in coeff[kernel][trs].keys():
                if var not in variables.keys():
                    variables[var] = {}

    fn = open("t_"+MODEL_NAME+"_summarized_coef.csv",'w')
    
    for var in variables.keys():
        mean = 0
        count = 0
        CI_lower = 1e10
        CI_upper = -1e10
        for model_name in coeff.keys():
            for trs in coeff[model_name].keys():
                if var in coeff[model_name][trs].keys():
                    mean += coeff[model_name][trs][var]['mean']
                    count += 1
                    CI_lower = min(CI_lower,coeff[model_name][trs][var]['CI_lower'])
                    CI_upper = max(CI_upper,coeff[model_name][trs][var]['CI_upper'])

        mean = mean/float(max(1,count))
        variables[var]['mean'] = mean
        variables[var]['CI_lower'] = CI_lower
        variables[var]['CI_upper'] = CI_upper

        if CI_lower <= 0 and CI_upper >= 0:
            #print var, "includes 0",CI_lower,'and', CI_upper
            variables[var]['mean'] = 0.0
            
        fn.write(var+","+str(mean)+","+str(CI_lower)+","+str(CI_upper)+"\n")                    
    fn.close()
    return variables

        
def summarize_model_performance(Model_Stats,stats,logF=None):
    """because each model is run on a few validaton/test sets, 
    the function averages the stats for each model based on which test/validation set was used;
    later one should run determine_best_model()  
    There are model types for which there is only one train set, e.g., Forest, but that's not important
    """
    train_sets = Model_Stats.keys()
    kernels = []
    for trs in Model_Stats.keys():
        for kernel in  Model_Stats[trs].keys():
            if kernel not in kernels:
                kernels.append(kernel)
    models = []
        
    for kernel in kernels:
        model = DISCRETE_MODEL(kernel)
        for stat in stats:
            t,count = 0,0
            for trs in train_sets:
                if kernel in Model_Stats[trs].keys():
                    t += Model_Stats[trs][kernel][stat]
                    count += 1
            model.__dict__[stat] = t/float(count)
        models.append(model)
        
#    if logF != None:
#        for model in models:
#            logF.write(model.name+"\n")
#            for stat in stats:
#                logF.write(stat+":"+str(round(model.__dict__[stat],4))+"\n")
    return models
    
 
            


def evaluate_models_on_validationSet(df,validation_set,target_name,models,TrainModel,MODEL_NAME,NORMALIZE,xVarN,stats=None):
    print "\n EVALUATING models on validation set"
    if stats == None:
        stats = ['sensitivity',"specificity",'precision','accuracy']

    VALIDATION_df = df[df['_TRAIN']==validation_set]
    validation_df = VALIDATION_df.copy()
    validation_target = np.array(validation_df[target_name])
    
    for modObj in models:
        model_name = modObj.name

        model = None
        all_train_sets = [key for key in TrainModel[model_name].keys()]
        trs = all_train_sets[0] #eventually this should be obsolete
        model = TrainModel[model_name][trs]['model']

        if MODEL_NAME in ["FOREST", "SVM","NaiveBayes",'GBT','AdaBoost']:
            validation_data = modelling_functions.make_dataMatrix_fromDF(xVarN,validation_df,normalizeInput=NORMALIZE)    
            predicted_values_validation = model.predict(validation_data)
    
        elif MODEL_NAME in ["Logit","LOGLOG"]:   
            LOGIT_TH = float(model_name.split('_TH')[-1])*0.1
            sign_VarN = TrainModel[model_name][trs]['model_variables']
            y_pred = model.predict(validation_df[sign_VarN]) #modelling_functions.calculate_yPred_logit(model,validation_df)
    
            predicted_values_validation = np.array(np.zeros(len(validation_df.index)))
            for i in xrange(len(validation_df.index)):
                if y_pred[i]>= LOGIT_TH:
                    predicted_values_validation[i] = 1
        
        sensitivity,specificity,precision,accuracy = modelling_functions.calculate_SensSpecifPrecAccur(predicted_values_validation,validation_target)                  
        #print  "For Validation prediction \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(sensitivity,specificity,precision,accuracy)
            
        for stat in stats:
            #print stat,"=",eval(stat)
            modObj.__dict__[stat] = modObj.__dict__[stat]*.3 + eval(stat)*.7
 
def determine_best_model(models,stats=None,stat_weights=None,logF=None):
    """to run AFTER summarize_model_performance() and after evaluating model on validation set"""
    best_avg = 0.0
    best_model_name = ""
    if stats == None:
        stats = ['sensitivity',"specificity",'precision','accuracy']

    if stat_weights == None:
        stat_weights = [1 for stat in stats]
    
    for model in models:
        #print model.name
        avg = 0
        for stat in stats: #:
            #print stat,'=', model.__dict__[stat]
            avg += model.__dict__[stat]*stat_weights[stats.index(stat)]
        avg = avg/float(len(stats))
        if avg >= best_avg:
            best_model_name = model.name
            best_avg = avg
    print "\nbest model is:", best_model_name

    for model in models:
        if model.name == best_model_name:
            for stat in stats:#['sensitivity',"specificity",'precision','accuracy']:
                print stat,'=', model.__dict__[stat]
                
    if logF != None:
        logF.write("\nbest model of this type:")
        for model in models:
            if model.name == best_model_name:
                logF.write(best_model_name +"\n")
                for stat in stats:
                    logF.write(stat+":"+str(round(model.__dict__[stat],4))+"\n")
        
    return best_model_name

               
    
def evaluate_model(trs_npArray,random_prediction_target,df,validation_set,target_name,TrainModel,best_model,MODEL_NAME,NORMALIZE,xVarN,modelCoeff2file=False,logF=None):

    if validation_set == -1:
        print "\n TESTING on the ENTIRE set"
        VALIDATION_df = df
        AdH_predicted_values_validation = random_prediction_target
    else:
        print "\n VALIDATING",MODEL_NAME,"on validation set (",validation_set,")"
        VALIDATION_df = df[df['_TRAIN']==validation_set]
        AdH_predicted_values_validation = random_prediction_target[trs_npArray==validation_set]

    validation_df = VALIDATION_df.copy()
    validation_target = np.array(validation_df[target_name])
    
    model = None
    for trs in TrainModel[best_model].keys():
        model = TrainModel[best_model][trs]['model']
        break
 
    if MODEL_NAME in ["FOREST", "SVM","NaiveBayes",'GBT','AdaBoost']:
        validation_data = modelling_functions.make_dataMatrix_fromDF(xVarN,validation_df,normalizeInput=NORMALIZE)    
        predicted_values_validation = model.predict(validation_data)

    elif MODEL_NAME in ["Logit","LOGLOG"]:   
        LOGIT_TH = float(best_model.split('_TH')[-1])*0.1
        sign_VarN = TrainModel[best_model][trs]['model_variables']
        if 'const' in sign_VarN:
            validation_df['const'] = 1
        y_pred = model.predict(validation_df[sign_VarN]) #modelling_functions.calculate_yPred_logit(model,validation_df)

        predicted_values_validation = np.array(np.zeros(len(validation_df.index)))
        for i in xrange(len(validation_df.index)):
            if y_pred[i]>= LOGIT_TH:
                predicted_values_validation[i] = 1
        if modelCoeff2file:
            write_2file_ModelCoeff("t_best_"+MODEL_NAME+"_coeff"+datetime.datetime.now().strftime("%Y%m%d-%H%M")+".csv",model)
        #print model.summary()
             


    AdH_sensitivity,AdH_specificity,AdH_precision,AdH_accuracy = modelling_functions.calculate_SensSpecifPrecAccur(AdH_predicted_values_validation,validation_target)                  
    s =  "For AdHoc prediction on validation set of observations \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(AdH_sensitivity,AdH_specificity,AdH_precision,AdH_accuracy)

    vs_sensitivity,vs_specificity,vs_precision,vs_accuracy = modelling_functions.calculate_SensSpecifPrecAccur(predicted_values_validation,validation_target)                  
    s +=  "For Validation prediction of "+MODEL_NAME+" \n sensitivity %f\n specificity %f\n precision %f\n accuracy %f\n" %(vs_sensitivity,vs_specificity,vs_precision,vs_accuracy)

    #print metrics.classification_report(validation_target,predicted_values_validation)    
    s += "\nTP = %d, FP = %d \nFN = %d, TN = %d" %modelling_functions.cal_TP_FP_FN_TN(predicted_values_validation,validation_target)


    s += "\n\nmodel's precision improvement: "+str(round(vs_precision/AdH_precision,2)) +"\n"
    s += "model's accuracy improvement: "+str(round(vs_accuracy/AdH_accuracy,2)) +"\n"
    print s
    if logF != None:
        logF.write(s)

def write_2file_ModelCoeff(fn,model):
    fn = open(fn,'w')
    coeff = modelling_functions.get_model_coefficients(model)
    for var in coeff.keys():
        fn.write(var+","+str(coeff[var]['mean'])+","+str(coeff[var]['CI_lower'])+","+str(coeff[var]['CI_upper'])+"\n")
    fn.close()

   
def viz_model_performance(models,stats,MODEL_NAME,plt=True):
    """called AFTER summarize_model_performance();
    this function is ONLY for graphing purposes"""
    if stats == None:
        stats = ['sensitivity',"specificity",'precision','accuracy']
    
    StatsD= {}
    labels = []
    for model in models:
        labels.append(model.name)
        for stat in stats:
            if stat not in StatsD.keys():
                StatsD[stat] = []
            if stat in model.__dict__.keys():
                StatsD[stat].append(model.__dict__[stat])

    if plt:
        if "sensitivity" in StatsD.keys() and  StatsD['sensitivity'] != []: #specificity should not be empty either in this case
            vec1 = np.array(StatsD['sensitivity'])
            vec2 = np.array(StatsD['precision'])
            plot_functions.indexed_plots(vec1,vec2,"sensitiviy(recall)","precision",labels,title="Precision Vs. recall of "+MODEL_NAME,figname="PrecVsRecall_"+MODEL_NAME)
 
       
#        if "precision" in StatsD.keys() and  StatsD['precision'] != []:
#            vec1 = np.array(StatsD['accuracy'])
#            vec2 = np.array(StatsD['precision'])
#            plot_functions.indexed_plots(vec1,vec2,"accuracy","precision",labels,title="Precision vs Accuracy of "+MODEL_NAME,figname="PrecVsAccur_"+MODEL_NAME)

#        if "sensitivity" in StatsD.keys() and  StatsD['sensitivity'] != []: #specificity should not be empty either in this case
#            vec1 = np.ones(len(StatsD['specificity'])) -np.array(StatsD['specificity'])
#            vec2 = np.array(StatsD['sensitivity'])
#            plot_functions.indexed_plots(vec1,vec2,"1-specificity","sensitivity",labels,title="ROC of "+MODEL_NAME,figname="ROC_"+MODEL_NAME)
        
   
def form_validation_set(df,random_prediction_target,validation_set,target_name,train_var):
    """currently not used"""
    VALIDATION_df = df[df[train_var]==validation_set]
    validation_df = VALIDATION_df.copy()
    validation_target = np.array(validation_df[target_name])
    AdH_predicted_values_validation = random_prediction_target[trs_npArray==validation_set]
    return validation_df,validation_target,AdH_predicted_values_validation
    

    
