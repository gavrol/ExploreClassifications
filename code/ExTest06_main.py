# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 08:53:58 2014

@author: ogavril

purpose:  testing Main on test06
"""

import os
import pandas as pd
import numpy as np
import datetime
import platform

import modelling_functions
import model_choices
import common_functions


####test06 stuff starts here    
def var_column_names(df):
    """this function must be changed per each DF used
        This function returns the column names which are the variables to be used for fitting"""
    var_names = []
    for col in df.columns:
        if col not in ["train","const",'vote', 'logpopul','_TRAIN']:
            var_names.append(col)
    return var_names

    
    
"""getting data"""   
fn = ".."+os.sep+"data"+os.sep +"test06.csv"
df = pd.read_csv(fn)

print "column names",df.columns
print "number of observations:",len(df.index)

column_names = ['popul', 'TVnews', 'selfLR', 'ClinLR', 'DoleLR', 'age', 'educ', 'income', 'PID',]
df = df.dropna()
print "number of observations:",len(df.index)


train_var = '_TRAIN'
df = common_functions.def_cross_validation_subsets(df,train_var,numK=5)
trs_npArray = np.array(df[train_var])
target_name = 'vote'
success_rate = float(df[target_name][df[target_name]==1].count())/float(df[target_name].count())
random_prediction_target = modelling_functions.randomize_prediction_v1(df[target_name],success_rate)



##### starting model choice #######

MODEL_STATSD = {} #dictionary to fill with model
MODELS = []
MODEL_NAMES = ['SVM',"Logit","LOGLOG","AdaBoost","FOREST","GBT","NaiveBayes",] 

logf = open("log_test06_"+datetime.datetime.now().strftime("%Y%m%d-%H%M")+".csv",'w')

model_pass = {'FOREST':False,'AdaBoost':False,'GBT':False,"SVM":False}#to enble pass for some models if they have been ran once
TrainModel = {}
validation_set = 4

for MODEL_NAME in MODEL_NAMES:
    Normalize_Vars = True
    
    xVarN = var_column_names(df)
    print "\n\n applying",MODEL_NAME   
    logf.write("\n\n applying "+MODEL_NAME+"\n")
        
    if MODEL_NAME in ["Logit","LOGLOG"] and ('const' not in df.columns):
        df["const"] = 1
        xVarN  += ["const"]
    print "starting independent variables",xVarN
    
    
    TrainModel_Stats = {}
    TestModel_Stats = {}


    train_sets = range(4)#[elem for elem in df[train_var].unique() if elem != validataion_set]
       
        
    for trs in train_sets:
        
        print "\n TEST set:",trs,"(i.e., train set excludes set",trs,'and',validation_set,")"
        if MODEL_NAME in ['FOREST','AdaBoost','GBT',"SVM"]:
            TRAIN_df = df[(df['_TRAIN'] != validation_set)]
        else:
            TRAIN_df = df[(df['_TRAIN']!=trs) & (df['_TRAIN'] != validation_set)]
        TEST_df = df[df['_TRAIN']==trs]
        TrainModel_Stats[trs] = {}
        TestModel_Stats[trs] = {}
    
        if MODEL_NAME == "FOREST" and not model_pass['FOREST']:
            print MODEL_NAME,"for testset",trs
            model_pass['FOREST'] = True
            Features = ["auto","log2"]
            NumTrees = xrange(10,100,20)
            min_samples_split=5
            min_samples_leaf=6
            kernels = model_choices.use_Forest(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,Normalize_Vars,Features,NumTrees,min_samples_split,min_samples_leaf)

        if MODEL_NAME == "AdaBoost" and not model_pass['AdaBoost']:
            print MODEL_NAME,"for testset",trs
            model_pass['AdaBoost'] = True
            NumEstimators = xrange(10,100,20)
            Features = ['log2','auto']
            min_samples_split=6
            min_samples_leaf=6
            kernels = model_choices.use_AdaBoost(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,Normalize_Vars,Features,NumEstimators,min_samples_split,min_samples_leaf)


        elif MODEL_NAME == "GBT" and not model_pass['GBT']:
            print MODEL_NAME,"for testset",trs
            model_pass['GBT'] = True
            learning_rate = 0.05
            Depths = xrange(8,20,10)
            NumBoostingStages = xrange(10,150,50)
            Features = ['log2','auto']
            min_samples_split=6
            min_samples_leaf=6
            kernels = model_choices.use_GBT(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,Normalize_Vars,Depths,NumBoostingStages,Features,min_samples_split,min_samples_leaf,learning_rate)

        elif MODEL_NAME == "SVM" and not model_pass['SVM']:
            print MODEL_NAME,"for testset",trs
            model_pass['SVM'] = True
            kernels = ['linear','rbf','sigmoid','poly',]
            poly_degrees = [2,3,4]
            kernels = model_choices.use_SVM(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,Normalize_Vars,kernels,poly_degrees)
    
        elif MODEL_NAME == "NaiveBayes":  
            print MODEL_NAME,"for testset",trs
            kernels = ["Bernoulli",'Multinomial','Gaussian']
            kernels = model_choices.use_NaiveBayes(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,Normalize_Vars,kernels)
        
        elif MODEL_NAME == "Logit":
            LOGIT_THS = xrange(3,8)
            kernels = model_choices.use_Logit(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,LOGIT_THS)
    
        elif MODEL_NAME == "LOGLOG":
            print MODEL_NAME,"for testset",trs
            LOGIT_THS = xrange(3,8)
            kernels = model_choices.use_LOGLOG(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,LOGIT_THS)
            
    """ for models for which scikit does not have CV, determine common coeff"""
    if MODEL_NAME in ["Logit","LOGLOG"]:
        model_coeff = model_choices.coefficient_tuning_via_CrossValidation(TrainModel,TestModel_Stats,MODEL_NAME)
        for trs in train_sets:
            TRAIN_df = df[(df['_TRAIN'] != validation_set)]
            TEST_df = df[df['_TRAIN']==trs]
            TrainModel_Stats[trs] = {}
            TestModel_Stats[trs] = {}
            xVarN = [ varName for varName in model_coeff.keys() if model_coeff[varName]['mean'] != 0] 
            print "\nre-doing",MODEL_NAME,"with",len(xVarN),"coeff"
            
            if MODEL_NAME == "Logit":
                LOGIT_THS = xrange(3,8)
                kernels = model_choices.use_Logit(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,LOGIT_THS)
            elif MODEL_NAME == "LOGLOG":
                LOGIT_THS = xrange(3,8)
                kernels = model_choices.use_LOGLOG(TestModel_Stats,TrainModel_Stats,TrainModel,TRAIN_df,TEST_df,xVarN,target_name,trs,LOGIT_THS)

        
    models = model_choices.summarize_model_performance(TestModel_Stats,['sensitivity',"specificity",'precision','accuracy'],logF= logf)
    MODELS.extend(models)

    model_choices.evaluate_models_on_validationSet(df,validation_set,target_name,models,TrainModel,MODEL_NAME,Normalize_Vars,xVarN,stats=None)
    best_model = model_choices.determine_best_model(models,stats=['sensitivity',"specificity",'precision','accuracy'],stat_weights=[0.6,0.0,0.4,0.0],logF= logf)    
    model_choices.evaluate_model(trs_npArray,random_prediction_target,df,validation_set,target_name,TrainModel,best_model,MODEL_NAME,Normalize_Vars,xVarN,logF=logf,modelCoeff2file=True)
    model_choices.evaluate_model(trs_npArray,random_prediction_target,df,-1,target_name,TrainModel,best_model,MODEL_NAME,Normalize_Vars,xVarN,logF=logf)


    if 'const' in df.columns:
        df = df.drop('const',1)
#    if platform.system().lower() not in ["linux"]:
#        model_choices.viz_model_performance(models,['sensitivity',"specificity",'precision','accuracy'],MODEL_NAME)    


"""now choose the best model"""
print "\nALL models"
logf.write("\n ALL models comparison\n")
ALL_best_model = model_choices.determine_best_model(MODELS,stats=['sensitivity',"specificity",'precision','accuracy'],stat_weights=[0.6,0.0,0.4,0.0],logF= logf)
MODEL_NAME = ALL_best_model.split("_")[0]
print MODEL_NAME
model_choices.evaluate_model(trs_npArray,random_prediction_target,df,validation_set,target_name,TrainModel,ALL_best_model,MODEL_NAME,Normalize_Vars,xVarN,logF=logf)

model_choices.evaluate_model(trs_npArray,random_prediction_target,df,-1,target_name,TrainModel,ALL_best_model,MODEL_NAME,Normalize_Vars,xVarN,logF=logf)

logf.close()

if platform.system().lower() not in ["linux"]:    
    model_choices.viz_model_performance(MODELS,['sensitivity',"specificity",'precision','accuracy'],"ALL_MODELS")
