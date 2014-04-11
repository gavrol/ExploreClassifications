# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 09:24:20 2014

@author: ogavril

PURPOSE: functions to clean up, transform  the data

"""
import numpy as np
import pandas as pd
import scipy as sp

import plot_functions

def define_response_vars(df):
    """this is a DF-specific function and hence all the variable names"""
    buy_option_varN = 'buyer:looking to buy'
    buyer_survey_response_options = ['Other', 'A more expensive home', 'A similar priced home', 'A less expensive home', 'An investment property_11306', 'Your first home']
    
    df["_RESPONSE_FHB"] = [0 for ex in df.index]
    df["_RESPONSE_INV"] = [0 for ex in df.index]
    df["_RESPONSE_BUYERTYPE"] = [0 for ex in df.index]
    df["_RESPONSE_VISITORTYPE"] = [0 for ex in df.index]
    df["_RESPONSE_HOTMOVER"] = [0 for ex in df.index]
    
    for i in df.index:
        if str(df[buy_option_varN][i]).find('Your first home')>=0:
            df["_RESPONSE_FHB"][i] = 1
        if str(df[buy_option_varN][i]).find('An investment property_11306')>=0:
            df["_RESPONSE_INV"][i] = 1
    
        if str(df[buy_option_varN][i]).find('Your first home')>=0:
            df["_RESPONSE_BUYERTYPE"][i] = 1#'FHB'
        elif str(df[buy_option_varN][i]).find('home')>=0:
            df["_RESPONSE_BUYERTYPE"][i] = 2#'CHANGE'
        elif str(df[buy_option_varN][i]).find('investment property') >=0:
            df["_RESPONSE_BUYERTYPE"][i] = 3#'INV'
        elif str(df[buy_option_varN][i]).find('Other') >=0:
            df["_RESPONSE_BUYERTYPE"][i] = 4#'OTHER'
        #else:df["_RESPONSE_BUYERTYPE"][i] = 0 #'UNKNOWN'
            
        
        #defining visitor type    
        if str(df[buy_option_varN][i]).split("|")[0] in buyer_survey_response_options:
            df["_RESPONSE_VISITORTYPE"][i] = 1#'BUYER'
        if str(df['seller?'][i]) == 'Yes':
            df["_RESPONSE_VISITORTYPE"][i] = 2#'SELLER'
        if str(df['builder'][i]) == 'Yes':
            df["_RESPONSE_VISITORTYPE"][i] = 3#'BUILDER'
        if str(df['renter'][i]) == 'Yes':
            df["_RESPONSE_VISITORTYPE"][i] = 4#'RENTER'
        if str(df['sharer'][i]) == 'Yes':
            df["_RESPONSE_VISITORTYPE"][i] = 5#'SHARER'
        if str(df['landlord'][i]) == 'Yes':
            df["_RESPONSE_VISITORTYPE"][i] = 6#'LANDLORD'
        if str(df['rennovator'][i]) == 'Yes':
            df["_RESPONSE_VISITORTYPE"][i] = 7#'RENOVATOR'

        #filling a variable to POSSIBLY detect hot-movers
        if df["_RESPONSE_VISITORTYPE"][i] in [1,4]:
            df["_RESPONSE_HOTMOVER"][i] = 1            
    return df


    
def valid_columns(df):
    cols = []
    
    for col in df.columns:
        if col not in ['_TRAIN','id','_RESPONSE_VISITORTYPE','_RESPONSE_BUYERTYPE','_RESPONSE_FHB','_RESPONSE_INV',"_RESPONSE_HOTMOVER"]:
            cols.append(col)
    return cols
    
def def_cross_validation_subsets(df,varN,numK=5):
    df[varN] = -1
    for i in xrange(len(df.index)):
        df[varN][i] = i%numK
    return df
    
def differ_by_last(name1,name2):
    if len(name1.split("_")) != len(name2.split("_")):
        return False
    else:
        name1 = name1.split("_")
        name2 = name2.split("_")
        for i in range(len(name1)-1):
            if name1[i] != name2[i]:
                return False
        return True
    
def change_variables(df):
    print "aggregate columns"
    columns = valid_columns(df)
    cols2ignore = []
    new_cols = []
    for c in range(len(columns)):
        aggr = []
        colN = columns[c]
        if colN not in cols2ignore:
            aggr.append(colN)
            for cn in range(c+1,len(columns)):
                colNn = columns[cn]
                if colNn not in cols2ignore:
                    if differ_by_last(colN,colNn):
                        cols2ignore.append(colNn)
                        aggr.append(colNn)
        if len(aggr) > 1:
            #print '\n',colN,"connected to:",aggr
            nname = "_".join(aggr[0].split("_")[:-1])
            new_cols.append(nname)
            df[nname] = 0
            for name in aggr:
                df[nname] += df[name]
                
            df = df.drop(aggr,1)
            df[nname][df[nname] > 0] = 1
            #df[nname][df[nname] <= df[nname].mean()-3*df[nname].std()] = 0
    #print "New columns",new_cols
    return df
                        
    
def check_for_NaNs(df):
    df1 = df.dropna()
    if len(df1.index) != len(df.index):
        print "there are NaNs in the DF...decide what you want to do with NaNs"
    else:
        print "there are no NaNs in the DF"

def drop_outliering_observations(df,col_names=None, num_std = 15.0):
    """ because of the way the data for FHB is, this function SHOULD NOT be used right now"""
    
    print "\ndropping outliers"
    ndf = df.dropna()    
    if col_names == None:
        col_names = valid_columns(ndf)
    obser2remove = []
    
    for col in col_names:
#        if col == "buy_page_views":
        limit = ndf[col].mean()+15.0*ndf[col].std()
        #print col, "limit=",limit
        for indx in ndf.index:
            if ndf[col][indx] > limit:
                obser2remove.append(indx)
                print col,"outlier=",ndf[col][indx]
    #print obser2remove
    df = df.drop(obser2remove,0)
    return df        


def determine_vars_for_scaling(df):
    means = []
    medians = []
    large_means = []
    for col in valid_columns(df):
        means.append(df[col].mean())
        medians.append(df[col].median())    

        
    Mmean = np.mean(np.array(means))
    stdMeans = np.std(np.array(means))
    Mmedian = np.mean(np.array(medians))
    print "mean of means=",Mmean,'mean of medians=',Mmedian,"std(Means)=",stdMeans

    num_large = 0
    for col in  valid_columns(df):
        if df[col].mean() > Mmean:
            num_large += 1
    print "number of cols to scale"
    if num_large < len(valid_columns(df))*0.05:
        for col in  valid_columns(df):
            if df[col].mean() > Mmean + stdMeans:     
                large_means.append(col)
    
#    fn = open("t_large_means.csv",'w')
#    fn.write("\n".join(large_means))
#    fn.close()
    #plot_functions.hist_plot(means,figname="means_of_vars")
    return large_means
    
        

def drop_sparse_columns(df,sparse_rate=0.99):
    """when number of columns is large feature elimination can be aided by eliminating columns that have very few entries. 
    What is "FEW" entries???
    e.g., a column is filled mostly one value (e.g., 0,1 Na, or some other value) EXCEPT for a few observations 
    (say less than 1% ) differ from the rest 99%, then such columns can be dismissed, as these 1% of observ can be classified
    as outliers for this particular variable/column"""
    ndf = df.dropna()
    cols2remove = []
    columns = valid_columns(ndf)

    for col in columns:
        v = np.array(ndf[col])
        arr,num_occr = sp.stats.mode(v)
        elem = arr[0]
        num_occr = num_occr[0]
        rate = float(num_occr)/float(len(v))
        if rate > sparse_rate:
            #print col,"filled with",elem,"in",round(rate*100,1),"occurrences"
            cols2remove.append(col)
    print "\nnumber of columns that will be removed because they are sparse",len(cols2remove)
    #print "columns that are filled with the same values (ignoring NaNs) are:",cols2remove
    s = "["    
    for col in cols2remove:
        s+='"'+col+'",'
    s += "]"
    t_Fn= open("col_names_sparse.txt",'w')
    t_Fn.write(s)
    t_Fn.close()
    df = df.drop(cols2remove,1)
    return df        
                        
    
def drop_homogeneous_columns(df):
    """columns that have min=max MUST be all filled with the same value,
       drop such columns 
       No need to worry about NaNs
    """
    cols2remove = []
    for col in df.columns:
        v = np.array(df[col])
        v = v[~np.isnan(v)]
        if np.min(v) == np.max(v):#df[col].min() == df[col].max():
            cols2remove.append(col)
 
    print "\nnumber of columns that will be removed because they are filled with only one value",len(cols2remove)
    #print "columns that are filled with the same values (ignoring NaNs) are:",cols2remove
    s = "["    
    for col in cols2remove:
        s+='"'+col+'",'
    s += "]"
    t_Fn= open("col_names_homogeneous.txt",'w')
    t_Fn.write(s)
    t_Fn.close()
    df = df.drop(cols2remove,1)
    return df

def drop_duplicated_columns(df):
    """there are columns that are duplicates of each other, i.e., a repeat of a column. drop the repeats """
    columns = df.columns
    cols2remove = []
    for c in range(len(columns)):
        if columns[c] not in cols2remove:
            for cn in range(c+1,len(columns)):
                if cn not in cols2remove:
                    v =  np.array(df[columns[c]] - df[columns[cn]])
                    v = v[~np.isnan(v)]
                    if np.min(v) == np.max(v):
                        print "in dupplicated columns",np.min(v)
                        #print columns[c],'=',columns[cn],'dropping col',columns[cn]
                        cols2remove.append(columns[cn])

    print "\nnumber of columns that will be removed because they are duplicates of other columns", len(cols2remove)
    #print "the following columns will be dropped due to being duplicates of other columns",cols2remove
    s = "["    
    for col in cols2remove:
        s+='"'+col+'",'
    s += "]"
    t_Fn= open("col_names_duplicated.txt",'w')
    t_Fn.write(s)
    t_Fn.close()    
    df = df.drop(cols2remove,1)
    return df

def drop_correlated_columns_fast(df,CorrCoefLimit=0.76):

    ndf = df.dropna()
    cols2remove = []
    columns = valid_columns(ndf)    
          
    mat = np.array(ndf[columns[0]])
    for c in range(1,len(columns)):
        mat = np.vstack((mat,ndf[columns[c]]))
    corrmat = np.corrcoef(mat)
    
    for c in range(1,len(columns)-1):
        if columns[c] not in cols2remove:
            for cn in range(c+1,len(columns)):
                if columns[cn] not in cols2remove:
                    if abs(corrmat[c][cn]) > CorrCoefLimit:
                        cols2remove.append(columns[cn])
#                        if columns[c] in columns or columns[cn] in columns:
#                            print "col",columns[c],columns[cn],"are correlated,dropping",columns[cn]
    #print "correlated columns are:",cols2remove
    print "number of correlated columns is",len(cols2remove)
    s = "["
    for col in cols2remove:
        s+='"'+col+'",'
    s += "]"
    t_Fn= open("col_names_correlated.txt",'w')
    t_Fn.write(s)
    t_Fn.close()    

    df = df.drop(cols2remove,1)
    return df
                   
def determine_var_correlated_to_resp(df,yVarN,CorrCoefLimit=0.4,write2file=True):
    
#for respN in ['_RESPONSE_VISITORTYPE','_RESPONSE_BUYERTYPE','_RESPONSE_FHB','_RESPONSE_INV']:
#    DataWrangling_functions.determine_var_correlated_to_resp(df,respN,CorrCoefLimit=0.4)
    
    ndf = df.dropna()
    cols2keep = []
    columns = valid_columns(ndf)  
    
    resp_col = np.array(ndf[yVarN])
    
    for col in columns:
        if col != yVarN:
            v1 = np.array(ndf[col])
            mat = np.vstack((v1,resp_col))        
            corrmat = np.corrcoef(mat)
            if abs(corrmat[0][1])>CorrCoefLimit:
                cols2keep.append(col)
    
    print "VERY correlated columns with",yVarN,"are:\n",cols2keep
    if write2file:
        s = "["
        for col in cols2keep:
            s+='"'+col+'",'
        s += "]"
        t_Fn= open("col_correlated_w_"+yVarN+".txt",'w')
        t_Fn.write(s)
        t_Fn.close()    


######    
def get_vars_4clustering(df):
    key_words = ['visit','view']
    cols2keep = []
    for col in valid_columns(df):
        for word in key_words:
            if col.lower().find(word.lower())>= 0:
                cols2keep.append(col)
    for col in cols2keep:
        vec = np.array(df[col])
        #plot_functions.hist_plot(vec,figname=col)
    return cols2keep
