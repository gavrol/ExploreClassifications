# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 12:09:13 2014

@author: olena

plots for fun
"""
import numpy as np
import matplotlib.pyplot as plt
import modelling_functions

def plot_resid_vs_true(data,yName,stk,out_dir,model):
    fig, ax = plt.subplots()
    ax.plot(data[yName],model.resid,'g.')
    ax.set_title(stk+": Residuals vs True "+yName)
    ax.set_ylabel("residual size")
    ax.set_xlabel(yName)
    plt.savefig(out_dir+"Resid_vs_True_"+stk+".jpg")

def plot_predicted_vs_true(data,yName,stk,out_dir):
    fig, ax = plt.subplots()
    ax.plot(data[yName],data['pred'],'o')
    ax.set_xlabel(yName)
    ax.set_ylabel('Predicted '+yName)
    ax.set_title(stk+" Predicted vs true "+yName)
    plt.savefig(out_dir+"pred_vs_true_"+stk+".jpg")        

def plot_xVar_yVar_logit(df,xVarN,yVarN,title='',figname=None,model=None):
    """plots ONE x-variable against the response variable """
    fig, ax = plt.subplots()
    ax.scatter(df[xVarN],df[yVarN], c='r', s=100,edgecolors='black', alpha=0.5)
    if model != None:
        y_pred = modelling_functions.calculate_yPred_basedOn_1xVar_logit(model,df,xVarN,yVarN)
        ax.plot(df[xVarN],y_pred,'bo')
    #ax.plot(df[xVarN],df[yVarN],'ro',alpha=0.7)
    ax.set_xlabel(xVarN)
    ax.set_ylabel(yVarN)
    if figname == None: 
        plt.savefig(yVarN+"_vs_"+xVarN+".jpg")
    else:
        plt.savefig(figname+".jpg")       
    
def plot_true_vs_pred_logit(y,y_pred,title="",figname=None):
    fig = plt.figure()
    ax = plt.subplot(111)
    z = np.random.rand(len(y))
    ax.scatter(y,y_pred, c='y', s=100,edgecolors='black', alpha=0.5)
    #ax.plot(y,y,'r--')
    ax.set_xlabel("observed y",fontsize=8)
    ax.set_ylabel("predicted y",fontsize=8)
    fig.suptitle(title+": Predicted vs Observed",fontsize=9)
    if figname == None: 
        plt.savefig("True_vs_Pred_XXX.jpg")
    else:
        plt.savefig(figname+"_true_vs_pred.jpg")       
   
def plots2_TrueVSpred_Resid(y,y_pred,residuals=None,weights=None,y_true=None,figfn ="plots"):
    fig,axes = plt.subplots(nrows=2,ncols=1)
    fig.subplots_adjust(hspace=0.25)
    fig.suptitle(figfn,ha='center', va='center',fontsize=10,color="#FF3300")     
    ax = axes[0]
    #ax.scatter(y,y_pred)
    ax.plot(y,y_pred,'r+')
    ax.set_xlabel("observed y",fontsize=8)
    ax.set_ylabel("predicted y",fontsize=8)
    
    ax = axes[1]

    if weights != None:
        residuals = (residuals)*weights
    else:
        residuals = (residuals)
    #residuals = np.append(residuals,mod.resid[weights != 1.]*)
    #ax.scatter(y,residuals)
    ax.plot(y_pred,residuals,'g+')
    ax.plot(y_pred,np.ones(len(y))*0,'b') #plotting x = 0 line
    ax.set_xlabel("predicted y",fontsize=8)
    ax.set_ylabel("residuals",fontsize=8)
    plt.savefig(figfn+".jpg")
    
def hist_plot(vec,title=None,figname=None):
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.hist(vec,bins=50)
    if title != None:
        fig.suptitle(title)
    if figname == None:
        plt.savefig('hist.jpg')
    else:
        plt.savefig(figname+".jpg")

def plots_scatter(var1,var2,Nvar1,Nvar2,figname =None):
    fig,ax = plt.subplots(nrows=1,ncols=1)
    fig.subplots_adjust(hspace=0.25)
    fig.suptitle(figname,ha='center', va='center',fontsize=10,color="#FF3300")     
    ax.scatter(var1,var2,c='b')
    #ax.plot(var1,var2,'bo',alpha=0.5)
    ax.set_xlabel(Nvar1,fontsize=8)
    ax.set_ylabel(Nvar2,fontsize=8)
    if figname == None: 
        plt.savefig("somePlot_XXX.jpg")
    else:
        plt.savefig(figname+".jpg")       

from matplotlib.transforms import offset_copy
def indexed_plots(var1,var2,Nvar1,Nvar2,labels,title=None,figname=None):
    fig = plt.figure()
    ax = plt.subplot(111)
    e = np.random.uniform(low=-0.05,high=0.05,size=len(var1))
    transOffset = offset_copy(ax.transData, fig=fig,x = 0.05+3*e, y=0.10-3*e, units='inches')

    for x, y,l in zip(var1,var2,labels):
        ax.plot((x,),(y,), 'ro')
        indx = labels.index(l)
        #ax.text(x, y, '%s' % (l), transform= transOffset, fontsize=8)
        ax.text(x, y, '%s' % (l), fontsize=8,
                transform=offset_copy(ax.transData, fig=fig,x = 0.05+e[indx], y=0.10-e[indx], units='inches'))
    

    ax.set_xlabel(Nvar1,fontsize=8)
    ax.set_ylabel(Nvar2,fontsize=8)

    if title != None:
        fig.suptitle(title)
    if figname == None:
        plt.savefig('CURVE.jpg')
    else:
        plt.savefig(figname+".jpg")
        
def main():    
    n = 100
    v1 = np.random.exponential(scale=10,size=n)
    #hist_plot(v1,title="Exp Dist: L=10",figname="ExpDistLambda10")
    v2 = np.random.triangular(-2,0,2,size=n)
    hist_plot(v2,title="Triang Dist: Mode=0",figname="TriangDistMode0")
    

