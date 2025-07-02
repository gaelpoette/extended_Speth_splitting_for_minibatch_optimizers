import numpy as np
import math
from numpy.lib.function_base import median
import pandas as pd
from scipy import stats
import seaborn as sns
import random as rd
import os
import matplotlib
import time
#from interval import interval, inf

""" matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
}) """
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.ticker as ticker
from matplotlib.patches import Polygon

np.random.seed(0)
a=-1; b=1
prec=10**(-16)

#ex1 et 2 sur [-1,1]
def R(x,i,type='ex1'):
    if(type=="ex1"):
        if(i==0):
            return x[0]**2+1
        else:
            return (2*x[0]-1)**2+1
    elif(type=="ex2"):
        if(i==0):
            return 2*x[0]**4-x[0]**2+1 # par défaut +1
        else:
            return 12*x[0]**4+4*x[0]**3-9*x[0]**2+4
    elif(type=="ex3"):
        if(i==0):
            return 12*x[0]**10-4*x[0]**9+5*x[0]**8+x[0]**6-3*x[0]**5-2*x[0]**4-x[0]**3+x[0]**2-x[0]+1
        else:
            return x[0]**6-x[0]**5-x[0]**3+x[0]+1
    elif(type=="ex4"):
        if(i==0):
            return x[0]**6+x[0]**5+x[0]**3-2*x[0]**2+9
        else:
            return x[0]**6-x[0]**5-x[0]**3+x[0]+1
    elif(type=="ex5"):
        if(i==0):
            return x[0]**2+1
        elif(i==1):
            return (2*x[0]-1)**2+1
        else:
            return (2*x[0]+3)**2+1
    elif(type=="ex6"):
        if(i==0):
            return 2*x[0]**4-x[0]**2+1
        elif(i==1):
            return 12*x[0]**4+4*x[0]**3-9*x[0]**2+4
        else:
            return 4*x[0]**6+14/5*x[0]**5-7/4*x[0]**4-x[0]**3+1
    elif(type=="ex7"):
        if(i==0):
            return x[0]**6+x[0]**5+x[0]**3-2*x[0]**2+9
        elif(i==1):
            return x[0]**6-x[0]**5-x[0]**3+x[0]+1
        else:
            return x[0]**6+0.5*x[0]**5+2*x[0]**3+2
    elif(type=="ex8"):
        if(i==0):
            return x[0]**6+x[0]**5+x[0]**3-2*x[0]**2+9
        elif(i==1):
            return x[0]**6-x[0]**5-x[0]**3+x[0]+1
        elif(i==2):
            return x[0]**6+0.5*x[0]**5+2*x[0]**3+2
        else:
            return x[0]**3+x[0]**2-2*x[0]+1
    elif(type=="ex9"):
        if(i==0):
            return x[0]**6+x[0]**5+x[0]**3-2*x[0]**2+9
        elif(i==1):
            return x[0]**6-x[0]**5-x[0]**3+x[0]+1
        elif(i==2):
            return x[0]**6+0.5*x[0]**5+2*x[0]**3+2
        elif(i==3):
            return x[0]**3+x[0]**2-2*x[0]+1
        elif(i==4):
            return 12*x[0]**10-4*x[0]**9+5*x[0]**8+x[0]**6-3*x[0]**5-2*x[0]**4-x[0]**3+x[0]**2-x[0]+1
        elif(i==5):
            return (2*x[0]-1)**2+1
        elif(i==6):
            return x[0]**2+1
        else:
            return 4*x[0]**6+14/5*x[0]**5-7/4*x[0]**4-x[0]**3+1
    elif(type=="polyTwo" or type=="polyThree" or type=="polyFour" or type=="polyFive"):
        if(i==0):
            return 0.5*g(x[1],typeR)**2
        else:
            return 0.5*g(x[0]+x[1],typeR)**2

def gradR(x,i,type='ex1'):
    if(type=="ex1"):
        if(i==0):
            return 2*x[0]
        else:
            return 4*(2*x[0]-1)
    elif(type=="ex2"):
        if(i==0):
            return 8*x[0]**3-2*x[0]
        else:
            return 48*x[0]**3+12*x[0]**2-18*x[0] 
    elif(type=="ex3"):
        if(i==0):
            return 120*x[0]**9-36*x[0]**8+40*x[0]**7+6*x[0]**5-15*x[0]**4-8*x[0]**3-3*x[0]**2+2*x[0]-1
        else:
            return 6*x[0]**5-5*x[0]**4-3*x[0]**2+1
    elif(type=="ex4"):
        if(i==0):
            return 6*x[0]**5+5*x[0]**4+3*x[0]**2-4*x[0]
        else:
            return 6*x[0]**5-5*x[0]**4-3*x[0]**2+1
    elif(type=="ex5"):
        if(i==0):
            return 2*x[0]
        elif(i==1):
            return 4*(2*x[0]-1)
        else:
            return 4*(2*x[0]+3)
    elif(type=="ex6"):
        if(i==0):
            return 8*x[0]**3-2*x[0]
        elif(i==1):
            return 48*x[0]**3+12*x[0]**2-18*x[0] 
        else:
            return 24*x[0]**5+14*x[0]**4-7*x[0]**3-3*x[0]**2
    elif(type=="ex7"):
        if(i==0):
            return 6*x[0]**5+5*x[0]**4+3*x[0]**2-4*x[0]
        elif(i==1):
            return 6*x[0]**5-5*x[0]**4-3*x[0]**2+1 
        else:
            return 6*x[0]**5+5/2*x[0]**4+6*x[0]**2
    elif(type=="ex8"):
        if(i==0):
            return 6*x[0]**5+5*x[0]**4+3*x[0]**2-4*x[0]
        elif(i==1):
            return 6*x[0]**5-5*x[0]**4-3*x[0]**2+1 
        elif(i==2):
            return 6*x[0]**5+5/2*x[0]**4+6*x[0]**2
        else:
            return 3*x[0]**2+2*x[0]-2
    elif(type=="ex9"):
        if(i==0):
            return 6*x[0]**5+5*x[0]**4+3*x[0]**2-4*x[0]
        elif(i==1):
            return 6*x[0]**5-5*x[0]**4-3*x[0]**2+1 
        elif(i==2):
            return 6*x[0]**5+5/2*x[0]**4+6*x[0]**2
        elif(i==3):
            return 3*x[0]**2+2*x[0]-2
        elif(i==4):
            return 120*x[0]**9-36*x[0]**8+40*x[0]**7+6*x[0]**5-15*x[0]**4-8*x[0]**3-3*x[0]**2+2*x[0]-1
        elif(i==5):
            return 4*(2*x[0]-1)
        elif(i==6):
            return 2*x[0]
        else:
            return 24*x[0]**5+14*x[0]**4-7*x[0]**3-3*x[0]**2
    elif(type=="polyTwo" or type=="polyThree" or type=="polyFour" or type=="polyFive"):
        grad=np.ones(2)
        if(i==0):
            grad[0]=0
            grad[1]=g(x[1],type)*gp(x[1],type)
        else:
            grad[0]=g(x[0]+x[1],type)*gp(x[0]+x[1],type)
            grad[1]=g(x[0]+x[1],type)*gp(x[0]+x[1],type)
        return grad

def grad_sum(x,typeR,m):
    g=np.zeros(N)
    for i in range(m):
        g+=gradR(x,i,typeR)
    return g

       
def u0(x,typeCI="uniform"):
    if(typeCI=="uniform"):
        if(x[0]>a and x[0]<b):
            return 1/(b-a)
        else:
            return 0
    elif(typeCI=="normal"):
        mu=0; sigma=1
        return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x[0]-mu)/sigma)**2)
    

#Dans nos examples, P=m
def eS_mom(x0,m,lr_init,eps,maxEpoch,typeR, num):
    epoch=0
    P=m
    v0 = 0
    x  = x0
    v  = v0 
    b1   = 0.9
    bb   = 1.0
    eta  = 0.
    v_a  = 0.
    grad = 0.
    g = 0.
    grad_tab =np.zeros(m)
    gNorm = 1000

    while epoch < maxEpoch and gNorm/P > eps:
       
      if epoch > 0:
        eta=lr_init; 
      x_prec=x0
      v_prec=v0
      for i in range(m):
        grad=gradR(x,i,typeR)  
        g -= grad_tab[i]
        g += grad
        #v = b1 *  v_prec/m - eta * g
        #x =       x_prec + eta * v/m
        v = b1 *  v_prec/m - eta * g/m
        x =       x_prec + eta * v/m
        grad_tab[i] = grad 
        x_prec = x
        v_prec = v
      x0 = x
      v0 = v
      epoch+=1
      gNorm=np.linalg.norm(g)
    if epoch == maxEpoch:
        print(str(num)+" max epoch reached")
    return x, epoch

def exs(nbPoints, nbParticules, lr_init, eps, maxEpoch, typeCI):
    x_maille=np.linspace(a,b,num=nbPoints)
    u=np.zeros((nbPoints,1))

    fig = plt.figure(figsize=(10,2.5))
    plt.gcf().subplots_adjust(wspace=0.2,hspace=0.57)
    axes = fig.subplots(nrows=1,ncols=2)

    lr_init0 = lr_init

    for (typeR,m,L1,L2) in [("ex1",2, 2, 8), ("ex2",2, 22, 150)]:

        lr_init = lr_init0 #1./ (L1+L2) * lr_init0

        print(typeR)
        t0 = time.time()
        for p in range(nbParticules):
            x_unif=np.random.uniform(a,b); x0=x_unif*np.ones(N)
            
            x,epoch = speth_mom(x0,m,lr_init,eps,maxEpoch,typeR, p)

            if(x>a and x<b):
                #print(x) 
                #print(u[int(nbPoints*(x[0]-a)/(b-a)),0])
                u[int(nbPoints*(x[0]-a)/(b-a)),0]+=u0(x0,typeCI)
        u/=nbParticules

        if(typeR=="ex1"):
            ax=axes[0]; titre="a)"
            ax.axvline(x=0.4,ymin=0,ymax=1.1,color='green')
        elif(typeR=="ex2"):
            ax=axes[1]; titre="b)"
            ax.axvline(x=0.5,ymin=0,ymax=1.1, color='blue')
            ax.axvline(x=-0.714285714,ymin=0,ymax=1.1, color="green")

        ax.set_title(titre)
        ax.set_xlabel(r"$\theta$")
        #locs, labels = ax.set_xticks()
        ax.set_xticks(np.arange(a, b, step=0.25))
        ax.plot(x_maille, u, label=r"$u$", color="black", ms=5, marker='.')
        ax.legend()
        t1 = time.time()
        print(f"time = {t1-t0:e}")
 
    plt.savefig('eS_mom_exs.pgf')
    plt.show()

#OK speth_mom
typeCI="uniform"
N=1
nbPoints=1000
nbParticules=10000
lr_init=0.125
eps=10**(-4); 
maxEpoch=2000
#maxEpoch=20

exs(nbPoints,nbParticules,lr_init,eps,maxEpoch,typeCI)
