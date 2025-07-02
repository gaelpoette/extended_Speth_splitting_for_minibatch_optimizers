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
def adam_fb(x0,m,lr_init,eps,maxEpoch,typeR, num):
    epoch=0
    P=m
    v0 = 0
    m0 = 0
    x  = x0
    v  = v0 
    m   = m0
    b1   = 0.9
    b2=0.999
    epsilon=1e-8
    bb   = 1.0
    eta  = 0.
    grad = 0.
    grad_tab =np.zeros(m)
    gNorm = 1000

    while epoch < maxEpoch and gNorm/P > eps:
       
      if epoch > 0:
        eta=lr_init; 
      x_prec=x0
      v_prec=v0
      m_prec=m0
      grad= 0.
      for i in range(P):
        grad+=gradR(x_prec,i,typeR)
      m = (1-b1) * m + b1 * grad
      v = (1-b2) * v + b2 * (grad** 2)
      f = np.sqrt((1 - b2**(epoch+1)))/(1 - b1**(epoch+1))
      x -= eta * f * m / (np.sqrt(v) + epsilon)
      x_prec = x
      v_prec = v
      m_prec = m

      x0 = x_prec
      v0 = v_prec
      m0 = m_prec
      epoch+=1
      gNorm=np.linalg.norm(grad)
    return x, epoch

def F(m, v, f, eta, grad, b1, b2, epsilon):
   m_n = (1-b1) * m + b1 * grad
   v_n = (1-b2) * v + b2 * (grad** 2)
   x_n = -f * m_n / (np.sqrt(v_n) + epsilon)
   m_n -=m
   v_n -=v
   m_n /=  eta
   v_n /=  eta
   F = [m_n, v_n, x_n]
   return F

def Fi(s, v, f, eta, grad, grad2, b1, b2, epsilon, m):
   s_n = (1-b1) * s / m + b1 * grad    / m
   v_n = (1-b2) * v / m + b2 * (grad2) / m
   x_n = -f * s_n   / m  / (np.sqrt(v_n) + epsilon)
   s_n -=s
   v_n -=v
   s_n /=  eta
   v_n /=  eta
   F = [s_n, v_n, x_n]
   return F


def adam_fb_F(x0,m,lr_init,eps,maxEpoch,typeR, num):
    epoch=0
    P=m
    v0 = 0
    m0 = 0
    x  = x0
    v  = v0 
    m   = m0
    b1   = 0.9
    b2=0.999
    epsilon=1e-8
    bb   = 1.0
    eta  = 1.e-15
    grad = 0.
    grad_tab =np.zeros(m)
    gNorm = 1000

    while epoch < maxEpoch and gNorm/P > eps:
       

      if epoch > 0:
        eta=lr_init; 
      x_prec=x0
      v_prec=v0
      m_prec=m0
      grad=gradR(x,0,typeR) +gradR(x,1,typeR)  
      f = np.sqrt((1 - b2**(epoch+1)))/(1 - b1**(epoch+1))
      Fl = F(m_prec, v_prec, f, eta, grad, b1, b2, epsilon)
      m += eta * Fl[0]
      v += eta * Fl[1]
      x += eta * Fl[2]
      x_prec = x
      v_prec = v
      m_prec = m

      x0 = x_prec
      v0 = v_prec
      m0 = m_prec
      epoch+=1
      gNorm=np.linalg.norm(grad)
    if epoch == maxEpoch:
        print(str(num)+" max epoch reached")
    print(num, x)
    return x, epoch

def adam(x0,m,lr_init,eps,maxEpoch,typeR, num):
    epoch=0
    P=m
    v0 = 0
    s0 = 0
    x  = x0
    v  = v0 
    s   = s0
    b1   = 0.9
    b2=0.999
    epsilon=1e-8
    bb   = 1.0
    eta  = 1.e-15
    grad = 0.
    grad_tab =np.zeros(m)
    gNorm = 1000

    while epoch < maxEpoch and gNorm/P > eps:
       
      if epoch > 0:
        eta=lr_init; 
      x_prec=x0
      v_prec=v0
      s_prec=s0
      for i in range(m):
        grad=gradR(x,i,typeR)  
        f = np.sqrt((1 - b2**(epoch+1)))/(1 - b1**(epoch+1))
        Fl = F(s_prec, v_prec, f, eta, grad, b1, b2, epsilon)
        s += eta * Fl[0]
        v += eta * Fl[1]
        x += eta * Fl[2]
        x_prec = x
        v_prec = v
        s_prec = s

      x0 = x_prec
      v0 = v_prec
      s0 = s_prec
      epoch+=1

    if epoch == maxEpoch:
        print(str(num)+" max epoch reached")
    print(num, x, epoch) 
    return x, epoch


def eS_adam(x0,m,lr_init,eps,maxEpoch,typeR, num):
    epoch=0
    P=m
    v0 = 0
    s0 = 0
    x  = x0
    v  = v0 
    s   = s0
    b1   = 0.9
    b2=0.999
    epsilon=1e-8
    bb   = 1.0
    eta  = 1.e-15
    grad = 0.
    grad_tab=np.zeros(m)
    for i in range(m):
       grad_tab[i] = gradR(x,i,typeR)
    g  =  np.sum(grad_tab)
    g2 = g**2

    F_tab =np.array([[0.,0.,0.],[0.,0.,0.]])
    gNorm = 1000
    Fl =np.array([0.,0.,0.])

    while epoch < maxEpoch and gNorm/P > eps:
      if epoch > 0:
        eta=lr_init; 
      x_prec=x0
      v_prec=v0
      s_prec=s0
      for i in range(m):
        grad=gradR(x,i,typeR)  
        g  -= grad_tab[i]
        g2 -= grad_tab[i]**2
        g += grad
        g2+= grad**2
        g2 = max(g2,0) # sans ça, g2 peut passer négatif
        f = np.sqrt((1 - b2**(epoch+1)))/(1 - b1**(epoch+1))
        Fl = np.array(Fi(s_prec, v_prec, f, eta, g, g2, b1, b2, epsilon, m))
        s += eta * Fl[0]
        v += eta * Fl[1]
        x += eta * Fl[2]
        grad_tab[i] = grad
        x_prec = x
        v_prec = v
        s_prec = s

      x0 = x_prec
      v0 = v_prec
      s0 = s_prec
      epoch+=1
      gNorm=np.linalg.norm(g)
    if epoch == maxEpoch:
        print(str(num)+" max epoch reached "+str(g)+"")
    print(num, x, epoch) 
    return x, epoch

def exs(nbPoints, nbParticules, lr_init, eps, maxEpoch, typeCI):
    x_maille=np.linspace(a,b,num=nbPoints)
    u=np.zeros((nbPoints,1))

    fig = plt.figure(figsize=(10,10))
    plt.gcf().subplots_adjust(wspace=0.2,hspace=0.57)
    axes = fig.subplots(nrows=4,ncols=2)

    lr_init0 = lr_init

    for (typeR,m,L1,L2) in [("ex1",2, 2, 8), ("ex2",2, 22, 150), ("ex3",2, 1000, 0), ("ex4",2, 30, 0), ("ex5",3, 20, 0), ("ex6",3, 100, 0), ("ex7",3, 100, 0), ("ex8",4, 20, 20 )]:


        lr_init = lr_init0 #1./ (L1+L2) * lr_init0

        print(typeR)
        t0 = time.time()
        for p in range(nbParticules):
            x_unif=np.random.uniform(a,b); x0=x_unif*np.ones(N)
            
            #x,epoch = adam_fb(x0,m,lr_init,eps,maxEpoch,typeR, p)
            #x,epoch = adam   (x0,m,lr_init,eps,maxEpoch,typeR, p)
            x,epoch = eS_adam(x0,m,lr_init,eps,maxEpoch,typeR, p)

            if(x>a and x<b):
                u[int(nbPoints*(x[0]-a)/(b-a)),0]+=u0(x0,typeCI)
        u/=nbParticules

        if(typeR=="ex1"):
            ax=axes[0,0]; titre="a)"
            ax.axvline(x=0.4,ymin=0,ymax=1.1,color='green')
        elif(typeR=="ex2"):
            ax=axes[0,1]; titre="b)"
            ax.axvline(x=0.5,ymin=0,ymax=1.1, color='blue')
            ax.axvline(x=-0.714285714,ymin=0,ymax=1.1, color="green")
        elif(typeR=="ex3"):
            ax=axes[1,0]; titre="c)"
            ax.axvline(x=0,ymin=0,ymax=1.1, color='blue')
            ax.axvline(x=0.7210246,ymin=0,ymax=1.1, color="green")
        elif(typeR=="ex4"):
            ax=axes[1,1]; titre="d)"
            ax.axvline(x=0.6771,ymin=0,ymax=1.1, color='blue')
            ax.axvline(x=-0.812540,ymin=0,ymax=1.1, color="green")
        elif(typeR=="ex5"):
            ax=axes[2,0]; titre="e)"
            ax.axvline(x=-0.444,ymin=0,ymax=1.1,color='green')
        elif(typeR=="ex6"):
            ax=axes[2,1]; titre="f)"
            ax.axvline(x=0.5,ymin=0,ymax=1.1, color='blue')
            ax.axvline(x=-0.71806,ymin=0,ymax=1.1, color="green")
        elif(typeR=="ex7"):
            ax=axes[3,0]; titre="g)"
            ax.axvline(x=-0.91228,ymin=0,ymax=1.1, color="green")
        elif(typeR=="ex8"):
            ax=axes[3,1]; titre="h)"
            ax.axvline(x=0.413212,ymin=0,ymax=1.1, color='blue')
            ax.axvline(x=-0.870466,ymin=0,ymax=1.1, color="green")

        ax.set_title(titre)
        ax.set_xlabel(r"$\theta$")
        #locs, labels = ax.set_xticks()
        ax.set_xticks(np.arange(a, b, step=0.25))
        ax.plot(x_maille, u, label=r"$u$", color="black", ms=5, marker='.')
        ax.legend()
        t1 = time.time()
        print(f"time = {t1-t0:e}")
 
    plt.savefig('eS_adam_exs.pgf')
    plt.show()

typeCI="uniform"
N=1
nbPoints=1000
nbParticules= 10000
lr_init=0.001 #25
eps=10**(-4); 
maxEpoch=10000

exs(nbPoints,nbParticules,lr_init,eps,maxEpoch,typeCI)
