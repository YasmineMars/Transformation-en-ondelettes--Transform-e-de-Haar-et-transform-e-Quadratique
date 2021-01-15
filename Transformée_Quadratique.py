#!/usr/bin/env python
# coding: utf-8

# In[207]:


import numpy as np
import math as math
import matplotlib.pyplot as plt


# Exercice 1

# In[208]:


# Fonction Quadratique

def Quad(n,u):
    v= np.zeros((n),float)
    v=np.copy(u)
    w=np.copy(u)
    m=n    
    vChap=np.zeros((n),float)
    print("CALCUL DE LA DIRECTE")
    print("\n")
    print("u=",u)
    print("\n")
    i=1
    while (m>1):
        w[0: int(m/2)] = (v[0:m-1:2]+ v[1:m:2])/2
        w[int(m/2): m] = (v[0:m-1:2]- v[1:m:2])/2  
        w[1: int(m/2)-1] = (v[2:m-2:2]+ v[3:m-1:2])/2
        if (m/2)>=2 :
             vChap[2:m-2:2]= w[1: int(m/2)-1] - (1/8) * (w[0:int(m/2)-2] - w[2: int(m/2)])
        w[1+int(m/2): m-1] = (v[2:m-2:2]- vChap[2:m-2:2])       
        v[0:m] = w[0:m]
        print("iteration {} : v= {}".format(i,v))
        print("\n")
        m= int(m/2)   
        i=i+1
    print("\n")
    print("Directe =",v)
    print("\n")
    return v

# Fonction Quadratique Inverse

def Quad_Inv(n,u):
    v= np.zeros((n),float)
    v=np.copy(u)
    w=np.zeros((n),float)
    m=1  
    vChap=np.zeros((n),float)
    i=1
    print("CALCUL DE L'INVERSE")
    print("u=",u)
    print("\n")
    while (m<n):
        w[0:2*m:2]= v[0:m]+v[m:2*m]
        w[1:2*m:2]= v[0:m]-v[m:2*m]
        if(m>=2) : 
            vChap[2:2*m-2:2] = v[1:m-1]-(1/8)*(v[0:m-2]-v[2:m])
            w[2:2*m-2:2]= vChap[2:2*m-3:2] + v[m+1 : 2*m-1]
            w[3:2*m-1:2]= 2*v[1:m-1] - w[2:2*m-2:2]
        v[0:2*m] = w[0:2*m]
        print("iteration {} : v= {}".format(i,v))
        print("\n")
        m=2*m
        i=i+1
    print("\n")
    print("Inverse =",v)
    print("\n")

    return v


# In[209]:


# Fonction Quadratique avec boucle FOR


def quad_for(n,u):
    v=np.copy(u)
    w=np.copy(u)
    m=n   
    vChap=np.zeros((n),float)
    print("CALCUL DE LA DIRECTE")
    print("\n")
    print("u=",u)
    print("\n")
    i=1
    while (m>1):  
        for k in range(0,int(m/2)):
            w[k] = (v[2*k]+v[2*k+1])/2
            w[k+int(m/2)] = (v[2*k]-v[2*k+1])/2
        
        for k in range(1,int(m/2)-1):
            vChap[2*k]= w[k]-(1/8)*(w[k-1]-w[k+1])
            w[k+int(m/2)]= v[2*k]-vChap[2*k]
        for k in range(0,m):
            v[k]=w[k]
        print("iteration {} : v= {}".format(i,v))
        print("\n") 
        m= int(m/2)
        i=i+1
    print("\n")
    print("Directe =",v)
    print("\n")    
    return v

# Fonction Quadratique Inverse avec boucle FOR

def quad_Inv(n,u):
    v=np.copy(u)
    w=np.copy(u)
    m=1    
    vChap=np.zeros((n),float)
    print("CALCUL DE L'INVERSE")
    print("u=",u)
    print("\n")
    i=1
    while (m<n):  
        for k in range(0,m):
            w[2*k] = v[k]+v[k+m]
            w[2*k+1] = v[k]-v[k+m]
        
        for k in range(1,m-1):
            vChap[2*k]= v[k]-(1/8)*(v[k-1]-v[k+1])            
            w[2*k]= vChap[2*k]+v[k+m]
            w[2*k+1]= 2*v[k]-w[2*k]

        for k in range(0,2*m):
            v[k]=w[k]
        print("iteration {} : v= {}".format(i,v))
        print("\n")  
        m= 2*m
        i=i+1
    print("\n")
    print("Inverse =",v)
    print("\n")    
    return v


# Exercice 2

# In[210]:


def ex1(n):
	x= np.zeros((n))
	for i in range (n):
		x[i] = i+1
	return x

def ex2(n,x):
    v= np.zeros((n))
    pi = math.pi
    for i in range(0,len(x)):
        if i < len(x)/2:  
            v[i]=math.sin(2*pi*x[i])
        else:
            v[i]=0.5+math.sin(2*pi*x[i])
    return v

def ex3(n):
	x= np.zeros((n))
	for i in range (int(n/2), n):
		x[i] = 0.5
	return x


# In[211]:


size = 64 #CHoisir une valeur égale a une puissance de 2

EX1 = ex1(size)

img1 = [78,15,47,96,12,45,125,147]
img2 = [78,15,47,96,12,45,125,147,12,58,78,45,12,12,47,15]
img3 = [78,15,47,96,12,45,125,147,12,58,78,45,12,12,47,15,48,78,210,45,87,95,21,36,78,147,63,14,25,35,14]
img4 = [78,15,47,96,12,45,125,147,12,58,78,45,12,12,47,15,48,78,210,45,87,95,21,36,78,147,63,14,25,35,14,78,111,46,85,21,74,36,52,52,52,14,174,175,177,68,71,68,74,7,6]
EX2 = ex2(size,img4) # Pour tester avec un autre size, on doit changer le numero de l'image en entrée(img1(size=8),img2(size(16)...))

EX3 = ex3(size)


# In[214]:


#Verification que ex1 = inverse(directe(ex1))

print("ex1=",EX1)
print("\n")
Quad_Inv(size,Quad(size,EX1))


# In[215]:


#Verification que ex3 = inverse(directe(ex3))

print("ex3=",EX3)
print("\n")
Quad_Inv(size,Quad(size,EX3))


# Exercice 3

# In[216]:


def sueillage(x,T):
	y= np.copy(x)
	y[np.absolute(y)<=T]=0
	return y


# In[217]:


quad1 = quad_for(size,EX1)
quad2 = quad_for(size,EX2)
quad3 = quad_for(size,EX3)


# In[218]:


print(sueillage(quad1,12))


# In[219]:


print(sueillage(quad1,128))


# In[220]:


print(sueillage(quad2,12))


# In[221]:


print(sueillage(quad2,128))


# In[222]:


print(sueillage(quad3,12))


# In[223]:


print(sueillage(quad3,128))


# In[224]:


#COMPARAISON AVEC HAAR :
"""
Les resultats du seuillage avec T=12 et T=128 sont les memes que ceux de Haar.


"""


# Exercice 4

# In[225]:


def normeL2(x,y):
    if len(x)!=len(y):
        print("Les deux vecteurs on des tailles différente")
    else:
        n=len(x)
        v=0
        for i in range(n):
            v=v + np.square(np.absolute(x[i]-y[i]))
        v= np.sqrt(v)
    return v

quad11 = quad_Inv(size,sueillage(quad1,12))
quad12 = quad_Inv(size,sueillage(quad1,128))

quad21 = quad_Inv(size,sueillage(quad2,12))
quad22 = quad_Inv(size,sueillage(quad2,128))

quad31 = quad_Inv(size,sueillage(quad3,12))
quad32 = quad_Inv(size,sueillage(quad3,128))


# In[226]:


print(normeL2(EX1,quad11))


# In[227]:


print(normeL2(EX1,quad12))


# In[228]:


print(normeL2(EX2,quad21))


# In[229]:


print(normeL2(EX2,quad22))


# In[230]:


print(normeL2(EX3,quad31))


# In[231]:


print(normeL2(EX3,quad32))


# In[232]:


#COMPARAISON AVEC HAAR :
"""
Les resultats du calcul des erreurs sont les memes que ceux de Haar, sauf pour l'erreur entre ex1 et quad11 qui est un peu plus 
supérieure de celle de Harr mais reste trés proche.

"""


# Exercice 5 

# In[233]:


t = ex1(128)
tab1 = np.zeros((128))

for i in range(128):
    tab1[i-1] = normeL2(EX1,Quad_Inv(size,sueillage(quad1,i)))


# In[234]:


plt.scatter(t,tab1)
plt.title('Qualité de reconstruction de quad l en fonction de sueillage')
plt.xlabel('Sueil')
plt.ylabel('Erreur')
plt.show()


# In[235]:


t = ex1(128)
tab2 = np.zeros((128))

for i in range(128):
    tab2[i-1] = normeL2(EX2,Quad_Inv(size,sueillage(quad2,i)))


# In[236]:


plt.scatter(t,tab2)
plt.title('Qualité de reconstruction de quad l en fonction de sueillage')
plt.xlabel('Sueil')
plt.ylabel('Erreur')
plt.show()


# In[237]:


t = ex1(128)
tab3 = np.zeros((128))

for i in range(128):
    tab3[i-1] = normeL2(EX3,Quad_Inv(size,sueillage(quad3,i)))


# In[238]:


plt.scatter(t,tab3)
plt.title('Qualité de reconstruction de quad 3 en fonction de sueillage')
plt.xlabel('Sueil')
plt.ylabel('Erreur')
plt.show()


# In[239]:


#COMPARAISON AVEC HAAR :
"""
Les graphes d'erreur en fonction du sueil sont exactement les memes que ceux de Haar.

"""


# In[ ]:




