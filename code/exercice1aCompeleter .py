# -*- coding: utf-8 -*-
"""
Exercice 1
@author: barthes
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.matlab as mio
from pomegranate.distributions import Categorical
from pomegranate.hmm import DenseHMM


from duree import duree
#%%
    
    
gamma=0.95
alpha=0.65 
beta=0.02 
Nsamples = 10000       # Nombre de valeurs d'une séquence créées à partir de la méthode sample 
return_states=True      # Retourne la séquence des états et la séquence des observables


start_probability = np.array([gamma, 1-gamma])
T = [
  [],
  []
]

B = [
  [],
  []
]


#%% Creation de la chaine de Markov
d0 = Categorical([B[0]])        # Distributions discrètes associées aux 2 états
d1 = Categorical([B[1]])

model = DenseHMM([d0, d1], edges=T, starts=start_probability,ends=[1e-20,1e-20],sample_length=Nsamples,return_sample_paths=return_states)   # définition du modèle de Markov


#%% Génération d'une séquence de longueur Nsamples
obsSeq,statesSeq = model.sample(1)                   # On génére 1 séquence de longueur Nsamples
obsSeq= obsSeq[0][:,0]                               # on recupere la sequence des observations (qui est dans l'exercice 1 identique à celle des etats )

#%%Calcul des histogrammes normalisées périodes Seches et pluvieuses sur la serie simulée
dSec,dPluie,pdfSec,pdfPluie,binsSec,binsPluie=duree(obsSeq)

#%% Etude des périodes pluie série expérimentale
ObsMesure = mio.loadmat('./RR5MN.mat')['Support'].astype(np.int8).squeeze()-1
#Calcul des histogrammes normalisées périodes Seches et pluvieuses sur la serie simulée
dSecMes,dPluieMes,pdfSecMes,pdfPluieMes,binsSecMes,binsPluieMes=duree(ObsMesure)

