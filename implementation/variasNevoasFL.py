from pqueue import Queue
import paho.mqtt.client as mqtt
import threading
import time
import subprocess as sp
import sys
import re

def on_connect(client, userdata, flags, rc):
  print("Connected with result code "+str(rc))
  client.subscribe("in")

def on_message(client, userdata, msg):
    q.put(msg.payload.decode()) 		 
    #client.disconnect()



#from crate import client
from scipy.spatial import distance

import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn import preprocessing
from sklearn.metrics import calinski_harabasz_score #calinski_harabasz_score#davies_bouldin_score#silhouette_score
import warnings
import statistics
import time

def current_milli_time():
  return round(time.time() * 1000)


def gerarCarga(qtdCentroides):

  arr1=[[9.22023810e-03  ,4.70463751e-03, -1.94289029e-16 ,-1.11022302e-16,   3.01163589e-02],
  [ 8.56217617e-03 , 1.36119717e-02 , 2.89943003e-01 , 3.09537517e-01,   5.77441766e-02],
  [ 8.01933333e-01 , 2.28735266e-02  ,2.87373886e-01  ,3.10185185e-01,   5.54258999e-02],
  [ 7.09557692e-01 , 3.00683706e-02 , 2.96224068e-01 , 6.22364672e-01,   9.45828632e-02],
  [ 1.00000000e+00 , 8.72546377e-01,  3.19112383e-01,  9.15099715e-01, 1.30327672e-01],
  [ 4.00562500e-01 , 3.32112138e-02 , 2.90016197e-01 , 1.28703704e-01,   3.11402195e-02]]
  
  arr2=[[ 4.79166667e-03,  5.79086698e-04, -3.05311332e-16, -5.55111512e-16 ,  1.29306152e-02],
  [ 7.07250667e-01,  1.08793294e-02 , 2.37816244e-01,  5.49754386e-01,   8.52203806e-02],
  [ 1.00000000e+00  ,8.92061817e-01,  2.61425386e-01  ,8.22368421e-01,   1.24888286e-01],
  [ 4.05630769e-01  ,2.13256631e-02 , 2.44545834e-01 , 1.12651822e-01,   4.21082476e-02],
  [ 6.96428571e-01,  9.73811100e-03 , 5.92917838e-01  ,5.73778195e-01,   8.01596672e-02]]
  
  client1 = mqtt.Client()
  client1.connect("127.0.0.1",1883,19000)
  qtdCentroides=int(qtdCentroides/2)
  for i in range (qtdCentroides):
  	client1.publish("in", str(arr1))
  	client1.publish("in", str(arr2))
  client1.disconnect()


#############
###############
##################

#--------------- GO !! -----------------------------------------
#-----------------------------------------------------------
n=0
qtdCentroides=10000
while (n<1):
  #numeroClusters=4
  #time.sleep(1)
  print('inicio mod:',current_milli_time())
  Tinicio=current_milli_time()
  gerarCarga(qtdCentroides)
  Tfim=current_milli_time()
  print('final mod:',current_milli_time())
  print('tempo: ',Tfim-Tinicio)
  
  

  #amostra = pd.read_csv ('testePredict_A_r4.csv')
  
  '''  
  for z in range (len(amostra)):
     p=MeuPredict(centros_aposR3,amostra.loc[[z]],model,a,b)
     print(p)
  '''
  n=n+1

  '''
  print('convencional')
  for z in range (len(amostra)):
     p=predictConvencional(model,amostra.loc[[z]],a,b)
     print(p)
  '''





'''
arr1 = (model.cluster_centers_).tolist()
string=str(arr1)
li = re.split(r'[, \t]+', string)
#print(li)

linhas=int((len(li))/5)
centroids= [[0]*5 for i in range(linhas)]
#print(centroids)


k=0
for i in range (int(linhas)):
    for j in range (5):
        aux = li[k].replace('[[', '')
        aux = aux.replace(']', '')
        aux = aux.replace('[', '')
        aux = aux.replace(']]', '')
        print(aux)
        centroids[i][j]=aux
        k=k+1
test=(np.float_(centroids))
#print(test)
'''









'''
#amostra ='input3.csv'
dfAmostras = selectAmostras()


listaPrevisto = classificaComportamentoAmostra_KMeans(dfAmostras,model,a,b)
moda = coletarModa(listaPrevisto)

NORMAL= moda
print('primeira rodada =',NORMAL)
aux = selectAmostras()
#print(aux)
insertAmostras_naFogTabela_eLimpaAmostras(aux)
print('oi3')
numVezes=0
while (numVezes<30):
  time.sleep(180)
  INI=current_milli_time()
  print("LETS GO")
  #amostra ='input3.csv' if (numVezes==0) else 'inputTeste1.csv'
  dfAmostras = selectAmostras()
  listaPrevisto = classificaComportamentoAmostra_KMeans(dfAmostras,model,a,b)
  moda = coletarModa(listaPrevisto)
  if (numVezes==0):
    NORMAL= moda
    print('NORMAL =',NORMAL)
  compararNORMALcomDominante(NORMAL, moda, listaPrevisto)
  numVezes=numVezes+1
  aux = selectAmostras()
  insertAmostras_naFogTabela_eLimpaAmostras(aux)
  FINI=current_milli_time()
  print('t= ',FINI-INI)
  #LEMBRAR DE INSERIR CADA AMOSTRA NO CRATEDB

########################################################
'''

