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

def silhuetaKMEANS(df):
  vAux=[]
  sum_squared_dist=[]
  for k in range (2,11):
    warnings.filterwarnings("ignore")
    model = KMeans(n_clusters=k).fit(df) #, threshold=0.25
    #print(model.labels_)
    #sum_squared_dist.append(model.inertia_)
    
    media_silhueta= calinski_harabasz_score(df,model.labels_)
    vAux.append(media_silhueta)
    #print(k)
  print(vAux)
  print('numero de clusters: ',vAux.index(max(vAux))+2)
  i=vAux.index(max(vAux))
  #print('sq:', sum_squared_dist)
  return i+2

def excluirColuna0(df):
  
  df.drop('0',axis='columns', inplace=True)
  df.rename(columns = {1:0}, inplace = True)
  df.rename(columns = {2:1}, inplace = True)
  df.rename(columns = {3:2}, inplace = True)
  df.rename(columns = {4:3}, inplace = True)
  df.rename(columns = {5:4}, inplace = True)
  return df


def coletarModa(listaPrevisto):
  return statistics.mode(listaPrevisto)
  

###################
def compararNORMALcomDominante(NORMAL, moda, listaPrevisto):
  count=0
  tamanho =len(listaPrevisto)
  for i in range (0,tamanho):
    if (listaPrevisto[i]==moda):
      count=count+1
  print(count/tamanho)
  if ((count/tamanho)>=0.51):
    print("comportamento da amostra é:", moda)
    if (NORMAL==moda):
      print('OK')
    else:
      print('anomalia?')
  else:
    print("estranho")
    

def gerarModelo_KMeans():
  #connection = client.connect("http://localhost:4200", username="crate")
  #cursor = connection.cursor()

  #cursor.execute("SELECT * FROM fogTabela ORDER BY 1 ASC")
  #df = pd.DataFrame.from_records(cursor.fetchall())
  df = pd.read_csv ('A_r3.csv')
  df= excluirColuna0(df)#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  #print(df)#one())

  minmax_scale=preprocessing.MinMaxScaler()
  np_df = minmax_scale.fit_transform(df)

  df = pd.DataFrame(np_df)
  numeroClusters = silhuetaKMEANS(df)
  modelo = KMeans(n_clusters=numeroClusters).fit(df) #,threshold=0.15 , random_state=0
  print('centros: ',modelo.cluster_centers_)
  client1 = mqtt.Client()
  client1.connect("localhost",1883,60)
  
  arr1 = (modelo.cluster_centers_).tolist()
  #string=str(arr1)
  #li = re.split(r'[, \t]+', string)
  #print(arr1)
  
  #print(type(modelo.cluster_centers_))
  client1.publish("in", str(arr1))
  client1.disconnect()
  
  
  
  
  print(modelo.labels_)
  my_df = pd.DataFrame(modelo.labels_)
  my_df.to_csv('b3.csv', index=False)
  print('moda é: ',statistics.mode(modelo.labels_))
  #pickle.dump(model, open("modelo.pkl", "wb"))
  #model = pickle.load(open("modelo.pkl", "rb"))

  newrow = [0,0,0,0,0]
  np_df = np.vstack([np_df, newrow])
  newrow = [1,1,1,1,1]
  np_df = np.vstack([np_df, newrow])

  dfAux = pd.DataFrame(np_df)
  ######## a
  nptotal = minmax_scale.inverse_transform(dfAux)
  #dfinal = pd.DataFrame(nptotal)

  a={0:nptotal[-2][0],1:nptotal[-2][1],2:nptotal[-2][2],3:nptotal[-2][3],4:nptotal[-2][4]}
  b={0:nptotal[-1][0],1:nptotal[-1][1],2:nptotal[-1][2],3:nptotal[-1][3],4:nptotal[-1][4]}
  #print('a:', a)
  return modelo,a,b

#############
###############
##################

#modelo feito e também o max e min que eu preciso por nas amostras de entrada

def classificaComportamentoAmostra_KMeans(amostra,model,a,b):
  
  df2= excluirColuna0(amostra)#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#pd.read_csv(amostra)##<<APENDICE A antes<<<<<<<<<< tem que inserir 1 linha com valores mínimos e 1 linha com valores máximos do dataset de treinamento
  #print('amostra\n',df2)
  #print('eeeeeeee',a, type(a))
  df2 = df2.append(a, ignore_index=True)
  df2 = df2.append(b, ignore_index=True)
  #print('aqui')
  #print(df2)
  
  minmax_scale=preprocessing.MinMaxScaler()
  np_df2 = minmax_scale.fit_transform(df2)
  df2 = pd.DataFrame(np_df2)

  entrada = pd.DataFrame(df2)

  previsãoKMeans = model.predict(entrada)
  
  #print(previsãoKMeans[:-2])
  #print(type(previsãoKMeans[:-2]))
  return previsãoKMeans[:-2]

def selectAmostras():
  df = pd.read_csv ('d.csv')
  #print('amostras\n',df)
  return df









###########################################
#######################################
####################################

#PREDIÇÃO DE TODAS AS AMOSTRAS DE UM ROUND BASEADA NOS CENTRÓIDES

################################
###########################

def predictConvencional(model,amostra,a,b):

  #amostra = 'compA0.csv'
  
  df2= amostra
  #print(df2)
   #excluirColuna0(amostra)#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#pd.read_csv(amostra)##<<APENDICE A antes<<<<<<<<<< tem que inserir 1 linha com valores mínimos e 1 linha com valores máximos do dataset de treinamento
  #print('amostra\n',df2)
  #print('eeeeeeee',a, type(a))
  #print('tipo:', type(a))
  #print(a[1])
  dfa = {'0': a[0], '1': a[1], '2': a[2], '3': a[3], '4': a[4]}
  dfb = {'0': b[0], '1': b[1], '2': b[2], '3': b[3], '4': b[4]}
  
  dfr=pd.DataFrame(columns=['0', '1', '2','3','4'])
  dfr = dfr.append(dfa, ignore_index = True)
  dfr = dfr.append(dfb, ignore_index = True)
  #print(dfr)
  #a=pd.from_dict(a)
  #aux.DataFrame(a,columns = ['0','1','2','3','4'])
  #aux= aux.append(a, ignore_index=True, sort=False)
  #print('antes:',aux)
 
 # df2.insert(49, "0", [50, 40])
  
  #df2 = df2.append(b, ignore_index=True)
  #print('aqui')
  #print(df2)
  
  frames = [df2, dfr]

  df2 = pd.concat(frames)
  
  #print('antes:', )
  minmax_scale=preprocessing.MinMaxScaler()
  np_df2 = minmax_scale.fit_transform(df2)
  df2 = pd.DataFrame(np_df2)

  entrada = pd.DataFrame(df2)
  
  #print('depois:', entrada)

  
  previsãoKMeans = model.predict(entrada)
  
  #print(previsãoKMeans[:-2])
  #print(type(previsãoKMeans[:-2]))
  return previsãoKMeans[:-2]




def MeuPredict(centroides,amostra,model,a,b):

  #amostra = 'compA0.csv'
  
  df2= amostra

  dfa = {'0': a[0], '1': a[1], '2': a[2], '3': a[3], '4': a[4]}
  dfb = {'0': b[0], '1': b[1], '2': b[2], '3': b[3], '4': b[4]}
  
  dfr=pd.DataFrame(columns=['0', '1', '2','3','4'])
  dfr = dfr.append(dfa, ignore_index = True)
  dfr = dfr.append(dfb, ignore_index = True)
  #print(dfr)
  #a=pd.from_dict(a)
  #aux.DataFrame(a,columns = ['0','1','2','3','4'])
  #aux= aux.append(a, ignore_index=True, sort=False)
  #print('antes:',aux)
 
 # df2.insert(49, "0", [50, 40])
  
  #df2 = df2.append(b, ignore_index=True)
  #print('aqui')
  #print(df2)
  #print('oi:',df2 )
  frames = [df2, dfr]

  df2 = pd.concat(frames)
  
  #print('antes:', )
  minmax_scale=preprocessing.MinMaxScaler()
  np_df2 = minmax_scale.fit_transform(df2)
  df2 = pd.DataFrame(np_df2)

  entrada = pd.DataFrame(df2)
  
  #print('depois:', entrada)
  #print(entrada)
  previsãoKMeans=[]
  for i in range (len(df2)-2):
    menoresDistancias=[]
    for j in range (len(centroides)):
        #print(entrada.loc[1])
        dst = distance.sqeuclidean(entrada.loc[i], centroides[j])
        #print('distância da amostra:',i,', com cluster:',j,'=',dst)
        menoresDistancias.append(dst)
    previsãoKMeans.append(menoresDistancias.index(min(menoresDistancias)))
    #print(i)
  
  #previsãoKMeans = model.predict(entrada)
  
  #print(previsãoKMeans[:-2])
  #print(type(previsãoKMeans[:-2]))
  return previsãoKMeans




centros_aposR0=[[0.00852617754, 0.042847980800000005, -1.387778779e-16, 3.885780575e-16, 0.03218278645], ##A0 = B0, idx 0
		 [ 1.07843137e-02  ,6.10416667e-01 , 3.30210817e-01,  6.66203704e-01,   9.84888812e-02],  ## A1,    idx 1
		 [ 9.43450980e-01  ,9.44736842e-01  ,3.28294849e-01 , 6.64682540e-01,   8.71142423e-02],  ## A2     idx 2
		 [ 8.55381733e-01 , 5.54088050e-01  ,2.52949895e-01,  7.95674603e-01,   2.07351925e-01]]   ## B1     idx 3
		 
		 
centros_aposR1=[[0.00852617754, 0.042847980800000005, -2.77555755e-17, 9.992007225e-16, 0.03218278645],   ##A0 = B0, idx 0
 		[1.00898693e-02, 5.92562135e-01 ,3.29006704e-01 ,6.63690476e-01, 9.93407213e-02],        ## A1,    idx 1
 		[9.43450980e-01, 9.44736842e-01 ,3.28294849e-01, 6.64682540e-01,8.71142423e-02],        ## A2     idx 2
 		[ 8.56900567e-01 , 5.68461740e-01,  2.52398296e-01 , 7.93948413e-01, 2.04726189e-01]]    ## B1     idx 3
 		
 		
 		
centros_aposR2=[[0.008408697395, 0.03855126515, -3.7470027100000002e-16, -5.82867088e-16, 0.0238272328],#A0 = B0, idx 0
[1.02941176e-02  ,5.21782946e-01  ,3.29799648e-01 , 4.12805281e-01,6.17531447e-02],  # A1,    idx 1
[9.43450980e-01 , 8.34883721e-01,  3.28294849e-01  ,4.14603960e-01,5.54258999e-02],  # A2     idx 2
[8.30196078e-01  ,9.85788114e-01,  3.37187359e-01 , 8.28107811e-01,1.03400264e-01],   #(a3), mas seria B1  idx 3
[8.26434141e-01,  5.83052561e-01,  2.51311274e-01 , 7.95096372e-01, 2.04399461e-01]]  ## (a3)B1 idx 4
		 

centros_aposR3= [[0.007970588235, 0.032758447089, 2.220446047e-16, -3.330669075e-16, 0.018458906005], ##A0 = B0, idx 0
		  [ 1.00912779e-02,  5.12997862e-01  ,2.90794175e-01  ,4.14475930e-01,5.79673068e-02],  ## A1,    idx 1
    		  [ 9.43450980e-01 , 8.34883721e-01 , 2.87373886e-01 , 4.14603960e-01, 5.54258999e-02], ## A2     idx 2
 		  [ 8.30196078e-01  ,9.85788114e-01  ,2.95157971e-01 , 8.28107811e-01, 1.03400264e-01], ## (a3) mas seria B1 idx 3
 		  [ 7.06114973e-01 , 1.04173671e-02 , 2.51264904e-01  ,5.92786438e-01, 8.58952823e-02], # (a3) B1      idx 4
		  [ 1.00000000e+00 , 8.93977498e-01 , 2.44446442e-01  ,8.89361702e-01, 1.86508848e-01]] # B2      idx 5
 		  

centros_aposR4=[[0.0071625000000000005, 0.001055708554, 8.32667269e-17, 5.551115150000001e-17, 0.018458906005],##A0 = B0, idx 0
		[8.56217617e-03 , 1.44889723e-02,  2.89943003e-01,  3.11847498e-01, 5.77441766e-02],# A1,    idx 1
		[8.01933333e-01 , 2.43472364e-02,  2.87373886e-01 , 3.12500000e-01,5.54258999e-02],# A2     idx 2
		[0.70589082, 0.01958267095, 0.2732114375, 0.590695032, 0.09464777315],## A3 = B1      idx 3
		[1.0, 0.9159749269999999, 0.313120218, 0.8847177905, 0.1516619195]]# A4 = B2      idx 4



centros_aposR5=[[0.007193452385, 0.0026072837440000002, 1.942890292e-16, 3.885780565e-16, 0.018902433955],#A0 = B0, idx 0
[8.56217617e-03  ,1.44889723e-02 , 2.89943003e-01 , 3.11847498e-01,5.77441766e-02],# A1,    idx 1
[8.01933333e-01 , 2.43472364e-02,  2.87373886e-01,  3.12500000e-01,5.54258999e-02],# A2     idx 2
[0.706411328, 0.01960258955, 0.2665503825, 0.5904790925, 0.09465522005],# A3 = B1      idx 3
[1.0, 0.9159749269999999, 0.313120218, 0.8847177905, 0.1516619195],# A4 = B2      idx 4
[4.02571429e-01 ,2.15359473e-02, 2.44354672e-01, 1.14829932e-01,4.58296484e-02], # B3  idx 5
[6.96428571e-01 ,9.73811100e-03 ,5.92917838e-01 ,5.81428571e-01,8.01596672e-02]] # idx 6 - B4 é um comportamento que foi clusterizado pelo modelo, ele deriva de poucas amostras de B1, e o K-Means o categorizou em um cluster por possuir o dobro do atraso de B1. 


centros_aposR6=[[0.007005952385, 0.0027934187540000002, 3.191891195e-16, -4.440892095e-16, 0.02152348705],#A0 = B0, idx 0
[ 8.56217617e-03 , 1.44889723e-02,  2.89943003e-01  ,3.11847498e-01,5.77441766e-02], #A1 idx 1
[ 8.01933333e-01 , 2.43472364e-02  ,2.87373886e-01 , 3.12500000e-01,5.54258999e-02], #A2 idx 2
[0.706411328, 0.01960258955, 0.2665503825, 0.5904790925, 0.09465522005],# A3=B1              idx 3
[1.0, 0.9159749269999999, 0.313120218, 0.8847177905, 0.1516619195],#      A4 = B2               idx 4
[0.40354613100000003, 0.027553324599999998, 0.2661719, 0.122558001, 0.041091315399999995], # A5 = B3 idx 5
[ 6.96428571e-01 , 9.73811100e-03  ,5.92917838e-01 , 5.81428571e-01,8.01596672e-02]] # B4 (que apareceu)  idx 6


centros_aposR7=[[0.007005952385, 0.002641862104, -2.498001805e-16, -3.33066907e-16, 0.02152348705], # A0=B0 idx 0
[8.56217617e-03 , 1.36119717e-02 , 2.89943003e-01 , 3.09537517e-01,   5.77441766e-02],  #A1 i 1
[8.01933333e-01 , 2.28735266e-02  ,2.87373886e-01  ,3.10185185e-01,   5.54258999e-02],  #A2  i 2
[0.7084041795, 0.020473850000000002, 0.267020156, 0.586059529, 0.0899016219],        #A3=B1   i3
[1.0, 0.882304097, 0.2902688845, 0.868734068, 0.127607979],                          #A4=B2   i4
[0.4030966345, 0.02726843845, 0.2672810155, 0.120677763, 0.03662423355],               #A5=B3  i5
[ 6.96428571e-01,  9.73811100e-03 , 5.92917838e-01  ,5.73778195e-01,   8.01596672e-02]] #B4 i6


#------------------------------------------------------
#--------------- GO !! -----------------------------------------
#-----------------------------------------------------------
n=0
while (n<1):
  #numeroClusters=4
  #time.sleep(1)
  print('inicio mod:',current_milli_time())
  Tinicio=current_milli_time()
  model,a,b = gerarModelo_KMeans()
  Tfim=current_milli_time()
  print('final mod:',current_milli_time())
  print('tempo: ',Tfim-Tinicio)
  
  

  amostra = pd.read_csv ('testePredict_A_r4.csv')
  
  #'''  
  for z in range (len(amostra)):
     p=MeuPredict(centros_aposR3,amostra.loc[[z]],model,a,b)
     print(p)
  #'''
  n=n+1

  '''
  print('convencional')
  for z in range (len(amostra)):
     p=predictConvencional(model,amostra.loc[[z]],a,b)
     print(p)
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

