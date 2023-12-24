from pqueue import Queue
import paho.mqtt.client as mqtt
import threading
import time
import subprocess as sp
import sys
import numpy as np
from scipy.spatial import distance

import re
nomeArquivo=sys.argv
print ('nome do arquivo:',str(nomeArquivo[1]))
q = Queue(nomeArquivo[1])


def current_milli_time():
    return round(time.time() * 1000)
  
def lerCentroids(entrada):
    #print('a: ',entrada)
    #arr1 = (entrada).tolist()
    string=(entrada)
    li = re.split(r'[,]', string)
    #print(li)
    #print(float(li[1]))
    #print([s for s in li if s != ''])
    
    
    linhas=int((len(li))/5)
    centroids= [[0]*5 for i in range(linhas)]
    #print()
    #print('\n\n\n\n\n')
    k=0
    for i in range (int(linhas)):
        for j in range (5):
            aux = li[k].replace('[[', '')
            aux = aux.replace(']', '')
            aux = aux.replace('[', '')
            aux = aux.replace(']]', '')
            aux = aux.replace('\n', '')
            #print(aux)
            centroids[i][j]=aux
            k=k+1
    res=(np.float_(centroids))
    return res.tolist()

def calcularNovosCentroides(fogA,fogB):
	#print('fogs>>>>>>>>> ', fogA, fogB)
	limiar=0.2
	menores = []
	for i in range (len(fogA)):
	    for j in range (len(fogB)):
	        dst = distance.euclidean(fogA[i], fogB[j])
	        if (limiar>dst):
	        	menores.append([dst,i,j])

	#print('\n')
	#listMedias=[]
	for w in range (len(menores)):
	    #print(menores[w])
	    #print('comportamentos A',menores[w][1], 'e B',menores[w][2],' são similares')
	    
	    i= menores[w][1]
	    j=menores[w][2]
	    l=[]
	    m0=( ((fogA[i][0]+fogB[j][0]) /2) )
	    m1=( ((fogA[i][1]+fogB[j][1]) /2) )
	    m2=( ((fogA[i][2]+fogB[j][2]) /2) )
	    m3=( ((fogA[i][3]+fogB[j][3]) /2) )
	    m4=( ((fogA[i][4]+fogB[j][4]) /2) )
	    
	    l=[m0,m1,m2,m3,m4]
	    #print(l)
	    fogA[i]=l
	    fogB[j]=l
	    #print('\n')
	    #listMedias.append(l)
	  #  print(str((fogA[i]+fogB[j])/2))
	#print('\n\n\n\n')
	#print(listMedias)
	listaCompConcatenados=[[]]

	listaCompConcatenados=fogA+fogB

	#print(listaCompConcatenados)

	res = []
	for i in listaCompConcatenados:
	    if i not in res:
	    	res.append(i)
	#print(res)
	return res

def on_connect(client, userdata, flags, rc):
  print("Connected with result code "+str(rc))
  client.subscribe("in")

def on_message(client, userdata, msg):
    q.put(msg.payload.decode()) 		 
    #client.disconnect()
    
def thread_function(lista):
    Tinicio=current_milli_time()
    while True:
      time.sleep(0.025)
      #print(q.qsize()) 
      status,result = sp.getstatusoutput("ping -c 1 -W 1 " + "127.0.0.1")
      if status == 0:
        if(q.qsize() >= 99): #fog nodes
          tamanho=q.qsize();
          #print("tamanho é: "+str(tamanho))
          centroAtual = lerCentroids(q.get())
          for a in range(tamanho-1):
            #print('atual: ',centroAtual)
            centroAtual = calcularNovosCentroides(centroAtual,lerCentroids(q.get()))
          client1 = mqtt.Client()
          client1.connect("127.0.0.1",1883,600)
          #print(centroAtual)
          clientN = mqtt.Client()
          clientN.connect("177.104.61.123",1883,600)
          for v in range(99):
            clientN.publish("out", str(centroAtual))
          clientN.connect("192.168.0.17",1883,600)
          clientN.publish("out", str(centroAtual))
          Tfim=current_milli_time()
          print(Tfim-Tinicio)
          client1.disconnect()
          Tinicio=current_milli_time()
      #else:
        #print("off")


x = threading.Thread(target=thread_function, args=(0,))
x.start()
client = mqtt.Client()
client.connect("127.0.0.1",1883,19000)

client.on_connect = on_connect
client.on_message = on_message

client.loop_forever()
