from pqueue import Queue
import paho.mqtt.client as mqtt
import threading
import time
import subprocess as sp
import sys
from pysondb import getDb

import psutil
import os
from datetime import datetime



def current_milli_time():
  return round(time.time() * 1000)

def INSERIR_PrimeiroModelo(cpu, ram, delay, qtd_pacotes, vazao):        #(tempo,cpu,ram,delay,qtd_pacotes,vazao):
  #print('agora')
  #connection = client.connect("http://localhost:4200", username="crate")
  #cursor = connection.cursor()
  #tempo=current_milli_time()
  workingDB = getDb('workingDB.json')
  new_item = {'cpu': cpu, 'ram': ram, 'delay': delay, 'qtd_pacotes': qtd_pacotes, 'vazao': vazao}
  workingDB.add(new_item)
  #cursor.execute('INSERT INTO amostras (tempo, cpu, ram, delay, qtd_pacotes, vazao) VALUES ('+str(tempo)+','+str(cpu)+',' +str(ram)+',' +str(delay)+',' +str(qtd_pacotes)+',' +str(vazao)+')') 




def get_CPU_Usage():
    # Calling psutil.cpu_precent() for 4 seconds
    #print('The CPU usage is: ', psutil.cpu_percent(1))
    return psutil.cpu_percent(1)


def get_Memory_Usage():
    # Getting all memory using os.popen()
    total_memory, used_memory, free_memory = map(
	int, os.popen('free -t -m').readlines()[-1].split()[1:])

    # Memory usage
    #print("RAM memory % used:", round((used_memory/total_memory) * 100, 2))
    return round((used_memory/total_memory) * 100, 2)


# Getting % usage of virtual_memory ( 3rd field)
#print('RAM memory % used:', psutil.virtual_memory()[2])



def get_bandwidth():
    # Get net in/out
    net1_out = psutil.net_io_counters().bytes_sent
    net1_in = psutil.net_io_counters().bytes_recv

    time.sleep(1)

    # Get new net in/out
    net2_out = psutil.net_io_counters().bytes_sent
    net2_in = psutil.net_io_counters().bytes_recv

    # Compare and get current speed
    if net1_in > net2_in:
        current_in = 0
    else:
        current_in = net2_in - net1_in

    if net1_out > net2_out:
        current_out = 0
    else:
        current_out = net2_out - net1_out

    network = {"KB/s  traffic_in" : current_in/1024, "traffic_out" : current_out/1024}
    return current_in/1024 


def get_TimeStampMS():
    # current date and time
    now = datetime.now()

    timestamp = datetime.timestamp(now)
    #print("timestamp =", timestamp)
    #print("timestamp =", round(timestamp*1000))

def getTempoInicioPacote():
    pass

def getListaAtrasoAteFog():
    #ler todos os pacotes
    #extrair os timestamps de criação
    #ler todos os 
    listaAtrasoAteFog =[] 
    return ini-fini
    
###################  #  
############ ######   #################
##################################    




nomeArquivo=sys.argv
print ('nome do arquivo:',str(nomeArquivo[1]))
q = Queue(nomeArquivo[1])



def on_connect(cliente, userdata, flags, rc):
  print("Connected with result code "+str(rc))
  cliente.subscribe("#")

def on_message(cliente, userdata, msg):
    q.put(current_milli_time() - int(msg.payload.decode()[-13:])) 		 
    #client.disconnect()
    
def thread_function(lista):
    #indice=0
    #print(indice)
	
    while True:
      time.sleep(3)
      #delay = q.get()
      qtd_pacotes = q.qsize();
      cpu = get_CPU_Usage()
      ram = get_Memory_Usage()
      vazao = get_bandwidth()
      #print("tamanho é: "+str(qtd_pacotes))
      soma=0
      media=0
      if (qtd_pacotes>1):
        for a in range(qtd_pacotes):
          soma = soma + int(q.get())
        media= soma/qtd_pacotes
        #print(cpu, ram, media, qtd_pacotes, vazao)
        INSERIR_PrimeiroModelo(cpu, ram, media, qtd_pacotes, vazao)
      elif ((qtd_pacotes==0)):  
        INSERIR_PrimeiroModelo(cpu, ram, 0, 0, vazao)

x = threading.Thread(target=thread_function, args=(0,))
x.start()
cliente = mqtt.Client()
cliente.connect("127.0.0.1",1883,60)

cliente.on_connect = on_connect
cliente.on_message = on_message

cliente.loop_forever()
