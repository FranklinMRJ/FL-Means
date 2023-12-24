import sqlite3
conn = sqlite3.connect('clientes.db')
cursor = conn.cursor()

# inserindo dados na tabela
cursor.execute("""
INSERT INTO clientes (cpu, ram, delay, qtd_pacotes, vazao)
VALUES (12.2, 21, 52, 100, 3.24441)
""")
# gravando no bd
conn.commit()
print('Dados inseridos com sucesso.')



# lendo os dados
cursor.execute("""
SELECT * FROM clientes;
""")

for linha in cursor.fetchall():
    print(linha)

conn.close()
