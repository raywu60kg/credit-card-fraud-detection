"""TODO"""
import psycopg2
import pandas as pd
import gc


conn = psycopg2.connect(
    database="database",
    user="user",
    password="password",
    host="localhost",
    port="5432")

print("Opened database successfully")

cur = conn.cursor()
cur.execute('''CREATE TABLE train_identity
    (ID INT PRIMARY KEY NOT NULL,
    SEPAL_LENGTH  REAL,
    SEPAL_WIDTH REAL,
    PETAL_LENGTH REAL,
    PETAL_WIDTH REAL,
    VARIETY TEXT);''')
print("Table train_identity created successfully")

cur.execute('''CREATE TABLE train_transaction
    (ID INT PRIMARY KEY NOT NULL,
    SEPAL_LENGTH  REAL,
    SEPAL_WIDTH REAL,
    PETAL_LENGTH REAL,
    PETAL_WIDTH REAL,
    VARIETY TEXT);''')
print("Table train_transaction created successfully")

data = pd.read_csv("data/train_identity.csv")
for idx in range(len(data)):
    cur.execute(

print("Records created successfully")
conn.commit()
conn.close()
