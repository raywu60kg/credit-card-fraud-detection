import psycopg2
import pandas as pd


conn = psycopg2.connect(
    database="database",
    user="user",
    password="password",
    host="localhost",
    port="5432")

print("Opened database successfully")

cur = conn.cursor()
cur.execute('''drop table train_transaction;''')
cur.execute('''drop table train_identity;''')
cur.execute('''drop table test_transaction;''')
cur.execute('''drop table test_identity;''')
print("Table deleted successfully")

conn.commit()
conn.close()