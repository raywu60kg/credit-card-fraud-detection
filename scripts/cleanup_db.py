import psycopg2
import pandas as pd

table_names = ["train_identity", "train_transaction", "test_identity", "test_transaction"]
conn = psycopg2.connect(
    database="database",
    user="user",
    password="password",
    host="localhost",
    port="5432")

print("Opened database successfully")

cur = conn.cursor()
def drop_table(table_name):

    try:
        cur.execute('''drop table {};'''.format(table_name))
    except Exception as e:
        print("Error in drop table {}: {}".format(table_name, e))

for table_name in table_names:
    drop_table(table_name=table_name)
print("Table deleted successfully")

conn.commit()
conn.close()