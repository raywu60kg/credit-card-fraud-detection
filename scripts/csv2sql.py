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
cur.execute('''CREATE TABLE IRIS
    (ID INT PRIMARY KEY NOT NULL,
    SEPAL_LENGTH  REAL,
    SEPAL_WIDTH REAL,
    PETAL_LENGTH REAL,
    PETAL_WIDTH REAL,
    VARIETY TEXT);''')
print("Table created successfully")

data = pd.read_csv("data/iris.csv")
for idx in range(len(data)):
    cur.execute(
        (
            "INSERT INTO IRIS"
            "(ID, SEPAL_LENGTH, SEPAL_WIDTH,"
            " PETAL_LENGTH, PETAL_WIDTH, VARIETY)"
            "VALUES ({}, {}, {}, {}, {}, '{}')".format(
                idx,
                data["sepal.length"][idx],
                data["sepal.width"][idx],
                data["petal.length"][idx],
                data["petal.width"][idx],
                data["variety"][idx])))
print("Records created successfully")
conn.commit()
conn.close()