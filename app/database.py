import mysql.connector
from mysql.connector import Error

def get_db_connection():
    return mysql.connector.connect(
        host="192.168.0.121",
        user="mysql_121",
        password="hdcdatarit9esoydld]o8i",
        database="officedd_photo"
    )
