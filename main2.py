import pymysql
import random
import datetime
"""
pymysql을 이용해서 localhost의 3308포트
계정 root
비밀번호 8073298c
db bssm

"""

conn = pymysql.connect(host='10.150.150.191', port=3307, user='djdn', password='djdn', db='djdn_warning', charset='utf8')
curs = conn.cursor()

def print_data():
    sql = "select * from tbl_density"
    curs.execute(sql)
    rows = curs.fetchall()
    print(rows)

def insert_data():
    # first = "2023-01-12"
    for i in range(1,59):
        time = datetime.datetime(2023, 1, 12, 19, i,0)
        people = random.randrange(1, 4)
        camera = 3
        print(time, people, camera)
        sql = f"insert into tbl_density(created_at, people,camera_id) values ('{time}',{people},{camera})"
        curs.execute(sql)
        conn.commit()


insert_data()
print_data()

