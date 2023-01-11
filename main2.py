"""
pymysql을 이용해서 localhost의 3308포트
계정 root
비밀번호 8073298c
db bssm

"""

conn = pymysql.connect(host='localhost', port=3308, user='root', password='8073298c', db='bssm', charset='utf8')
curs = conn.cursor()

def print_data():
    sql = "select * from my_table"
    curs.execute(sql)
    rows = curs.fetchall()
    print(rows)

def insert_data():
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    c = random.randint(1, 100)
    d = random.randint(1, 100)
    sql = f"insert into my_table values (1,{a},{b},{c},{d})"
    curs.execute(sql)
    conn.commit()

insert_data()
print_data()

