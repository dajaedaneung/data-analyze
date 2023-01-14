from datetime import datetime

from flask import Flask , jsonify, request
from fastapi import FastAPI
from model1 import model3333
from model2 import model2
from model3 import model3
app1 = FastAPI()

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.get('/model1/<date>')
def model1(date):
    date = str(date)
    date_time_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    return model3333(date_time_obj)

@app.get('/model2/<date>')
def model2(date):
    date = str(date)
    date_time_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    return model3333(date_time_obj)

@app.get('/model3/<date>')
def model3(date):
    date = str(date)
    date_time_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    return model3333(date_time_obj)


if __name__ == '__main__':
    app.run()