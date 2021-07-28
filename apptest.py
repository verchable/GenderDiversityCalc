import os
from flask import Flask, render_template, request, redirect, url_for
from run_edi import talking_time_count,gender_time_count
from run_edi import main
import time

import threading
# from yolov5.detect import get_opt, obj_detect  #added for detect convert
app = Flask(__name__)

begin = 0
stop = 0
print_result = 'this will take a while'
file_path = ''
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST','GET'])

def upload_file(): 
    global begin
    global print_result
    global gender_result
    global file_path
    global stop
    if begin == 0:  
        stop = 0    #first
        print_result = 'this will take a while'
        uploaded_file = request.files['file']
        basepath = os.path.dirname(__file__)
        if uploaded_file.filename != '':
            file_path = os.path.join(
                basepath, 'data',uploaded_file.filename)
            uploaded_file.save(file_path)
        begin = 1
        t1 = threading.Thread(target=test1)  
        t1.start()                   
        return render_template('result.html',name = print_result, stop=stop)  
    
def test1(): 
    global begin
    if begin == 1:
        global file_path
        global print_result
        global gender_result
        global stop
        command_line = 'python run_edi.py --video '+file_path
        os.system(command_line)
        print_result= talking_time_count (file_path)
        gender_result =gender_time_count(file_path)
        print_result = print_result+gender_result
        print (print_result)

        Percent = open("percent.txt", mode = "w") 
        Percent.write('0%')
        Percent.close()
        print ('analysis finished, printing on html')
        stop = 1
        begin = 0
        #return render_template('result.html',name= print_result, stop=stop)

@app.route('/result', methods=['POST','GET'])  
def result():
    global print_result
    global gender_result
    global stop
    if stop == 0:
        Percent = open("percent.txt", mode = "r") 
        print_result = Percent.read()
        Percent.close()    
    return render_template('result.html',name= print_result, stop=stop)


    
if __name__ == '__main__':
	app.run(port=9999,debug=True,host='0.0.0.0' )