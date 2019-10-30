import serial.tools.list_ports
from datetime import datetime
import time
import numpy as np
import csv

default_port = 'COM3'

def list_COMports():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        print(port)

def initialize(com_port=default_port):
    ser = serial.Serial(com_port, 115200)

    print(f'Arduino at {com_port} connected')

    return ser

def read_Arduino(com_port=default_port, fname='data.txt'):
    ser = initialize(com_port)

    try:
        while True:
            data = ser.readline()

            if data:
                now = str(time.time())
                with open(fname, 'ab+') as file:
                    line = now + ', ' + data.decode('utf-8')
                    file.write(line.encode('utf-8'))

    except KeyboardInterrupt:
        ser.close()
        pass

if __name__ == '__main__':
    read_Arduino()