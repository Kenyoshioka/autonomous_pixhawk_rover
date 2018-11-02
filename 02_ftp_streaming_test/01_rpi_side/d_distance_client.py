#!/usr/bin/env python2
# -*- coding: utf-8 -*-  

# setting up modules used in the program
from socket import *
import socket
import time

# create a socket and bind socket to the host later
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
HOST = '192.168.2.105' # change '〇.〇.〇.〇' to user's PC IP address
PORT = 8002


# initialization
connected = False

# when server is not connected
while not connected:

    # bind socket to the host (when the server is connected)
    try:

        client_socket.connect((HOST, PORT))
        connected = True

    # loop (pass) back to while not connected (when the server is not connected)
    except Exception as e:

        pass

# send rangefinder measured distance
def send_reading(measured_distance):

    client_socket.send(measured_distance.encode())

# close the socket
def close_socket():

    client_socket.close()