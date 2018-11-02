#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# setting up modules used in the program
from __future__ import print_function
from dronekit import connect
import exceptions
import socket
import time
import os

# connect to the rover
os.system("clear")
vehicle = connect('/dev/ttyS0', heartbeat_timeout = 30, baud = 57600)
time.sleep(2)

# instruction
print("\nPress 'Ctrl + c' to quit.\n\n")
time.sleep(3)

# measure distance
while True:

    # reading from rangefinder
    rangefinder_distance = vehicle.rangefinder.distance
    # print out the reading from rangefinder
    print ("Rangefinder Distance: %.2f [m]" % float(rangefinder_distance))
    # 1 sec delay
    time.sleep(1)