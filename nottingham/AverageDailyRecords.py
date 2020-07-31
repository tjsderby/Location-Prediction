# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 09:35:37 2020

@author: prosk
"""
import json
import datetime

with open('location-history.json', 'r') as f:
    location = json.load(f)
    
entry = 766600 # entry point of data
entryRange = entry + 96887 # amount of records to go through

count = 0
arr = []

prvDate = datetime.datetime.fromtimestamp(int(location["locations"][entry]["timestampMs"])/1000.0)
    
prvDate = prvDate.strftime('%d')

while entry <= entryRange:
    
    date = datetime.datetime.fromtimestamp(int(location["locations"][entry]["timestampMs"])/1000.0)
    
    date = date.strftime('%d')
    
    if (prvDate == date):
        count += 1
        
    if (prvDate != date):
        arr.append(count)
        count = 0
        
    prvDate = date
    
    entry += 1
    
def Average(lst): 
    return sum(lst) / len(lst) 

average = Average(arr)

print(average)