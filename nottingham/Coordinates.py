# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import json
import datetime
import webbrowser

with open('location-history.json', 'r') as f:
    location = json.load(f)

    
entry = 766600 # entry point of data
entryRange = entry + 2000 # amount of records to go through
timeRange = 100 # amount of space between each record returned

while entry <= entryRange:

    long = int(location["locations"][entry]["longitudeE7"])
    
    lat = int(location["locations"][entry]["latitudeE7"])
    
    long = long / 10**7
    
    lat = lat / 10**7
    
    date = datetime.datetime.fromtimestamp(int(location["locations"][entry]["timestampMs"])/1000.0)
    
    date = date.strftime('%Y-%m-%d %H:%M:%S')
    
    print(str(entry) + '\ndate: ' + str(date) + '\nlatitude: ' + str(lat) + '\nlongitude: ' + str(long) + '\n')
    
    url = 'https://www.google.com/maps/search/' + str(lat) + ',+' + str(long)
    
    webbrowser.open(url)
    
    entry += timeRange

# use open street map instead