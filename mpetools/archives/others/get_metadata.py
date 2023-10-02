"""
This module allows to retrieve metadata from either an ee.Image or ee.ImageCollection.
FOR NOW: only retrieves dates.
TODO: allow the user to select other metadata.

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

import ee
import datetime
import pytz

def retrieveDate(ee_object):
    
    time = ee_object.get('system:time_start').getInfo()
    timestamp = datetime.datetime.fromtimestamp(time/1000, tz=pytz.utc)
    date = timestamp.strftime('%Y-%m-%d')

    return date
