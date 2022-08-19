from OSMPythonTools.nominatim import Nominatim
from OSMPythonTools.overpass import Overpass, overpassQueryBuilder
from OSMPythonTools.data import Data, dictRangeYears, ALL
from OSMPythonTools.cachingStrategy import CachingStrategy, JSON, Pickle
import csv

# CachingStrategy.use(Pickle, cacheDir='')

from collections import OrderedDict

# dimensions = OrderedDict([
#   ('year', dictRangeYears(2013, 2017.5, 1)),
#   ('city', OrderedDict({
#     'heidelberg': 'Heidelberg, Germany',
#     'manhattan': 'Manhattan, New York',
#     'vienna': 'Vienna, Austria',
#   })),
#   ('typeOfRoad', OrderedDict({
#     'primary': 'primary',
#     'secondary': 'secondary',
#     'tertiary': 'tertiary',
#   })),
# ])

dimensions = OrderedDict([
    ('city', OrderedDict({
        'boulder': 'Boulder, Colorado',
        # 'louisville': 'Louisville, Colorado',
    })),
    ('typeOfRoad', OrderedDict({
        'primary': 'primary',
        'secondary': 'secondary',
        'tertiary': 'tertiary',
        'service': 'service',
        'unpaved': 'unpaved',
        # 'residential': 'residential',
        # 'track': 'paved',
        # 'motorway': 'path',
        # 'footway': 'footway',
        # 'path': 'path',
        # 'cycleway': 'cycleway',
        # 'unclassified': 'unclassified'
    }))
])

nominatim = Nominatim()
overpass = Overpass()


def fetch(city, typeOfRoad):
    areaId = nominatim.query(city).areaId()
    query = overpassQueryBuilder(area=areaId, elementType='way', selector='"highway"="' + typeOfRoad + '"', includeGeometry=True)
    return overpass.query(query, timeout=120)


# areaId = nominatim.query('Boulder, Colorado').areaId()

# result = overpass.query(query)
# print(areaId)
data = Data(fetch, dimensions)
# dCSV = data.getCSV()
# f = open('BoulderOSM.csv', 'w', encoding='UTF8', newline='')
# writer = csv.writer(f)
# writer.writerows(dCSV)
# print(dCSV)

