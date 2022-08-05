import copy

from OSMPythonTools.nominatim import Nominatim
from OSMPythonTools.overpass import Overpass, overpassQueryBuilder
from OSMPythonTools.data import Data, dictRangeYears, ALL
import numpy as np
from collections import OrderedDict

# Box constraints

# Walker Ranch Box
# north = 39.96651
# south = 39.898609
# east = -105.30969
# west = -105.367494

# NIST Facility
north = 39.99989
west = -105.30153
south = 39.97685
east = -105.24541
grid_size = 0.01
# Decimal degrees conversion
# 1 deg = 111km, 0.1 deg = 11.1km, 0.01 deg = 1.1km, 0.001d = 111m, 0.0001d = 11.1m

# Latent Vector = lat, lon,
#  [highway]  primary, secondary, tertiary, service, unpaved, residential, track, path, footway, cycleway, unclassified,
#  [power]    power_line,
#  [waterway] river, stream, dam, ditch, drain, canal
#  [Building] roof, retail, apartment, construction, house, unclassified, school,
#  [natural]  bare_rock, wetland, water
#             railway
#

LVS_dict = {'lat': 0.0, 'lon': 0.0, 'primary': 0, 'secondary': 0, 'tertiary': 0, 'service': 0, 'unpaved': 0,
            'residential': 0, 'track': 0, 'motorway': 0, 'footway': 0, 'path': 0, 'cycleway': 0, 'unclass_highway': 0,
            'power_line': 0,
            'river': 0, 'stream': 0, 'dam': 0, 'ditch': 0, 'drain': 0, 'canal': 0,
            'roof': 0, 'retail': 0, 'apartment': 0, 'construction': 0, 'house': 0, 'unclass_building': 0, 'school': 0,
            'bare_rock': 0, 'wetland': 0, 'water': 0}

power_dict = OrderedDict([
    ('type', OrderedDict({
        'line': 'line',
    }))
])

natural_dict = OrderedDict([
    ('type', OrderedDict({
        'bare_rock': 'bare_rock',
        'wetland': 'wetland',
        'water': 'water',
    }))
])

building_dict = OrderedDict([
    ('type', OrderedDict({
        'roof': 'roof',
        'retail': 'retail',
        'apartment': 'apartment',
        'construction': 'construction',
        'house': 'house',
        'school': 'school',
        'unclassified': 'unclassified',
    }))
])

waterway_dict = OrderedDict([
    ('type', OrderedDict({
        'river': 'river',
        'stream': 'stream',
        'dam': 'dam',
        'ditch': 'ditch',
        'drain': 'drain',
        'canal': 'canal',
    }))
])

# Total grid size
ns_size = int(np.ceil((north - south) / grid_size))
ew_size = int(np.ceil((east - west) / grid_size))
data_set = [None] * ns_size * ew_size
data_set = [copy.deepcopy(LVS_dict) for i in data_set]

nominatim = Nominatim()
overpass = Overpass()


def fetch_highway(coordinates, typeOfRoad):
    query = overpassQueryBuilder(bbox=coordinates, elementType='way', selector='"highway"="' + typeOfRoad + '"', out='count')
    return overpass.query(query).countElements()


def fetch_waterway(coordinates, typeOfWater):
    query = overpassQueryBuilder(bbox=coordinates, elementType='way', selector='"waterway"="' + typeOfWater + '"', out='count')
    return overpass.query(query).countElements()


def fetch_power(coordinates):
    query = overpassQueryBuilder(bbox=coordinates, elementType='way', selector='"power"="line"', out='count')
    return overpass.query(query).countElements()


def fetch_natural(coordinates, typeOfNature):
    query = overpassQueryBuilder(bbox=coordinates, elementType='way', selector='"natural"="' + typeOfNature + '"', out='count')
    return overpass.query(query).countElements()


# Generate set of bounding boxes
coord_set = np.zeros((ns_size * ew_size, 2))
westing = west; northing = north
coord_dict = {}
i = 0
for r in range(ns_size):
    for c in range(ew_size):
        westing += grid_size
        # bounding box center point
        coord_set[r * ew_size + c, :] = [northing, westing]
        # Make bounding box coordinates
        south_edge = northing - grid_size/2
        north_edge = northing + grid_size/2
        west_edge = westing - grid_size/2
        east_edge = westing + grid_size/2
        coord_dict[i] = [south_edge, west_edge, north_edge, east_edge]
        i += 1
    northing -= grid_size
    westing = west

# Create dictionary for parsing


highway_dict = OrderedDict([
    ('coordinates', coord_dict),
    ('typeOfRoad', OrderedDict({
        'primary': 'primary',
        'secondary': 'secondary',
        'tertiary': 'tertiary',
        'service': 'service',
        'unpaved': 'unpaved',
        'residential': 'residential',
        'track': 'track',
        'footway': 'footway',
        'path': 'path',
        'cycleway': 'cycleway',
        'unclassified': 'unclassified'
    }))
])

data = Data(fetch_highway, highway_dict)
print(data)
# print(coord_set)
