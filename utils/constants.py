ACTIONS = [
    'walk to bed',
    'sit down on bed',
    'stand up from bed',
    'walk to chair',
    'sit down on chair',
    'stand up from chair',
    'walk to room',
    'fall on floor of room',
    'stand up from floor of room',
    'get in  bed',
    'lie in bed',
    'get out  bed',
    'roll in bed',
    'sit in bed'
]


NON_AGGREGATED_LABELS_DICT = {
    'walk to bed': 0,
    'sit down on bed': 1,
    'stand up from bed': 2,
    'walk to chair': 3,
    'sit down on chair': 4,
    'stand up from chair': 5,
    'walk to room': 6,
    'fall on floor of room': 7,
    'stand up from floor of room': 8,
    'get in  bed': 9,
    'lie in bed': 10,
    'get out  bed': 11,
    'roll in bed': 12,
    'sit in bed': 13,
}


AGGREGATED_LABELS_DICT = {
    'walk to bed': 0,
    'sit down on bed': 1,
    'stand up from bed': 2,
    'walk to chair': 0,
    'sit down on chair': 1,
    'stand up from chair': 2,
    'walk to room': 0,
    'fall on floor of room': 3,
    'stand up from floor of room': 4,
    'get in  bed': 5,
    'lie in bed': 6,
    'get out  bed': 7,
    'roll in bed': 8,
    'sit in bed': 9,
}


MAPPING_LABELS_DICT = {
    NON_AGGREGATED_LABELS_DICT[key]: AGGREGATED_LABELS_DICT[key] for key in NON_AGGREGATED_LABELS_DICT.keys()
}