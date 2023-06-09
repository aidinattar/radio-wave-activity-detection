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


MAPPING_LABELS_NAMES_DICT = {
    'walk to bed': 'walk',
    'sit down on bed': 'sit down',
    'stand up from bed': 'stand up',
    'walk to chair': 'walk',
    'sit down on chair': 'sit down',
    'stand up from chair': 'stand up',
    'walk to room': 'walk',
    'fall on floor of room': 'fall on floor',
    'stand up from floor of room': 'stand up from floor',
    'get in  bed':'get in  bed',
    'lie in bed':'lie in bed',
    'get out  bed':'get out  bed',
    'roll in bed':'roll in bed',
    'sit in bed':'sit in bed',
}


AGGREGATED_LABELS_DICT_REVERSE = {
    0: 'walk',
    1: 'sit down',
    2: 'stand up',
    3: 'fall on the floor',
    4: 'stand up from the floor',
    5: 'get in bed',
    6: 'lie in bed',
    7: 'get out of bed',
    8: 'roll in bed',
    9: 'sit in bed'
}


NON_AGGREGATED_LABELS_DICT_REVERSE = {
    0: 'walk to bed',
    1: 'sit down on bed',
    2: 'stand up from bed',
    3: 'walk to chair',
    4: 'sit down on chair',
    5: 'stand up from chair',
    6: 'walk to room',
    7: 'fall on floor of room',
    8: 'stand up from floor of room',
    9: 'get in  bed',
    10: 'lie in bed',
    11: 'get out  bed',
    12: 'roll in bed',
    13: 'sit in bed',
}


STANDING_LABELS = {
    0: 'walk to bed',
    1: 'sit down on bed',
    2: 'stand up from bed',
    3: 'walk to chair',
    4: 'sit down on chair',
    5: 'stand up from chair',
    6: 'walk to room',
    7: 'fall on floor of room',
    8: 'stand up from floor of room',
}


LYING_LABELS = {
    9: 'get in  bed',
    10: 'lie in bed',
    11: 'get out  bed',
    12: 'roll in bed',
    13: 'sit in bed'
}