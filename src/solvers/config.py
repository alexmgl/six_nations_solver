position_rules = {
    'BACK-THREE': (lambda x: x <= 3, "x <= 3"),
    'CENTRE': (lambda x: x <= 2, "x <= 2"),
    'FLY-HALF': (lambda x: x <= 1, "x <= 1"),
    'SCRUM-HALF': (lambda x: x <= 1, "x <= 1"),
    'BACK-ROW': (lambda x: x <= 3, "x <= 3"),
    'SECOND-ROW': (lambda x: x <= 2, "x <= 2"),
    'PROP': (lambda x: x <= 2, "x <= 2"),
    'HOOKER': (lambda x: x <= 1, "x <= 1"),

}

team_rule = (
    lambda min_players, max_players, num_players: min_players <= num_players <= max_players,
    "min_players <= num_players <= max_players"
)

team_to_emoji_map = {
    "England":  "🌹",  # Red rose (symbol of England Rugby)
    "Scotland": "🦄",  # Unicorn is the national animal
    "Wales":    "🐉",  # Welsh dragon
    "Ireland":  "☘️",  # Shamrock
    "France":   "🐓",  # Gallic rooster (le coq gaulois)
    "Italy":    "🤌"   # Iconic Italian "pinched fingers" gesture
}

points_map = {
    'try_by_a_back': 10,
    'try_by_a_forward': 15,
    'try_assist': 4,
    'try_conversion': 2,
    'penalty_kick': 3,
    'drop_goal': 5,
    'defenders_beaten': 2,
    'carried_metres': 1,  # pt per 10m made (e.g., 19m = 1 pt)
    '50-22': 7,
    'offload_to_hand': 2,
    'attacking_scrum_win': 1,
    'tackles': 1,
    'breakdown_steals': 5,
    'lineout_steals': 7,
    'penalty_conceded': -1,
    'official_player_of_the_match': 15,
    'yellow_card': -5,
    'red_card': -8
}

