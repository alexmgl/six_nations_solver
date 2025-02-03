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