# Six Nations Solver

🏉 **Six Nations Solver** is an optimisation tool for selecting the best fantasy team for the Six Nations Championship. It utilises **Mixed-Integer Linear Programming (MILP)** with **Pyomo** to maximise expected points while adhering to constraints such as budget, player positions, team balance, and special multipliers (captain, super-sub).

![Language](https://img.shields.io/badge/language-Python-blue)
![Version](https://img.shields.io/badge/version-1.0.0-brightgreen)

## 🏆 Features
- **Optimised Team Selection**: Selects the best 15-player squad plus substitutes while considering position and budget constraints.
- **Customisable Constraints**: Users can enforce specific players, exclude others, or set team-based limits.
- **Captain & Super-Sub Multipliers**: Includes special scoring rules such as captains (x2 points) and super-subs (x3 points).
- **Data Filtering**: Automatically excludes players outside specified cost bounds.
- **Rich Output**: Prints a formatted results table with player names, positions, teams, and expected points.
- **Flexible Solver**: Supports various solvers, including `cplex`, `glpk`, and `gurobi`.

## 📥 Installation
Ensure you have Python 3.8+ installed. Clone the repository and install dependencies:
```bash
$ git clone https://github.com/alexmgl/six_nations_solver.git
$ cd six_nations_solver
$ pip install -r requirements.txt
```

## 📊 Data Format
The input data should be in a CSV or DataFrame format with the following columns:
| Column   | Description                        |
|----------|------------------------------------|
| ID       | Unique player ID                  |
| Name     | Player name                        |
| Club     | The Six Nations team (e.g., "France") |
| Position | Player's position (e.g., "PROP")  |
| Value    | Player cost (budget impact)       |
| Points   | Expected fantasy points           |

Example CSV (`example_2025_gw1.csv`):
```csv
ID,Name,Club,Position,Value,Points
74,A. Porter,Ireland,PROP,24,40
171,D. Fischetti,Italy,PROP,26,35
110,J. Marchand,France,HOOKER,27,45
```

## 🚀 Usage

### 1️⃣ Basic Usage (Custom Data)
```python
from six_nations_solver import SixNationsSolver
import pandas as pd

# Load custom data
data = pd.read_csv("path_to_your_data.csv")

# Initialise solver
solver = SixNationsSolver(starting_budget=230, max_team_size=15, max_substitutes=1, max_same_team=4)

# Load player data
solver.load_data(data)

# Build optimisation model
solver.build_model()

# Solve the model
solver.solve(solver_name='cplex')

# Print results
solver.print_result()
```

### 2️⃣ Quick Test with Built-in Data (2025 gameweek 1 actual points)
```python
from six_nations_solver import SixNationsSolver

# Initialise solver
solver = SixNationsSolver()

# Load built-in 2025 gameweek 1 data
solver.load_test_data()

# Build and solve the model
solver.build_model()
solver.solve(solver_name='cplex')

# Print results
solver.print_result()

```

### 3️⃣ Advanced Usage: Custom Constraints
```python
solver = SixNationsSolver(
    starting_budget=225,         # Custom budget
    max_team_size=15,            # Limit team to 15 players
    max_substitutes=2,           # Allow 2 substitutes
    max_same_team=4,             # Max 4 players from the same country
    captain_multiplier=2,        # Captain earns double points
    super_sub_multiplier=3,      # Super sub earns triple points
    team_must_include=[101, 202], # Must include Sexton & Dupont
    team_must_exclude=[303]       # Exclude Maro Itoje
)

# Load player data
solver.load_data(data)

# Build optimisation model
solver.build_model()

# Solve the model
solver.solve(solver_name='cplex')

# Print results
solver.print_result()

```


## 🎛️ Configuration Parameters
The `SixNationsSolver` constructor allows customisation through various parameters:
| Parameter          | Default | Description |
|-------------------|---------|-------------|
| `starting_budget` | 230     | Maximum total team cost |
| `max_team_size`  | 15      | Number of players in the squad |
| `max_substitutes` | 1       | Number of substitutes allowed |
| `max_same_team`  | 4       | Maximum players per Six Nations team |
| `captain_multiplier` | 2 | Captain's points multiplier |
| `super_sub_multiplier` | 3 | Super-sub's points multiplier |
| `team_must_include` | None | List of player IDs required in the squad |
| `team_must_exclude` | None | List of player IDs to exclude |
| `set_captain` | None | Enforce a specific player as captain |
| `set_super_sub` | None | Enforce a specific player as super-sub |

## 📜 Example Output
Upon solving, the solver prints a formatted table:
```md
                 SIX NATIONS SOLVER (767.0 points)
┌───────┬───────────────────┬────────────┬──────────┬──────────────┐
│ Index │       Name        │  Position  │   Club   │    Points    │
├───────┼───────────────────┼────────────┼──────────┼──────────────┤
│  74   │     A. Porter     │    PROP    │ Ireland  │      24      │
│  171  │   D. Fischetti    │    PROP    │  Italy   │      26      │
│  110  │    J. Marchand    │   HOOKER   │  France  │      27      │
│  682  │    D. Jenkins     │ SECOND-ROW │  Wales   │      35      │
│  159  │    W. Rowlands    │ SECOND-ROW │  Wales   │      31      │
│  677  │ T. Reffell (SUB)  │  BACK-ROW  │  Wales   │ 87 (29 * 3)  │
│  351  │     R. Darge      │  BACK-ROW  │ Scotland │      52      │
│  150  │     T. Curry      │  BACK-ROW  │ England  │      50      │
│  118  │  G. Alldritt (C)  │  BACK-ROW  │  France  │ 142 (71 * 2) │
│  283  │  J. Gibson-Park   │ SCRUM-HALF │ Ireland  │      42      │
│  361  │     M. Smith      │  FLY-HALF  │ England  │      29      │
│  82   │     H. Jones      │   CENTRE   │ Scotland │      66      │
│  400  │   T. Menoncello   │   CENTRE   │  Italy   │      39      │
│  704  │ L. Bielle-Biarrey │ BACK-THREE │  France  │      41      │
│  686  │     C. Murley     │ BACK-THREE │ England  │      38      │
│ 1322  │   T. Attissogbe   │ BACK-THREE │  France  │      38      │
└───────┴───────────────────┴────────────┴──────────┴──────────────┘
```

* (C) → Captain (earns 2x points)
* (SUB) → Super Sub (earns 3x points)

## 🔧 Solver Options
The solver defaults to `cplex`, but you can use other solvers like:
```python
solver.solve(solver_name='glpk')  # Open-source alternative
```

Ensure the solver is installed on your system.

## 💡 Future Enhancements
* 📊 Graphical UI
* 🌍 Web App Version

## ❤️ Support the Project
If you find this project useful, consider supporting it!
<p align="center">
  <a href="https://www.buymeacoffee.com/alexmgl">
    <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Support-orange?logo=buy-me-a-coffee&logoColor=white" alt="Buy Me a Coffee">
  </a>
</p>

---
### 📌 Notes
- Ensure all dependencies are installed (`pip install -r requirements.txt`).
- The solver requires **Pyomo** and a compatible solver (e.g., CPLEX, GLPK, Gurobi).
- CSV input data should follow the format outlined above.

This README provides a **detailed overview** of the Six Nations Solver, including **installation, data format, usage instructions, and output interpretation**. 🚀

