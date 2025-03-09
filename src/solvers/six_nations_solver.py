import time
import os
import pandas as pd
import numpy as np
from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint, ConstraintList,
    NonNegativeReals, Binary, Integers, Reals, SolverFactory, value, maximize,
    NonNegativeIntegers, Param, Piecewise
)
from pyomo.opt import SolverStatus, TerminationCondition, check_available_solvers
from rich.console import Console
from rich.table import Table

# Adjust these imports to your own project structure:
from src.utils.path_config import DATA_DIR
from src.solvers.config import team_rule, position_rules
from src.utils.logger import setup_logger

sl = setup_logger(__name__)


###############################################################################
# SixNationsSolver
###############################################################################
class SixNationsSolver:
    """
    A solver for the Six Nations Fantasy Team optimization problem using Pyomo.

    This class handles:
    - Configuration loading (via constructor parameters).
    - Data import (either test data or a user-provided dataset).
    - Model building (variable creation, constraints, objective).
    - Solving and reporting of results.
    """

    def __init__(
            self,
            starting_budget=230,
            max_team_size=15,
            enforce_all_starting_players=True,  # enforces whether there must be 15 starting players on the pitch
            max_substitutes=1,
            max_same_team=4,
            captain_multiplier=2,
            super_sub_multiplier=3,
            team_must_include=None,  # list of players the starting team must include
            team_must_exclude=None,  # players to exclude from the team
            starting_players_list=None,  # if this list has len > 0 then use it to determine starting players
            substitutes_list=None,  # if this list has len > 0 then use it to determine subs
            set_captain=None,  # will set the team captain, default None
            set_super_sub=None,  # will set the super sub captain, default None
            set_max_player_cost=1e2,
            set_min_player_cost=0,
            example_data_path='example_2025_gw1.csv'
    ):
        """
        Initialize the solver with the various constraints and configuration settings.

        :param starting_budget: The total budget available to build the team (default: 230).
        :param max_team_size: The maximum number of players in the starting team (default: 15).
        :param max_substitutes: The maximum number of substitutes (default: 1).
        :param max_same_team: The maximum number of players allowed from the same real-life team (default: 4).
        :param captain_multiplier: The points multiplier for the captain (default: 2).
        :param super_sub_multiplier: The points multiplier for the "super sub" (default: 3).
        :param team_must_include: A list of player IDs that must be in the starting team.
        :param team_must_exclude: A list of player IDs that must be excluded entirely.
        :param substitutes_list: (Optional) A list of player IDs that must be considered as substitutes.
        :param set_captain: (Optional) Player ID to enforce as captain.
        :param set_super_sub: (Optional) Player ID to enforce as the super sub.
        :param set_max_player_cost: Filter out players above this cost (default: 1e2).
        :param set_min_player_cost: Filter out players below this cost (default: 0).
        :param example_data_path: Path for reading test data (default: 'example_2025_gw1.csv').
        """
        self.data = None  # Will store player data as a pandas DataFrame

        # Default to empty lists if None were passed
        self.starting_players_list = starting_players_list
        self.substitutes_list = substitutes_list

        self.team_must_include = team_must_include if team_must_include else []
        self.team_must_exclude = team_must_exclude if team_must_exclude else []

        # Constraints and model parameters
        self.position_rules = position_rules
        self.team_rule = team_rule
        self.starting_budget = starting_budget
        self.max_team_size = max_team_size
        self.enforce_all_starting_players = enforce_all_starting_players  # enforces whether there must be 15 starting players on the pitch
        self.max_substitutes = max_substitutes
        self.max_same_team = max_same_team
        self.captain_multiplier = captain_multiplier
        self.super_sub_multiplier = super_sub_multiplier
        self.set_captain = set_captain
        self.set_super_sub = set_super_sub
        self.set_max_player_cost = set_max_player_cost
        self.set_min_player_cost = set_min_player_cost
        self.example_data_path = example_data_path

        # Pyomo model and related attributes
        self.model = ConcreteModel()
        self.build_time = None
        self.solve_time = None
        self.solver_status = None
        self.solver_results = None

        # Internally used structures for sets/dicts
        self.players = None
        self.positions = None
        self.points_dict = None
        self.player_price_dict = None
        self.player_pos_dict = None
        self.player_name_dict = None
        self.player_club_dict = None

    # --------------------------------------------------------------------------
    # Data Loading Methods
    # --------------------------------------------------------------------------
    # todo - need to add some validation in here to check whether the dataframe is valid.
    def load_data(self, data):
        """
        Load the player data into the solver from a DataFrame or NumPy array.

        The data is expected to have the following columns at minimum:
            - ID: Unique player ID
            - Name: Player name
            - Club: The team/country the player belongs to
            - Position: The player's position
            - Value: The cost/value of the player
            - Points: The expected (or projected) fantasy points for the player

        :param data: A pandas DataFrame or NumPy array with the required columns.
        :raises ValueError: If the data format is not supported or is missing required columns.
        """
        sl.info("Loading data into the solver.")

        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        elif isinstance(data, np.ndarray):
            self.data = pd.DataFrame(data)
        else:
            raise ValueError("Data must be a pandas DataFrame or a NumPy array.")

        # Ensure columns exist
        for col in ["ID", "Name", "Club", "Position", "Value", "Points"]:
            if col not in self.data.columns:
                raise ValueError(f"Expected column '{col}' missing in the provided data.")

        # Filter data based on cost constraints
        self.data = self.data[
            (self.data["Value"] >= self.set_min_player_cost) &
            (self.data["Value"] <= self.set_max_player_cost)
            ]

        # Prepare internal sets and dictionaries
        self.__set_model_params()

    def load_test_data(self):
        """
        Convenience method to load a test dataset from a CSV file located at DATA_DIR.
        """
        sl.info(f"Loading test data from {self.example_data_path}.")
        path = os.path.join(DATA_DIR, self.example_data_path)
        self.data = pd.read_csv(path)

        # Filter data based on cost constraints and prepare model parameters
        self.data = self.data[
            (self.data["Value"] >= self.set_min_player_cost) &
            (self.data["Value"] <= self.set_max_player_cost)
            ]

        self.substitutes_list = self.data.loc[self.data['Substitute'] == True]['ID'].tolist()

        self.__set_model_params()

    def __set_model_params(self):
        """
        Internal method to set up the sets and dictionaries (Pyomo parameters) based on the DataFrame.
        """
        if self.data is not None:
            sl.debug("Setting internal model parameters from data.")
            self.players = self.data['ID'].unique()
            self.positions = self.data['Position'].unique()

            # Create dictionaries for easy access in constraints/objective
            self.points_dict = self.data.set_index('ID')['Points'].to_dict()
            self.player_price_dict = self.data.set_index('ID')['Value'].to_dict()
            self.player_pos_dict = self.data.set_index('ID')['Position'].to_dict()
            self.player_name_dict = self.data.set_index('ID')['Name'].to_dict()
            self.player_club_dict = self.data.set_index('ID')['Club'].to_dict()

    # --------------------------------------------------------------------------
    # Model Construction: Variables, Objective, Constraints
    # --------------------------------------------------------------------------
    def build_model(self):
        """
        Build the Pyomo model (variables, constraints, objective) based on the loaded data.

        This method:
         1. Creates variables (team_var, captain_var, substitutes, super_sub).
         2. Sets up the objective function (maximize points with multipliers).
         3. Adds the constraints (budget, team size, position rules, etc.).
         4. Tracks the time taken to build the model in 'build_time'.
        """
        sl.info("Building the optimization model.")
        if self.data is None:
            raise ValueError("No data loaded. Please load data before building the model.")

        start_time = time.time()

        # 1. Add Variables
        self.__add_variables()

        # 2. Add Objective
        self.model.objective = Objective(rule=self.__objective_rule, sense=maximize)
        sl.debug("Objective function has been added to the model.")

        # 3. Add Constraints
        self.__add_constraints()

        self.build_time = time.time() - start_time
        sl.info(f"Model built in {self.build_time:.4f} seconds.")

    def __add_variables(self):
        """
        Create the Pyomo variables for the model:
          - team_var[i]: 1 if player i is in the starting team.
          - captain_var[i]: 1 if player i is the captain.
          - substitutes[i]: 1 if player i is a substitute (but not in the starting team).
          - super_sub[i]: 1 if player i is the 'super sub'.
        """
        sl.debug("Adding model variables.")

        self.model.team_var = Var(self.players, domain=Binary)  # all players in starting team
        self.model.captain_var = Var(self.players, domain=Binary)  # a subset of starting team that earns a multiplier

        self.model.substitutes = Var(self.players, domain=Binary)  # all substitutes
        self.model.super_sub = Var(self.players, domain=Binary)  # a subset of substitutes that earns a multiplier

    def __objective_rule(self, m):
        """
        Objective rule for the model:
         - Sum of base points (for all players in the team).
         - Plus the additional captain multiplier points.
         - Plus the super sub multiplier points.
        """
        normal_points = sum(m.team_var[i] * self.points_dict[i] for i in self.players)

        captain_points = sum(
            m.captain_var[i] * self.points_dict[i] * (self.captain_multiplier - 1)
            for i in self.players
        )

        super_sub_points = sum(
            m.super_sub[i] * self.points_dict[i] * self.super_sub_multiplier
            for i in self.players
        )

        return normal_points + captain_points + super_sub_points

    def __add_constraints(self):
        """
        Add all constraints to the Pyomo model, each in its own method for clarity.
        """
        sl.debug("Adding constraints to the model.")

        # Core constraints
        self.model.team_size_constraint = Constraint(rule=self.__c1_team_size)
        self.model.substitutes_constraint = Constraint(rule=self.__c2_total_number_of_substitutes)
        self.model.subs_mutually_exclusive_constraint = Constraint(self.players,
                                                                   rule=self.__c3_substitutes_cannot_be_in_team)
        self.model.super_sub_constraint = Constraint(self.players, rule=self.__c4_super_sub_must_be_substitute)
        self.model.budget_constraint = Constraint(rule=self.__c5_budget_rule)
        self.model.one_captain_constraint = Constraint(rule=self.__c6_one_captain_rule)
        self.model.captain_must_be_in_team_constraint = Constraint(self.players, rule=self.__c7_captain_from_team_rule)

        # Team-based and position-based constraints
        self.__c8_limit_number_of_players_from_team()
        self.model.position_constraints = Constraint(self.positions, rule=self.__c9_position_rules)

        # Inclusion/Exclusion constraints
        self.__c10_exclude_players()
        self.__c11_include_players()

        # Force the substitutes/super_sub to be from self.substitutes_list if it is provided
        self.__c12_force_substitutes_from_list()

        # Force the starting team to be from self.starting_players_list if it is provided
        self.__c15_force_substitutes_from_list()

        # Fix the captain and/or super sub if they are defined
        self.__c13_fix_captain_if_defined()
        self.__c14_fix_super_sub_if_defined()

        # Remove any players in the substitutes_list from being selected in the starting team
        self.model.remove_subs_from_team_constraint = Constraint(rule=self.__c16_remove_subs_from_starting_team)

    # --------------------------------------------------------------------------
    # Individual Constraint Rules
    # --------------------------------------------------------------------------
    def __c1_team_size(self, m):
        """
        Enforce the team size: if enforce_all_starting_players is True,
        then exactly self.max_team_size players must be selected; otherwise,
        the number of players cannot exceed self.max_team_size.
        """
        team_count = sum(m.team_var[i] for i in self.players)
        return team_count == self.max_team_size if self.enforce_all_starting_players else team_count <= self.max_team_size

    def __c2_total_number_of_substitutes(self, m):
        """
        The total number of subs cannot exceed self.max_substitutes.
        """
        return sum(m.substitutes[i] for i in self.players) <= self.max_substitutes

    def __c3_substitutes_cannot_be_in_team(self, m, i):
        """
        A player cannot be both in the starting team and a super sub simultaneously.
        """
        return m.team_var[i] + m.super_sub[i] <= 1

    def __c4_super_sub_must_be_substitute(self, m, i):
        """
        The super sub must be designated as a substitute (substitutes[i] == 1).
        This constraint is flexible in case your rules allow more than one sub.
        """
        return m.super_sub[i] <= m.substitutes[i]

    def __c5_budget_rule(self, m):
        """
        The total cost of all selected players (team + super sub) cannot exceed the starting budget.
        """
        return (
                sum(m.team_var[i] * self.player_price_dict[i] for i in self.players) +
                sum(m.substitutes[i] * self.player_price_dict[i] for i in self.players)
                <= self.starting_budget
        )

    def __c6_one_captain_rule(self, m):
        """
        Exactly one captain must be selected.
        """
        return sum(m.captain_var[i] for i in self.players) == 1

    def __c7_captain_from_team_rule(self, m, i):
        """
        The captain must come from the players who are in the starting team.
        """
        return m.captain_var[i] <= m.team_var[i]

    def __c8_limit_number_of_players_from_team(self):
        """
        No more than self.max_same_team players from each real-life team/club.
        """
        sl.debug("Adding constraint: No more than %d players from the same club.", self.max_same_team)
        self.model.c8 = ConstraintList()

        grouped_codes = self.data.groupby('Club')['ID'].apply(list)
        for club, ids_in_club in grouped_codes.items():
            self.model.c8.add(
                sum(self.model.team_var[i] + self.model.substitutes[i] for i in ids_in_club) <= self.max_same_team
            )

    def __c9_position_rules(self, m, position):
        """
        Each position has a custom rule, e.g. 'You must have at least 2 props, at most 2 scrum-halves, etc.'
        The rules are stored in self.position_rules, which is a dict of the form:
            position_rules = {
                'PROP': [lambda x: x == 2],
                'HOOKER': [lambda x: x == 1],
                ...
            }
        Each entry is a list containing a single callable that returns a constraint expression.
        """
        idx = [i for i in self.players if self.player_pos_dict[i] == position]
        position_constraint_func = self.position_rules[position][0]
        total_in_position = sum(m.team_var[i] for i in idx)
        return position_constraint_func(total_in_position)

    def __c10_exclude_players(self):
        """
        Exclude players listed in self.team_must_exclude from both the starting team and the super sub list.
        """
        sl.debug("Adding constraints to exclude specific players.")
        self.model.exclude_players_constraint = ConstraintList()
        for player_id in self.team_must_exclude:
            if player_id in self.players:
                self.model.exclude_players_constraint.add(
                    self.model.team_var[player_id] + self.model.substitutes[player_id] == 0
                )

    def __c11_include_players(self):
        """
        Ensure that certain players (in self.team_must_include) are in the starting team.
        """
        sl.debug("Adding constraints to include specific players.")
        self.model.include_players_constraint = ConstraintList()
        for player_id in self.team_must_include:
            if player_id in self.players:
                self.model.include_players_constraint.add(
                    self.model.team_var[player_id] == 1
                )

    def __c12_force_substitutes_from_list(self):
        """
        If self.substitutes_list is non-empty, ensure that any substitute or super sub
        must be chosen from that list. That is, for any player not in self.substitutes_list,
        substitutes[i] == 0 and super_sub[i] == 0.
        """
        if self.substitutes_list is None:
            sl.debug('No substitutes specified.')
            print(self.substitutes_list)
            return  # If no specific substitute list is provided, do nothing

        sl.debug("Adding constraint: Only players in 'substitutes_list' can be substitutes or super subs.")

        self.model.valid_substitutes_list_constraint = ConstraintList()
        allowed_subs = set(self.substitutes_list)  # for fast membership checks

        for player_id in self.players:
            if player_id not in allowed_subs:
                # Force these variables to 0 if the player is not in the sub list
                self.model.valid_substitutes_list_constraint.add(self.model.substitutes[player_id] == 0)

    def __c13_fix_captain_if_defined(self):
        """
        If self.set_captain is defined, force that player to be captain (captain_var[player] == 1).
        Because of the sum(captain_var) == 1 constraint, that player will be the only captain.
        Also forces that player to be in the team.
        """
        if self.set_captain is not None:
            # Make sure that the chosen captain is in the data
            if self.set_captain not in self.players:
                raise ValueError(f"Captain ID {self.set_captain} not found in loaded data.")

            # Because the constraint sum(captain_var[i]) = 1 already exists,
            # setting captain_var[set_captain] = 1 forces the solver to pick that player as captain.
            # Also, the 'captain must be in the team' constraint ensures team_var[set_captain] == 1.
            self.model.fixed_captain_constraint = Constraint(
                expr=self.model.captain_var[self.set_captain] == 1
            )

    def __c14_fix_super_sub_if_defined(self):
        """
        If self.set_super_sub is defined, force that player to be a super sub (super_sub[player] == 1).
        We still allow multiple super subs if max_substitutes > 1, but if only one is allowed,
        this will fix exactly that one.
        """
        if self.set_super_sub is not None:
            # Make sure that the chosen super sub is in the data
            if self.set_super_sub not in self.players:
                raise ValueError(f"Super sub ID {self.set_super_sub} not found in loaded data.")

            # This constraint will force that player to have the super_sub variable = 1.
            # The model already ensures super_sub[i] + team_var[i] <= 1, so the forced super sub
            # won't appear in the starting team.
            self.model.fixed_super_sub_constraint = Constraint(
                expr=self.model.super_sub[self.set_super_sub] == 1
            )

    def __c15_force_substitutes_from_list(self):
        """
        If self.starting_players_list is non-empty, ensure that any substitute or super sub
        must be chosen from that list. That is, for any player not in self.substitutes_list,
        substitutes[i] == 0 and super_sub[i] == 0.
        """
        if self.starting_players_list is None:
            sl.debug('No starting players specified.')
            return  # If no specific substitute list is provided, do nothing

        sl.debug("Adding constraint: Only players in 'starting_players_list' can be in the starting team.")

        self.model.valid_players_list_constraint = ConstraintList()
        allowed_players = set(self.starting_players_list)  # for fast membership checks

        for player_id in self.players:
            if player_id not in allowed_players:
                # Force these variables to 0 if the player is not in the sub list
                self.model.valid_players_list_constraint.add(self.model.team_var[player_id] == 0)

    def __c16_remove_subs_from_starting_team(self, m):
        """
        Ensure that players listed in substitutes_list are not selected in the starting team.
        """
        if self.substitutes_list is None:
            return Constraint.Skip
        return sum(m.team_var[i] for i in self.substitutes_list if i in self.players) == 0

    # --------------------------------------------------------------------------
    # Solving and Reporting
    # --------------------------------------------------------------------------
    def solve(self, solver_name='cplex'):
        """
        Solve the built Pyomo model using the specified solver, storing results and solver status.

        :param solver_name: The name of the solver to use (e.g., 'glpk', 'cplex', 'gurobi').
        :return: The solver result object containing status, termination condition, etc.
        :raises ValueError: If the model has not been built yet.
        """
        if self.model is None:
            raise ValueError("Model not built. Please build the model before solving.")

        sl.info(f"Requested solver: {solver_name}")

        sl.info(f"Solving the model with {solver_name}...")
        start_time = time.time()

        solver = SolverFactory(solver_name)
        result = solver.solve(self.model, tee=False)

        self.solve_time = time.time() - start_time
        self.solver_status = result.solver.status
        self.solver_results = result

        sl.info(f"Solver finished in {self.solve_time:.4f} seconds with status {self.solver_status}.")

        # Check solver status and print result if optimal
        if (result.solver.status == SolverStatus.ok) and \
                (result.solver.termination_condition == TerminationCondition.optimal):
            sl.info("Optimal solution found. Printing result:")
            self.print_result()
        else:
            sl.warning(f"Solver ended with status {result.solver.status} "
                       f"and termination condition {result.solver.termination_condition}.")

        return result

    def print_result(self):

        """
        Print the final team selection (team, captain, substitutes) in a rich table format.
        """
        sl.debug("Preparing solution for display.")

        # Extract solution values
        team_solution = {p: self.model.team_var[p].value for p in self.players}
        captain_solution = {p: self.model.captain_var[p].value for p in self.players}
        sub_solution = {p: self.model.super_sub[p].value for p in self.players}

        # Convert to DataFrames
        df_team = pd.DataFrame.from_dict(team_solution, orient='index', columns=['Team'])
        df_captain = pd.DataFrame.from_dict(captain_solution, orient='index', columns=['Captain'])
        df_sub = pd.DataFrame.from_dict(sub_solution, orient='index', columns=['Substitute'])

        # Merge into one DataFrame
        df_merged = pd.concat([df_team, df_captain, df_sub], axis=1)

        # Filter to only selected players (Team == 1 or Substitute == 1)
        selected = (df_merged['Team'] == 1) | (df_merged['Substitute'] == 1)
        df_selected = df_merged[selected].copy()

        # Insert user-friendly columns
        df_selected.insert(0, 'Club', df_selected.index.map(self.player_club_dict))
        df_selected.insert(0, 'Position', df_selected.index.map(self.player_pos_dict))
        df_selected.insert(0, 'Name', df_selected.index.map(self.player_name_dict))
        df_selected['Points'] = df_selected.index.map(self.points_dict)

        # Sort positions in a custom order
        custom_order = ["PROP", "HOOKER", "SECOND-ROW", "BACK-ROW", "SCRUM-HALF", "FLY-HALF", "CENTRE", "BACK-THREE"]
        df_selected["Position"] = pd.Categorical(df_selected["Position"], categories=custom_order, ordered=True)
        df_selected.sort_values("Position", inplace=True)

        # Add labels to name if Captain or Substitute
        df_selected['Name'] += df_selected['Captain'].apply(lambda x: ' (C)' if x == 1 else '')
        df_selected['Name'] += df_selected['Substitute'].apply(lambda x: ' (SUB)' if x == 1 else '')

        # Modify the points column to reflect multipliers
        df_selected["Points"] = df_selected.apply(self.__format_points_with_multiplier, axis=1)

        # Print in a colorized table
        self.__print_dataframe_with_colors(df_selected)

    def __format_points_with_multiplier(self, row):
        """
        Helper method to display points along with any multiplier in parentheses.
        """
        base_points = round(row["Points"], 0)
        if row["Substitute"] == 1:
            return f'{base_points * self.super_sub_multiplier} ({base_points}*{self.super_sub_multiplier})'
        elif row["Captain"] == 1:
            return f'{base_points * self.captain_multiplier} ({base_points}*{self.captain_multiplier})'
        return str(base_points)

    def __print_dataframe_with_colors(self, df, header_colour='bright_red', chip_colour='dark_olive_green1'):
        """
        Print the solution DataFrame using Rich with conditional coloring for
        captain/substitute rows.
        """
        print('\n')
        console = Console(force_terminal=True)
        table = Table(title=f"SIX NATIONS SOLVER ({self.model.objective():,.0F} points)", show_header=True,
                      header_style=header_colour)

        # Create columns
        table.add_column("Index", justify="center", style="yellow", no_wrap=True)
        columns_to_display = [col for col in df.columns if col not in ["Captain", "Substitute", "Team"]]
        for col in columns_to_display:
            table.add_column(col, justify="center", style="cyan", no_wrap=True)

        # Populate table rows
        for idx, row in df.iterrows():
            row_values = [str(idx)] + [str(row[col]) for col in columns_to_display]

            # Color captain or substitute row in green
            if row["Captain"] == 1 or row["Substitute"] == 1:
                table.add_row(*row_values, style=chip_colour)
            else:
                table.add_row(*row_values, style="white")

        console.print(table)

    # --------------------------------------------------------------------------
    # Magic Methods
    # --------------------------------------------------------------------------
    def __repr__(self):
        """
        Debug-friendly representation of the solver object.
        """
        return (f"<SixNationsSolver("
                f"data_loaded={self.data is not None}, "
                f"model_built={hasattr(self.model, 'objective')}, "
                f"build_time={self.build_time}, "
                f"solve_time={self.solve_time})>")

    def __str__(self):
        """
        User-friendly string representation of the solver object.
        """
        return (f"SixNationsSolver with constraints: budget={self.starting_budget}, "
                f"team_size={self.max_team_size}, max_subs={self.max_substitutes}, "
                f"max_same_team={self.max_same_team}, captain_x{self.captain_multiplier}, "
                f"super_sub_x{self.super_sub_multiplier}. "
                f"Includes {len(self.team_must_include)} forced-in players and "
                f"{len(self.team_must_exclude)} excluded players.")


# ------------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    sl.info("Running example usage of SixNationsSolver in standalone mode.")
    solver = SixNationsSolver()
    solver.load_test_data()  # Loads test data from CSV
    solver.build_model()  # Builds the Pyomo model
    solver.solve()  # Solves and prints results
