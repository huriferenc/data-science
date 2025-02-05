import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
from sympy import symbols, Eq, Matrix, solve, sympify
import glob
import cv2
from datetime import timedelta, date
import seaborn as sns
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import interact_manual
from ipywidgets import interact
import json


def plot_lines(M):
    x_1 = np.linspace(-10, 10, 100)
    x_2_line_1 = (M[0, 2] - M[0, 0] * x_1) / M[0, 1]
    x_2_line_2 = (M[1, 2] - M[1, 0] * x_1) / M[1, 1]

    _, ax = plt.subplots(figsize=(10, 10))
    ax.plot(
        x_1,
        x_2_line_1,
        "-",
        linewidth=2,
        color="#0075ff",
        label=f"$x_2={-M[0,0]/M[0,1]:.2f}x_1 + {M[0,2]/M[0,1]:.2f}$",
    )
    ax.plot(
        x_1,
        x_2_line_2,
        "-",
        linewidth=2,
        color="#ff7300",
        label=f"$x_2={-M[1,0]/M[1,1]:.2f}x_1 + {M[1,2]/M[1,1]:.2f}$",
    )

    A = M[:, 0:-1]
    b = M[:, -1::].flatten()
    d = np.linalg.det(A)

    if d != 0:
        solution = np.linalg.solve(A, b)
        ax.plot(
            solution[0],
            solution[1],
            "-o",
            mfc="none",
            markersize=10,
            markeredgecolor="#ff0000",
            markeredgewidth=2,
        )
        ax.text(
            solution[0] - 0.25,
            solution[1] + 0.75,
            f"$(${solution[0]:.0f}$,{solution[1]:.0f})$",
            fontsize=14,
        )
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xticks(np.arange(-10, 10))
    ax.set_yticks(np.arange(-10, 10))

    plt.xlabel("$x_1$", size=14)
    plt.ylabel("$x_2$", size=14)
    plt.legend(loc="upper right", fontsize=14)
    plt.axis([-10, 10, -10, 10])

    plt.grid()
    plt.gca().set_aspect("equal")

    plt.show()


def string_to_augmented_matrix(equations):
    # Split the input string into individual equations
    equation_list = equations.split("\n")
    equation_list = [x for x in equation_list if x != ""]
    # Create a list to store the coefficients and constants
    coefficients = []

    ss = ""
    for c in equations:
        if c in "abcdefghijklmnopqrstuvwxyz":
            if c not in ss:
                ss += c + " "
    ss = ss[:-1]

    # Create symbols for variables x, y, z, etc.
    variables = symbols(ss)
    # Parse each equation and extract coefficients and constants
    for equation in equation_list:
        # Remove spaces and split into left and right sides
        sides = equation.replace(" ", "").split("=")

        # Parse the left side using SymPy's parser
        left_side = sympify(sides[0])

        # Extract coefficients for variables
        coefficients.append([left_side.coeff(variable) for variable in variables])

        # Append the constant term
        coefficients[-1].append(int(sides[1]))

    # Create a matrix from the coefficients
    augmented_matrix = Matrix(coefficients)
    augmented_matrix = np.array(augmented_matrix).astype("float64")

    A, B = augmented_matrix[:, :-1], augmented_matrix[:, -1].reshape(-1, 1)

    return ss, A, B


def plot_transformation(T, e1, e2):
    color_original = "#129cab"
    color_transformed = "#cc8933"

    _, ax = plt.subplots(figsize=(7, 7))
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xticks(np.arange(-5, 5))
    ax.set_yticks(np.arange(-5, 5))

    plt.axis([-5, 5, -5, 5])
    plt.quiver(
        [0, 0],
        [0, 0],
        [e1[0], e2[0]],
        [e1[1], e2[1]],
        color=color_original,
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    plt.plot(
        [0, e2[0][0], e1[0][0], e1[0][0]],
        [0, e2[1][0], e2[1][0], e1[1][0]],
        color=color_original,
    )
    e1_sgn = 0.4 * np.array([[1] if i == 0 else [i[0]] for i in np.sign(e1)])
    ax.text(
        e1[0] - 0.2 + e1_sgn[0],
        e1[1] - 0.2 + e1_sgn[1],
        f"$e_1$",
        fontsize=14,
        color=color_original,
    )
    e2_sgn = 0.4 * np.array([[1] if i == 0 else [i[0]] for i in np.sign(e2)])
    ax.text(
        e2[0] - 0.2 + e2_sgn[0],
        e2[1] - 0.2 + e2_sgn[1],
        f"$e_2$",
        fontsize=14,
        color=color_original,
    )

    e1_transformed = T(e1)
    e2_transformed = T(e2)

    plt.quiver(
        [0, 0],
        [0, 0],
        [e1_transformed[0], e2_transformed[0]],
        [e1_transformed[1], e2_transformed[1]],
        color=color_transformed,
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    plt.plot(
        [
            0,
            e2_transformed[0][0],
            e1_transformed[0][0] + e2_transformed[0][0],
            e1_transformed[0][0],
        ],
        [
            0,
            e2_transformed[1][0],
            e1_transformed[1][0] + e2_transformed[1][0],
            e1_transformed[1][0],
        ],
        color=color_transformed,
    )
    e1_transformed_sgn = 0.4 * np.array(
        [[1] if i == 0 else [i[0]] for i in np.sign(e1_transformed)]
    )
    ax.text(
        e1_transformed[0][0] - 0.2 + e1_transformed_sgn[0][0],
        e1_transformed[1][0] - e1_transformed_sgn[1][0],
        f"$T(e_1)$",
        fontsize=14,
        color=color_transformed,
    )
    e2_transformed_sgn = 0.4 * np.array(
        [[1] if i == 0 else [i[0]] for i in np.sign(e2_transformed)]
    )
    ax.text(
        e2_transformed[0][0] - 0.2 + e2_transformed_sgn[0][0],
        e2_transformed[1][0] - e2_transformed_sgn[1][0],
        f"$T(e_2)$",
        fontsize=14,
        color=color_transformed,
    )

    plt.gca().set_aspect("equal")
    plt.show()


def initialize_parameters(n_x):
    """
    Returns:
    params -- python dictionary containing your parameters:
                    W -- weight matrix of shape (n_y, n_x)
                    b -- bias value set as a vector of shape (n_y, 1)
    """

    ### START CODE HERE ### (~ 2 lines of code)
    W = np.random.randn(1, n_x) * 0.01  # @REPLACE EQUALS None
    b = np.zeros((1, 1))  # @REPLACE EQUALS None
    ### END CODE HERE ###

    assert W.shape == (1, n_x)
    assert b.shape == (1, 1)

    parameters = {"W": W, "b": b}

    return parameters


def compute_cost(Y_hat, Y):
    """
    Computes the cost function as a sum of squares

    Arguments:
    Y_hat -- The output of the neural network of shape (n_y, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)

    Returns:
    cost -- sum of squares scaled by 1/(2*number of examples)

    """
    # Number of examples.
    m = Y.shape[1]

    # Compute the cost function.
    cost = np.sum((Y_hat - Y) ** 2) / (2 * m)

    return cost


def backward_propagation(A, X, Y):
    """
    Implements the backward propagation, calculating gradients

    Arguments:
    parameters -- python dictionary containing our parameters
    A -- the output of the neural network of shape (1, number of examples)
    X -- input data of shape (n_x, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)

    Returns:
    grads -- python dictionary containing gradients with respect to different parameters
    """
    m = X.shape[1]

    # Backward propagation: calculate dW, db.
    dZ = A - Y
    dW = 1 / m * np.matmul(dZ, X.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

    grads = {"dW": dW, "db": db}

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule

    Arguments:
    parameters -- python dictionary containing parameters
    grads -- python dictionary containing gradients

    Returns:
    parameters -- python dictionary containing updated parameters
    """
    # Retrieve each parameter from the dictionary "parameters".
    W = parameters["W"]
    b = parameters["b"]

    # Retrieve each gradient from the dictionary "grads".
    dW = grads["dW"]
    db = grads["db"]

    # Update rule for each parameter.
    W = W - learning_rate * dW
    b = b - learning_rate * db

    parameters = {"W": W, "b": b}

    return parameters


def train_nn(parameters, A, X, Y, learning_rate=0.01):
    # Backpropagation. Inputs: "A, X, Y". Outputs: "grads".
    grads = backward_propagation(A, X, Y)

    # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
    parameters = update_parameters(parameters, grads, learning_rate)

    return parameters


def load_images(directory):
    images = []
    for filename in glob.glob(directory + "*.jpg"):
        img = np.array(image.imread(filename))
        gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(gimg)

        height, width = gimg.shape

    return images


def plot_reduced_data(X):
    plt.figure(figsize=(12, 12))
    plt.scatter(X[:, 0], X[:, 1], s=60, alpha=0.5)
    for i in range(len(X)):
        plt.text(X[i, 0], X[i, 1], str(i), size=15)
    plt.show()


class your_bday:
    def __init__(self) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        self.fig = fig
        self.ax = ax1
        self.ax_hist = ax2
        self.dates = [
            (date(2015, 1, 1) + timedelta(days=n)).strftime("%m-%d") for n in range(365)
        ]
        self.match = False
        self.bday_str = None
        self.bday_index = None
        self.n_students = 0
        self.history = []
        self.bday_picker = widgets.DatePicker(
            description="Pick your bday",
            disabled=False,
            style={"description_width": "initial"},
        )
        self.start_button = widgets.Button(description="Simulate!")

        display(self.bday_picker)
        display(self.start_button)

        self.start_button.on_click(self.on_button_clicked)

    def on_button_clicked(self, b):
        self.match = False
        self.n_students = 0

        self.get_bday()
        self.add_students()

    def get_bday(self):
        try:
            self.bday_str = self.bday_picker.value.strftime("%m-%d")
        except AttributeError:
            self.ax.set_title(f"Input a valid date and try again!")
            return
        self.bday_index = self.dates.index(self.bday_str)

    def generate_bday(self):
        # gen_bdays = np.random.randint(0, 365, (n_people))
        gen_bday = np.random.randint(0, 365)
        # if not np.isnan(self.y[gen_bday]):
        if gen_bday == self.bday_index:
            self.match = True

    def add_students(self):

        if not self.bday_str:
            return

        while True:
            if self.match:
                self.history.append(self.n_students)
                #                 print(f"Match found. It took {self.n_students} students to get a match")
                n_runs = [i for i in range(len(self.history))]
                self.ax.scatter(n_runs, self.history)
                # counts, bins = np.histogram(self.history)
                # plt.stairs(counts, bins)
                # self.ax_hist.hist(bins[:-1], bins, weights=counts)
                self.ax_hist.clear()
                sns.histplot(data=self.history, ax=self.ax_hist, bins=16)
                # plt.show()
                break

            self.generate_bday()
            self.n_students += 1
            self.ax.set_title(
                f"Match found. It took {self.n_students} students.\nNumber of runs: {len(self.history)+1}"
            )
            # self.fig.canvas.draw()
            # self.fig.canvas.flush_events()


big_classroom_sizes = [*range(1, 1000, 5)]
small_classroom_sizes = [*range(1, 80)]


def plot_simulated_probs(sim_probs, class_size):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    #     ax.scatter(class_size, sim_probs)
    sns.scatterplot(x=class_size, y=sim_probs, ax=ax, label="simulated probabilities")
    ax.set_ylabel("Simulated Probability")
    ax.set_xlabel("Classroom Size")
    ax.set_title("Probability vs Number of Students")
    ax.plot([0, max(class_size)], [0.5, 0.5], color="red", label="p = 0.5")
    ax.grid(which="minor", color="#EEEEEE", linewidth=0.8)
    ax.minorticks_on()
    ax.legend()
    plt.show()


class third_bday_problem:
    def __init__(self) -> None:
        fig, axes = plt.subplot_mosaic(
            [["top row", "top row"], ["bottom left", "bottom right"]], figsize=(10, 8)
        )
        self.fig = fig
        self.ax = axes["top row"]
        self.count_ax = axes["bottom left"]
        self.ax_hist = axes["bottom right"]
        self.ax.spines["top"].set_color("none")
        self.ax.spines["right"].set_color("none")
        self.ax.spines["left"].set_color("none")
        self.ax.get_yaxis().set_visible(False)
        x = np.arange(365)
        y = np.zeros((365,))
        y[y == 0] = np.nan

        y_match = np.zeros((365,))
        y_match[y_match == 0] = np.nan

        self.x = x
        self.y = y
        self.y_match = y_match
        self.match = False
        self.n_students = 0

        self.dates = [
            (date(2015, 1, 1) + timedelta(days=n)).strftime("%m-%d") for n in range(365)
        ]
        self.month_names = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

        self.history = []
        self.match_index = None
        self.match_str = None

        self.cpoint = self.fig.canvas.mpl_connect(
            "button_press_event", self.on_button_clicked
        )

        # self.start_button = widgets.Button(description="Simulate!")

        # display(self.start_button)

        # self.start_button.on_click(self.on_button_clicked)

    def generate_bday(self):
        gen_bday = np.random.randint(0, 365)

        if not np.isnan(self.y[gen_bday]):
            self.match_index = gen_bday
            self.match_str = self.dates[gen_bday]
            self.y_match[gen_bday] = 1
            self.match = True

        self.y[gen_bday] = 0.5

    def on_button_clicked(self, event):
        if event.inaxes in [self.ax]:
            self.new_run()
            self.add_students()

    def add_students(self):

        while True:
            if self.match:
                self.history.append(self.n_students)
                n_runs = [i for i in range(len(self.history))]
                self.count_ax.scatter(n_runs, self.history)
                self.count_ax.set_ylabel("# of students")
                self.count_ax.set_xlabel("# of simulations")

                month_str = self.month_names[int(self.match_str.split("-")[0]) - 1]
                day_value = self.match_str.split("-")[1]
                self.ax.set_title(
                    f"Match found for {month_str} {day_value}\nIt took {self.n_students} students to get a match"
                )
                self.ax_hist.clear()
                sns.histplot(data=self.history, ax=self.ax_hist, bins="auto")
                break

            self.generate_bday()
            self.n_students += 1
            self.ax.set_title(f"Number of students: {self.n_students}")

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            if not np.isnan(self.y_match).all():
                markerline, stemlines, baseline = self.ax.stem(
                    self.x, self.y_match, markerfmt="*"
                )
                plt.setp(markerline, color="green")
                plt.setp(stemlines, "color", plt.getp(markerline, "color"))
                plt.setp(stemlines, "linestyle", "dotted")
            self.ax.stem(self.x, self.y, markerfmt="o")

    def new_run(self):
        y = np.zeros((365,))
        y[y == 0] = np.nan
        y_match = np.zeros((365,))
        y_match[y_match == 0] = np.nan
        self.y_match = y_match
        self.y = y
        self.n_students = 0
        self.match = False
        self.ax.clear()


class monty_hall_game:
    def __init__(self) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig = fig
        self.ax = ax1
        self.results_ax = ax2
        self.memory_wins = {"switch": 0, "stay": 0}
        self.memory_games = {"switch": 0, "stay": 0}
        self.games_finished = 0
        self.start()

        self.cpoint = self.fig.canvas.mpl_connect("button_press_event", self.click_plot)

    def start(self) -> None:

        self.ax.clear()
        values = [10, 10, 10]
        door_numbers = ["Door 1", "Door 2", "Door 3"]

        self.ax.spines["top"].set_color("none")
        self.ax.spines["right"].set_color("none")
        self.ax.spines["left"].set_color("none")
        self.ax.get_yaxis().set_visible(False)

        self.ax.bar(
            door_numbers,
            values,
            color=["brown", "brown", "brown"],
            width=0.6,
            edgecolor=["black", "black", "black"],
        )
        self.ax.set_title(f"New game started, pick any door.")

        self.prize_coordinates = [-0.15, 0.85, 1.85]

        self.doors, self.winner_index = self.init_monty_hall()
        self.prizes = list(map(lambda x: "GOAT" if x == 0 else "CAR", list(self.doors)))

        self.choice = None
        self.switch = None
        self.temptative_final_door = None
        self.final_choice = None
        self.first_pick = True
        self.game_over = False
        self.won = None
        self.ilegal_move = False

    def click_plot(self, event):

        if event.inaxes in [self.ax]:
            if self.game_over:
                self.start()
                return

            # if self.choice and self.final_choice:
            #     self.start()

            if self.first_pick:
                self.first_pick_mtd(event.xdata)
            else:
                self.second_pick_mtd(event.xdata)

            if ((self.choice is not None) or (self.final_choice is not None)) and (
                not self.ilegal_move
            ):
                self.update_bar_chart()

    def first_pick_mtd(self, x_coord):

        if (x_coord >= -0.3) and (x_coord <= 0.3):
            self.choice = 0
        elif (x_coord >= 0.7) and (x_coord <= 1.3):
            self.choice = 1
        elif (x_coord >= 1.7) and (x_coord <= 2.3):
            self.choice = 2
        else:
            self.choice = None
            #             print("click a door")
            # self.ax.set_title(f"Click on a door to move forward")
            self.start()

    def second_pick_mtd(self, x_coord):

        if (x_coord >= -0.3) and (x_coord <= 0.3):
            self.final_choice = 0
        elif (x_coord >= 0.7) and (x_coord <= 1.3):
            self.final_choice = 1
        elif (x_coord >= 1.7) and (x_coord <= 2.3):
            self.final_choice = 2
        else:
            self.final_choice = None
            #             print("click a door")
            # self.ax.set_title(f"Click on a door to move forward")
            self.start()

        if self.final_choice == self.opened_door:
            self.ax.set_title(f"You selected the opened door.\nThis game doesn't count")
            self.ilegal_move = True
            self.game_over = True

    def update_bar_chart(self):

        if self.first_pick:
            values = [10, 10, 10]
            colors = ["brown", "brown", "brown"]
            edge_colors = ["black", "black", "black"]
            linewidths = [1, 1, 1]
            # colors[self.choice] = "red"
            edge_colors[self.choice] = "red"
            linewidths[self.choice] = 5

            door_numbers = ["Door 1", "Door 2", "Door 3"]
            self.opened_door = self.open_door()
            # values[opened_door] = 0

            colors[self.opened_door] = "gray"

            self.ax.clear()
            self.ax.text(
                self.prize_coordinates[self.opened_door],
                5,
                f"{self.prizes[self.opened_door]}",
            )

            # self.ax.text(0,10,f"You chose door {self.choice} and host opened door {opened_door}")
            # self.ax.text(0.4,5,f"{self.doors}")
            # self.results_ax.bar(door_numbers, values, color = colors,width = 0.6, edgecolor = ['black', 'black', 'black'])

            self.ax.bar(
                door_numbers,
                values,
                color=colors,
                width=0.6,
                edgecolor=edge_colors,
                linewidth=linewidths,
            )
            self.ax.set_title(
                f"You chose door {self.choice+1} and host opened door {self.opened_door+1}.\nDecide your final door."
            )
            self.first_pick = False
        else:
            values = [10, 10, 10]
            colors = ["gray", "gray", "gray"]
            colors[self.winner_index] = "green"
            edge_colors = ["black", "black", "black"]
            edge_colors[self.final_choice] = "red"
            linewidths = [1, 1, 1]
            linewidths[self.final_choice] = 5
            door_numbers = ["Door 1", "Door 2", "Door 3"]
            self.ax.clear()
            for i in range(3):
                self.ax.text(self.prize_coordinates[i], 5, f"{self.prizes[i]}")
            self.ax.bar(
                door_numbers,
                values,
                color=colors,
                width=0.6,
                edgecolor=edge_colors,
                linewidth=linewidths,
            )
            self.game_over = True
            self.check_if_switch()
            msg = " " if self.switch else " NOT "
            self.ax.set_title(
                f"You decided{msg}to switch and chose door #{self.final_choice+1}\n You got a {self.prizes[self.final_choice]}"
            )

            self.games_finished += 1
            if self.switch:
                self.memory_wins["switch"] += self.doors[self.final_choice]
                self.memory_games["switch"] += 1
            else:
                self.memory_wins["stay"] += self.doors[self.final_choice]
                self.memory_games["stay"] += 1

            self.update_results_chart()

    def update_results_chart(self):
        self.results_ax.clear()
        self.results_ax.set_title(
            f"Games finished: {self.games_finished}\nGames you switched: {self.memory_games['switch']}, Games you stayed: {self.memory_games['stay']}"
        )
        self.results_ax.scatter(
            ["switch", "stay"],
            [
                self.memory_wins["switch"] / self.memory_games["switch"],
                self.memory_wins["stay"] / self.memory_games["stay"],
            ],
            s=350,
        )
        self.results_ax.set_ylim(0, 1)

    def check_if_switch(self):
        self.switch = False if self.choice == self.final_choice else True

    def init_monty_hall(self):
        doors = np.array([0, 0, 0])
        winner_index = np.random.randint(0, 3)
        doors[winner_index] = 1

        return doors, winner_index

    def open_door(self):
        openable_doors = [
            i for i in range(3) if i not in (self.winner_index, self.choice)
        ]
        door_to_open = np.random.choice(openable_doors)

        return door_to_open


def success_rate_plot(f):
    def _plot(switch, n_iterations):
        wins = 0
        # iterations = 1000

        for _ in range(n_iterations):
            wins += f(switch=switch)

        win_rate = wins / n_iterations
        loss_rate = 1 - win_rate

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.pie(
            [win_rate, loss_rate],
            labels=["Win a car", "Win... a goat?"],
            colors=sns.color_palette("pastel")[2:],
            autopct="%.0f%%",
        )

        msg = "always" if switch else "never"
        ax.set_title(f"Win rate if you {msg} switch doors ({n_iterations} simulations)")
        plt.show()

    def _plot_generalized(switch, n_iterations, n=3, k=1):
        wins = 0
        # iterations = 1000

        for _ in range(n_iterations):
            try:
                wins += f(switch=switch, n=n, k=k)
            except ValueError:
                print(
                    "n is the number of doors and k is the amount of doors the host opens. Since you have already picked one door, k has to be at most n-2, so there is at least one openable door after the host open the k doors."
                )
                return

        win_rate = wins / n_iterations
        loss_rate = 1 - win_rate

        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.pie(
            [win_rate, loss_rate],
            labels=["Win a car", "Win... a goat?"],
            colors=sns.color_palette("pastel")[2:],
            autopct="%.0f%%",
        )

        msg = "always" if switch else "never"
        ax.set_title(f"Win rate if you {msg} switch doors ({n_iterations} simulations)")
        plt.show()

    n_iterations_selection = widgets.SelectionSlider(
        options=[1, 10, 100, 1000],
        value=1,
        description="# iterations",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
    )

    strategy_selection = widgets.RadioButtons(
        options=[True, False],
        value=False,
        description="Switch Doors?",
        disabled=False,
    )

    if f.__qualname__ == "monty_hall":

        interact_manual(
            _plot,
            switch=strategy_selection,
            n_iterations=n_iterations_selection,
        )

    if f.__qualname__ == "generalized_monty_hall":

        disabled = False

        n_selection = widgets.SelectionSlider(
            options=range(3, 101),
            value=3,
            description="n",
            disabled=disabled,
        )

        k_selection = widgets.SelectionSlider(
            options=range(0, 99),
            value=1,
            description="k",
            disabled=disabled,
        )

        interact_manual(
            _plot_generalized,
            switch=strategy_selection,
            n_iterations=n_iterations_selection,
            n=n_selection,
            k=k_selection,
        )


def reset_answers():
    with open("answers.json", "w") as f:
        json.dump({}, f)


def exercise_example():
    mean = widgets.FloatText(
        #         value='',
        placeholder=0.0,
        description="Probability:",
        disabled=False,
    )

    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()

    display(mean)

    display(button, output)

    def on_button_clicked(b):

        with output:
            print("Answer for the example exercise saved.")

    button.on_click(on_button_clicked)


def exercise_1():
    mean = widgets.FloatText(
        #         value='',
        placeholder=0.0,
        description="Mean:",
        disabled=False,
    )

    var = widgets.FloatText(
        #         value='',
        placeholder=0.0,
        description="Variance:",
        disabled=False,
    )

    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()

    display(mean)
    display(var)
    #     display(cov)

    display(button, output)

    def on_button_clicked(b):

        with open("answers.json", "r") as f:
            source_dict = json.load(f)

        answer_dict = {
            "ex1": {
                "mean": mean.value,
                "var": var.value,
            }
        }

        source_dict.update(answer_dict)

        with open("answers.json", "w") as f:
            json.dump(source_dict, f)

        with output:
            print("Answer for exercise 1 saved.")

    button.on_click(on_button_clicked)


def exercise_2():
    hist = widgets.ToggleButtons(
        options=["left", "center", "right"],
        description="Your answer:",
        disabled=False,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
    )

    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()

    display(hist)

    display(button, output)

    def on_button_clicked(b):

        with open("answers.json", "r") as f:
            source_dict = json.load(f)

        answer_dict = {"ex2": {"hist": hist.value}}

        source_dict.update(answer_dict)

        with open("answers.json", "w") as f:
            json.dump(source_dict, f)

        with output:
            print("Answer for exercise 2 saved.")

    button.on_click(on_button_clicked)


def exercise_3():
    sum_2_8 = widgets.FloatText(
        #         value='',
        placeholder=0.0,
        description="P for sum=2",
        style={"description_width": "initial"},
        disabled=False,
    )

    sum_3_7 = widgets.FloatText(
        #         value='',
        placeholder=0.0,
        description="P for sum=3:",
        style={"description_width": "initial"},
        disabled=False,
    )

    sum_4_6 = widgets.FloatText(
        #         value='',
        placeholder=0.0,
        description="P for sum=4:",
        style={"description_width": "initial"},
        disabled=False,
    )

    sum_5 = widgets.FloatText(
        #         value='',
        placeholder=0.0,
        description="P for sum=5:",
        style={"description_width": "initial"},
        disabled=False,
    )

    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()

    display(sum_2_8)
    display(sum_3_7)
    display(sum_4_6)
    display(sum_5)

    display(button, output)

    def on_button_clicked(b):

        with open("answers.json", "r") as f:
            source_dict = json.load(f)

        answer_dict = {
            "ex3": {
                "sum_2_8": sum_2_8.value,
                "sum_3_7": sum_3_7.value,
                "sum_4_6": sum_4_6.value,
                "sum_5": sum_5.value,
            }
        }

        source_dict.update(answer_dict)

        with open("answers.json", "w") as f:
            json.dump(source_dict, f)

        with output:
            print("Answer for exercise 3 saved.")

    button.on_click(on_button_clicked)


def exercise_4():
    mean = widgets.FloatText(
        #         value='',
        placeholder=0.0,
        description="Mean:",
        disabled=False,
    )

    var = widgets.FloatText(
        #         value='',
        placeholder=0.0,
        description="Variance:",
        disabled=False,
    )

    cov = widgets.FloatText(
        #         value='',
        placeholder=0.0,
        description="Covariance:",
        disabled=False,
    )

    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()

    display(mean)
    display(var)
    display(cov)

    display(button, output)

    def on_button_clicked(b):

        with open("answers.json", "r") as f:
            source_dict = json.load(f)

        answer_dict = {"ex4": {"mean": mean.value, "var": var.value, "cov": cov.value}}

        source_dict.update(answer_dict)

        with open("answers.json", "w") as f:
            json.dump(source_dict, f)

        with output:
            print("Answer for exercise 4 saved.")

    button.on_click(on_button_clicked)


def exercise_5():
    hist = widgets.ToggleButtons(
        options=["left", "center", "right"],
        description="Your answer:",
        disabled=False,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
    )

    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()

    display(hist)

    display(button, output)

    def on_button_clicked(b):

        with open("answers.json", "r") as f:
            source_dict = json.load(f)

        answer_dict = {"ex5": {"hist": hist.value}}

        source_dict.update(answer_dict)

        with open("answers.json", "w") as f:
            json.dump(source_dict, f)

        with output:
            print("Answer for exercise 5 saved.")

    button.on_click(on_button_clicked)


def exercise_6():
    max_sum = widgets.IntSlider(
        value=2,
        min=2,
        max=12,
        step=1,
        description="Sum:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
    )

    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()

    display(max_sum)

    display(button, output)

    def on_button_clicked(b):

        with open("answers.json", "r") as f:
            source_dict = json.load(f)

        answer_dict = {"ex6": {"max_sum": max_sum.value}}

        source_dict.update(answer_dict)

        with open("answers.json", "w") as f:
            json.dump(source_dict, f)

        with output:
            print("Answer for exercise 6 saved.")

    button.on_click(on_button_clicked)


def exercise_7():
    hist = widgets.ToggleButtons(
        options=["left-most", "left-center", "right-center", "right-most"],
        description="Your answer:",
        disabled=False,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
    )

    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()

    display(hist)

    display(button, output)

    def on_button_clicked(b):

        with open("answers.json", "r") as f:
            source_dict = json.load(f)

        answer_dict = {"ex7": {"hist": hist.value}}

        source_dict.update(answer_dict)

        with open("answers.json", "w") as f:
            json.dump(source_dict, f)

        with output:
            print("Answer for exercise 7 saved.")

    button.on_click(on_button_clicked)


def exercise_8():
    hist = widgets.ToggleButtons(
        options=["left-most", "left-center", "right-center", "right-most"],
        description="Your answer:",
        disabled=False,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
    )

    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()

    display(hist)

    display(button, output)

    def on_button_clicked(b):

        with open("answers.json", "r") as f:
            source_dict = json.load(f)

        answer_dict = {"ex8": {"hist": hist.value}}

        source_dict.update(answer_dict)

        with open("answers.json", "w") as f:
            json.dump(source_dict, f)

        with output:
            print("Answer for exercise 8 saved.")

    button.on_click(on_button_clicked)


def exercise_9():
    mean = widgets.ToggleButtons(
        options=["stays the same", "increases", "decreases"],
        description="The mean of the sum:",
        disabled=False,
        button_style="",
    )

    var = widgets.ToggleButtons(
        options=["stays the same", "increases", "decreases"],
        description="The variance of the sum:",
        disabled=False,
        button_style="",
    )

    cov = widgets.ToggleButtons(
        options=["stays the same", "increases", "decreases"],
        description="The covariance of the joint distribution:",
        disabled=False,
        button_style="",
    )

    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()

    print("As the number of sides in the die increases:")
    display(mean)
    display(var)
    display(cov)

    display(button, output)

    def on_button_clicked(b):

        with open("answers.json", "r") as f:
            source_dict = json.load(f)

        answer_dict = {
            "ex9": {
                "mean": mean.value,
                "var": var.value,
                "cov": cov.value,
            }
        }

        source_dict.update(answer_dict)

        with open("answers.json", "w") as f:
            json.dump(source_dict, f)

        with output:
            print("Answer for exercise 9 saved.")

    button.on_click(on_button_clicked)


def exercise_10():
    options = widgets.RadioButtons(
        options=[
            "the mean and variance is the same regardless of which side is loaded",
            "having the sides 3 or 4 loaded will yield a higher covariance than any other sides",
            "the mean will decrease as the value of the loaded side increases",
            "changing the loaded side from 1 to 6 will yield a higher mean but the same variance",
        ],
        layout={"width": "max-content"},
    )

    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()

    display(options)

    display(button, output)

    def on_button_clicked(b):

        with open("answers.json", "r") as f:
            source_dict = json.load(f)

        answer_dict = {
            "ex10": {
                "options": options.value,
            }
        }

        source_dict.update(answer_dict)

        with open("answers.json", "w") as f:
            json.dump(source_dict, f)

        with output:
            print("Answer for exercise 10 saved.")

    button.on_click(on_button_clicked)


def exercise_11():
    options = widgets.RadioButtons(
        options=[
            "yes, but only if one of the sides is loaded",
            "no, regardless if the die is fair or not",
            "yes, but only if the die is fair",
            "yes, regardless if the die is fair or not",
        ],
        layout={"width": "max-content"},
    )

    button = widgets.Button(description="Save your answer!", button_style="success")
    output = widgets.Output()

    display(options)

    display(button, output)

    def on_button_clicked(b):

        with open("answers.json", "r") as f:
            source_dict = json.load(f)

        answer_dict = {
            "ex11": {
                "options": options.value,
            }
        }

        source_dict.update(answer_dict)

        with open("answers.json", "w") as f:
            json.dump(source_dict, f)

        with output:
            print("Answer for exercise 11 saved.")

    button.on_click(on_button_clicked)


def check_submissions():
    with open("./answers.json", "r") as f:
        answer_dict = json.load(f)

    saved_exercises = [k for k in answer_dict.keys()]
    expected = [
        "ex1",
        "ex2",
        "ex3",
        "ex4",
        "ex5",
        "ex6",
        "ex7",
        "ex8",
        "ex9",
        "ex10",
        "ex11",
    ]
    missing = [e for e in expected if not e in saved_exercises]

    if missing:
        print(
            f"missing answers for exercises {[ex.split('ex')[1] for ex in missing]}\n\nSave your answers before submitting for grading!"
        )
        return

    print("All answers saved, you can submit the assignment for grading!")
