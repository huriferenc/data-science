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
