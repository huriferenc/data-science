import numpy as np
from sklearn.datasets import make_blobs


v = np.array(
    [
        [
            0.84290395,
            0.44813828,
            0.81547429,
            0.03314107,
            0.85953645,
            0.92439938,
            0.0708698,
            0.15325087,
            0.33119878,
            0.20287232,
            0.41923244,
            0.35807655,
            0.49793058,
            0.08762816,
            0.70849596,
            0.07306568,
            0.49047445,
            0.66584279,
            0.72546527,
            0.58252153,
            0.2479397,
            0.83928946,
            0.45860364,
            0.90068042,
            0.37107194,
            0.03048847,
            0.08892689,
            0.89084274,
            0.45335152,
            0.91484284,
        ],
        [
            0.07607491,
            0.20404837,
            0.99709655,
            0.60518291,
            0.6043017,
            0.15693135,
            0.18987089,
            0.92820667,
            0.07679335,
            0.48296973,
            0.52911491,
            0.43483309,
            0.11369568,
            0.62955213,
            0.5055929,
            0.93913324,
            0.86731505,
            0.99889454,
            0.28338204,
            0.36295851,
            0.83535739,
            0.60186712,
            0.47561196,
            0.70575941,
            0.8826434,
            0.09415726,
            0.34605956,
            0.70188784,
            0.62037162,
            0.33194618,
        ],
    ]
)


def test_T_stretch(target):

    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {"v": v, "a": 77},
            "expected": np.array(
                [
                    [
                        64.90360415,
                        34.50664756,
                        62.79152033,
                        2.55186239,
                        66.18430665,
                        71.17875226,
                        5.4569746,
                        11.80031699,
                        25.50230606,
                        15.62116864,
                        32.28089788,
                        27.57189435,
                        38.34065466,
                        6.74736832,
                        54.55418892,
                        5.62605736,
                        37.76653265,
                        51.26989483,
                        55.86082579,
                        44.85415781,
                        19.0913569,
                        64.62528842,
                        35.31248028,
                        69.35239234,
                        28.57253938,
                        2.34761219,
                        6.84737053,
                        68.59489098,
                        34.90806704,
                        70.44289868,
                    ],
                    [
                        5.85776807,
                        15.71172449,
                        76.77643435,
                        46.59908407,
                        46.5312309,
                        12.08371395,
                        14.62005853,
                        71.47191359,
                        5.91308795,
                        37.18866921,
                        40.74184807,
                        33.48214793,
                        8.75456736,
                        48.47551401,
                        38.9306533,
                        72.31325948,
                        66.78325885,
                        76.91487958,
                        21.82041708,
                        27.94780527,
                        64.32251903,
                        46.34376824,
                        36.62212092,
                        54.34347457,
                        67.9635418,
                        7.25010902,
                        26.64658612,
                        54.04536368,
                        47.76861474,
                        25.55985586,
                    ],
                ]
            ),
        }
    ]

    for test_case in test_cases:

        result = target(**test_case["input"])

        try:
            assert np.allclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Failed in test case: {failed_cases[-1]['name']}, with a = {test_case['input']['a']} and v = {v}. \n\tExpected: {failed_cases[-1]['expected']}. \n\tGot: {failed_cases[-1]['got']}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_T_hshear(target):

    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {"v": v, "m": 2},
            "expected": np.array(
                [
                    [
                        0.99505377,
                        0.85623502,
                        2.80966739,
                        1.24350689,
                        2.06813985,
                        1.23826208,
                        0.45061158,
                        2.00966421,
                        0.48478548,
                        1.16881178,
                        1.47746226,
                        1.22774273,
                        0.72532194,
                        1.34673242,
                        1.71968176,
                        1.95133216,
                        2.22510455,
                        2.66363187,
                        1.29222935,
                        1.30843855,
                        1.91865448,
                        2.0430237,
                        1.40982756,
                        2.31219924,
                        2.13635874,
                        0.21880299,
                        0.78104601,
                        2.29461842,
                        1.69409476,
                        1.5787352,
                    ],
                    [
                        0.07607491,
                        0.20404837,
                        0.99709655,
                        0.60518291,
                        0.6043017,
                        0.15693135,
                        0.18987089,
                        0.92820667,
                        0.07679335,
                        0.48296973,
                        0.52911491,
                        0.43483309,
                        0.11369568,
                        0.62955213,
                        0.5055929,
                        0.93913324,
                        0.86731505,
                        0.99889454,
                        0.28338204,
                        0.36295851,
                        0.83535739,
                        0.60186712,
                        0.47561196,
                        0.70575941,
                        0.8826434,
                        0.09415726,
                        0.34605956,
                        0.70188784,
                        0.62037162,
                        0.33194618,
                    ],
                ]
            ),
        },
        {
            "name": "default_check",
            "input": {"v": v, "m": 17},
            "expected": np.array(
                [
                    [
                        2.13617742,
                        3.91696057,
                        17.76611564,
                        10.32125054,
                        11.13266535,
                        3.59223233,
                        3.29867493,
                        15.93276426,
                        1.63668573,
                        8.41335773,
                        9.41418591,
                        7.75023908,
                        2.43075714,
                        10.79001437,
                        9.30357526,
                        16.03833076,
                        15.2348303,
                        17.64704997,
                        5.54295995,
                        6.7528162,
                        14.44901533,
                        11.0710305,
                        8.54400696,
                        12.89859039,
                        15.37600974,
                        1.63116189,
                        5.97193941,
                        12.82293602,
                        10.99966906,
                        6.5579279,
                    ],
                    [
                        0.07607491,
                        0.20404837,
                        0.99709655,
                        0.60518291,
                        0.6043017,
                        0.15693135,
                        0.18987089,
                        0.92820667,
                        0.07679335,
                        0.48296973,
                        0.52911491,
                        0.43483309,
                        0.11369568,
                        0.62955213,
                        0.5055929,
                        0.93913324,
                        0.86731505,
                        0.99889454,
                        0.28338204,
                        0.36295851,
                        0.83535739,
                        0.60186712,
                        0.47561196,
                        0.70575941,
                        0.8826434,
                        0.09415726,
                        0.34605956,
                        0.70188784,
                        0.62037162,
                        0.33194618,
                    ],
                ]
            ),
        },
    ]

    for test_case in test_cases:

        result = target(**test_case["input"])

        try:
            assert np.allclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Failed in test case: {failed_cases[-1]['name']}, with m = {test_case['input']['m']} and v = {v}. \n\tExpected: {failed_cases[-1]['expected']}. \n\tGot: {failed_cases[-1]['got']}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_T_rotation(target):

    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {"v": v, "theta": np.pi / 4},
            "expected": np.array(
                [
                    [
                        0.54223001,
                        0.17259763,
                        -0.12842633,
                        -0.40449466,
                        0.18047822,
                        0.54268185,
                        -0.08414648,
                        -0.5479765,
                        0.1798918,
                        -0.19805878,
                        -0.07769864,
                        -0.05427507,
                        0.2716951,
                        -0.38319811,
                        0.14347413,
                        -0.61240224,
                        -0.26646654,
                        -0.23550315,
                        0.31260005,
                        0.1552545,
                        -0.41536703,
                        0.16788295,
                        -0.0120267,
                        0.13782997,
                        -0.36173565,
                        -0.04502063,
                        -0.18182025,
                        0.13361129,
                        -0.11810105,
                        0.41217018,
                    ],
                    [
                        0.64981618,
                        0.4611656,
                        1.28168113,
                        0.45136321,
                        1.03508988,
                        0.76461629,
                        0.18437151,
                        0.76470596,
                        0.288494,
                        0.48496356,
                        0.67058284,
                        0.56067178,
                        0.43248508,
                        0.50712305,
                        0.85849047,
                        0.71573272,
                        0.96010216,
                        1.17714705,
                        0.71336277,
                        0.66855535,
                        0.76600672,
                        1.01905159,
                        0.66059019,
                        1.1359245,
                        0.88651062,
                        0.08813784,
                        0.30758187,
                        1.12623059,
                        0.75923691,
                        0.88161297,
                    ],
                ]
            ),
        }
    ]

    for test_case in test_cases:

        result = target(**test_case["input"])

        try:
            assert np.allclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Failed in test case: {failed_cases[-1]['name']}, with theta = {test_case['input']['theta']} and v = {v}. \n\tExpected: {failed_cases[-1]['expected']}. \n\tGot: {failed_cases[-1]['got']}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_T_rotation_and_stretch(target):

    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {"v": v, "theta": np.pi / 6, "a": 4},
            "expected": np.array(
                [
                    [
                        2.76775511,
                        1.1442998,
                        0.83069271,
                        -1.09556179,
                        1.7689182,
                        2.88835069,
                        -0.13424159,
                        -1.32553675,
                        0.99371953,
                        -0.26316913,
                        0.39403395,
                        0.37074738,
                        1.49749077,
                        -0.95555141,
                        1.4431162,
                        -1.62515954,
                        -0.03557677,
                        0.308758,
                        1.94632133,
                        1.29199675,
                        -0.81182646,
                        1.70364973,
                        0.63742569,
                        1.70852968,
                        -0.47985589,
                        -0.08269936,
                        -0.38406734,
                        1.68219409,
                        0.32971249,
                        2.5052162,
                    ],
                    [
                        1.94933912,
                        1.60312085,
                        5.08499235,
                        2.16269724,
                        3.8124354,
                        2.3924249,
                        0.79947166,
                        3.52190396,
                        0.92841753,
                        2.07880086,
                        2.67137269,
                        2.22245911,
                        1.38971455,
                        2.35608887,
                        3.1684171,
                        3.39938433,
                        3.98541637,
                        4.79195777,
                        2.43259472,
                        2.42236822,
                        3.38964228,
                        3.76350778,
                        2.56477544,
                        4.24618315,
                        3.79971031,
                        0.38714726,
                        1.37663926,
                        4.21309628,
                        3.05573337,
                        2.97958098,
                    ],
                ]
            ),
        }
    ]

    for test_case in test_cases:

        result = target(**test_case["input"])

        try:
            assert np.allclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Failed in test case: {failed_cases[-1]['name']}, with a = {test_case['input']['a']}, theta = {test_case['input']['theta']} and v = {v}. \n\tExpected: {failed_cases[-1]['expected']}. \n\tGot: {failed_cases[-1]['got']}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


# variables for the default_check test cases
m = 30
X = np.array(
    [
        [
            0.3190391,
            -1.07296862,
            0.86540763,
            -0.17242821,
            1.14472371,
            0.50249434,
            -2.3015387,
            -0.68372786,
            -0.38405435,
            -0.87785842,
            -2.06014071,
            -1.10061918,
            -1.09989127,
            1.13376944,
            1.74481176,
            -0.12289023,
            -0.93576943,
            1.62434536,
            1.46210794,
            0.90159072,
            -0.7612069,
            0.53035547,
            -0.52817175,
            -0.26788808,
            0.58281521,
            0.04221375,
            0.90085595,
            -0.24937038,
            -0.61175641,
            -0.3224172,
        ]
    ]
)
Y = np.array(
    [
        [
            -3.01854669,
            -65.65047675,
            26.96755728,
            8.70562603,
            57.94332628,
            -0.69293498,
            -78.66594473,
            -12.73881492,
            -13.26721663,
            -24.80488085,
            -74.24484385,
            -39.99533724,
            -22.70174437,
            73.46766345,
            55.7257405,
            23.80417646,
            -13.45481508,
            25.57952246,
            75.91238321,
            50.91155323,
            -43.7191551,
            -1.7025559,
            -16.44931235,
            -33.54041234,
            20.4505961,
            18.35949302,
            37.69029586,
            -1.04801683,
            -4.47915933,
            -20.89431647,
        ]
    ]
)
n_x = X.shape[0]
n_y = 1


def test_forward_propagation(target_forward_propagation):

    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "X": X,
                "parameters": {"W": np.array([[0.01788628]]), "b": np.zeros((n_y, 1))},
            },
            "expected": {
                "A": np.array(
                    [
                        [
                            0.00570642,
                            -0.01919142,
                            0.01547892,
                            -0.0030841,
                            0.02047485,
                            0.00898775,
                            -0.04116597,
                            -0.01222935,
                            -0.0068693,
                            -0.01570162,
                            -0.03684825,
                            -0.01968598,
                            -0.01967296,
                            0.02027892,
                            0.03120819,
                            -0.00219805,
                            -0.01673743,
                            0.0290535,
                            0.02615167,
                            0.0161261,
                            -0.01361516,
                            0.00948609,
                            -0.00944703,
                            -0.00479152,
                            0.0104244,
                            0.00075505,
                            0.01611296,
                            -0.00446031,
                            -0.01094205,
                            -0.00576684,
                        ]
                    ]
                )
            },
        },
        {
            "name": "change_weights_check",
            "input": {
                "X": X,
                "parameters": {"W": np.array([[-0.00768836]]), "b": np.zeros((n_y, 1))},
            },
            "expected": {
                "A": np.array(
                    [
                        [
                            -0.00245289,
                            0.00824937,
                            -0.00665357,
                            0.00132569,
                            -0.00880105,
                            -0.00386336,
                            0.01769506,
                            0.00525675,
                            0.00295275,
                            0.00674929,
                            0.0158391,
                            0.00846196,
                            0.00845636,
                            -0.00871683,
                            -0.01341474,
                            0.00094482,
                            0.00719453,
                            -0.01248855,
                            -0.01124121,
                            -0.00693175,
                            0.00585243,
                            -0.00407756,
                            0.00406077,
                            0.00205962,
                            -0.00448089,
                            -0.00032455,
                            -0.0069261,
                            0.00191725,
                            0.0047034,
                            0.00247886,
                        ]
                    ]
                )
            },
        },
        {
            "name": "change_dataset_check",
            "input": {
                "X": np.array([[0, 1, 0, 0, 1], [0, 0, 0, 0, 1]]),
                "parameters": {
                    "W": np.array([[-0.00768836, -0.00230031]]),
                    "b": np.zeros((n_y, 1)),
                },
            },
            "expected": {"A": np.array([[0, -0.00768836, 0, 0, -0.00998867]])},
        },
    ]

    for test_case in test_cases:
        result = target_forward_propagation(
            test_case["input"]["X"], test_case["input"]["parameters"]
        )

        try:
            assert result.shape == test_case["expected"]["A"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["A"].shape,
                    "got": result.shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the array A. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result, test_case["expected"]["A"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["A"],
                    "got": result,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong array A. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_nn_model(target_nn_model):

    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "X": X,
                "Y": Y,
                "num_iterations": 10,
            },
            "expected": {
                "W": np.array(
                    [[0.19732238]]
                ),  # no check of the actual values in the unit tests
                "b": np.array(
                    [[-0.2]]
                ),  # no check of the actual values in the unit tests
            },
        },
        {
            "name": "extra_check",
            "input": {
                "X": np.array([[0, 1, 0, 0, 1], [0, 0, 0, 0, 1]]),
                "Y": np.array([[0, 0, 0, 0, 1]]),
                "num_iterations": 100,
            },
            "expected": {
                "W": np.array(
                    [[-0.23147202, 0.49108187]]
                ),  # no check of the actual values in the unit tests
                "b": np.array(
                    [[-0.24]]
                ),  # no check of the actual values in the unit tests
            },
        },
    ]

    for test_case in test_cases:

        result = target_nn_model(
            test_case["input"]["X"],
            test_case["input"]["Y"],
            test_case["input"]["num_iterations"],
            False,
        )

        try:
            assert result["W"].shape == test_case["expected"]["W"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["W"].shape,
                    "got": result["W"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the weights matrix W. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result["b"].shape == test_case["expected"]["b"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b"].shape,
                    "got": result["b"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the bias vector b. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


# +
# variables for the default_check test cases
m = 2000
samples, labels = make_blobs(
    n_samples=m,
    centers=([2.5, 3], [6.7, 7.9], [2.1, 7.9], [7.4, 2.8]),
    cluster_std=1.1,
    random_state=0,
)
labels[(labels == 0) | (labels == 1)] = 1
labels[(labels == 2) | (labels == 3)] = 0
X = np.transpose(samples)
Y = labels.reshape((1, m))

n_x = X.shape[0]
n_h = 2
n_y = Y.shape[0]


# -


def test_sigmoid(target_sigmoid):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "z": -2,
            },
            "expected": {
                "sigmoid": 0.11920292202211755,
            },
        },
        {
            "name": "extra_check_1",
            "input": {
                "z": 0,
            },
            "expected": {
                "sigmoid": 0.5,
            },
        },
        {
            "name": "extra_check_2",
            "input": {
                "z": 3.5,
            },
            "expected": {
                "sigmoid": 0.9706877692486436,
            },
        },
    ]

    for test_case in test_cases:
        result = target_sigmoid(test_case["input"]["z"])

        try:
            assert np.allclose(result, test_case["expected"]["sigmoid"], atol=1e-12)
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["sigmoid"],
                    "got": result,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of sigmoid for z = {test_case['input']['z']}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_layer_sizes(target_layer_sizes):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {"X": X, "Y": Y},
            "expected": {"n_x": n_x, "n_h": n_h, "n_y": n_y},
        },
        {
            "name": "extra_check",
            "input": {"X": np.ones((5, 100)), "Y": np.ones((3, 100))},
            "expected": {"n_x": 5, "n_h": 2, "n_y": 3},
        },
    ]

    for test_case in test_cases:
        (result_n_x, result_n_h, result_n_y) = target_layer_sizes(
            test_case["input"]["X"], test_case["input"]["Y"]
        )

        try:
            assert result_n_x == test_case["expected"]["n_x"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["n_x"],
                    "got": result_n_x,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong size of the input layer n_x for the test case, where array X has a shape {test_case['input']['X'].shape}. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result_n_h == test_case["expected"]["n_h"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["n_h"],
                    "got": result_n_h,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong size of the hidden layer n_h. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result_n_y == test_case["expected"]["n_y"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["n_y"],
                    "got": result_n_y,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong size of the output layer n_y for the test case, where array Y has a shape {test_case['input']['Y'].shape}. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_initialize_parameters(target_initialize_parameters):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "n_x": n_x,
                "n_h": n_h,
                "n_y": n_y,
            },
            "expected": {
                "W1": np.zeros(
                    (n_h, n_x)
                ),  # no check of the actual values in the unit tests
                "b1": np.zeros((n_h, 1)),
                "W2": np.zeros(
                    (n_y, n_h)
                ),  # no check of the actual values in the unit tests
                "b2": np.zeros((n_y, 1)),
            },
        },
        {
            "name": "extra_check",
            "input": {
                "n_x": 5,
                "n_h": 4,
                "n_y": 3,
            },
            "expected": {
                "W1": np.zeros(
                    (4, 5)
                ),  # no check of the actual values in the unit tests
                "b1": np.zeros((4, 1)),
                "W2": np.zeros(
                    (3, 4)
                ),  # no check of the actual values in the unit tests
                "b2": np.zeros((3, 1)),
            },
        },
    ]

    for test_case in test_cases:
        result = target_initialize_parameters(
            test_case["input"]["n_x"],
            test_case["input"]["n_h"],
            test_case["input"]["n_y"],
        )

        try:
            assert result["W1"].shape == test_case["expected"]["W1"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["W1"].shape,
                    "got": result["W1"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the weights matrix W1. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result["b1"].shape == test_case["expected"]["b1"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b1"].shape,
                    "got": result["b1"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the bias vector b1. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result["b1"], test_case["expected"]["b1"], atol=1e-12)
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b1"],
                    "got": result["b1"],
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong bias vector b1. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

        try:
            assert result["W2"].shape == test_case["expected"]["W2"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["W2"].shape,
                    "got": result["W2"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the weights matrix W2. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result["b2"].shape == test_case["expected"]["b2"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b2"].shape,
                    "got": result["b2"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the bias vector b2. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result["b2"], test_case["expected"]["b2"], atol=1e-12)
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b2"],
                    "got": result["b2"],
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong bias vector b2. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_forward_propagation(target_forward_propagation):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "X": X,
                "parameters": {
                    "W1": np.array(
                        [[0.01788628, 0.0043651], [0.00096497, -0.01863493]]
                    ),
                    "b1": np.zeros((n_h, 1)),
                    "W2": np.array([[-0.00277388, -0.00354759]]),
                    "b2": np.zeros((n_y, 1)),
                },
            },
            "expected": {
                "Z1_array": {
                    "shape": (2, 2000),
                    "Z1": [
                        {
                            "i": 0,
                            "j": 0,
                            "Z1_i_j": 0.11050400276471689,
                        },
                        {
                            "i": 1,
                            "j": 1999,
                            "Z1_i_j": -0.11866556808051022,
                        },
                        {
                            "i": 0,
                            "j": 100,
                            "Z1_i_j": 0.08570563958483839,
                        },
                    ],
                },
                "A1_array": {
                    "shape": (2, 2000),
                    "A1": [
                        {
                            "i": 0,
                            "j": 0,
                            "A1_i_j": 0.5275979229090347,
                        },
                        {
                            "i": 1,
                            "j": 1999,
                            "A1_i_j": 0.47036837134568177,
                        },
                        {
                            "i": 0,
                            "j": 100,
                            "A1_i_j": 0.521413303959268,
                        },
                    ],
                },
                "Z2_array": {
                    "shape": (1, 2000),
                    "Z2": [
                        {
                            "i": 0,
                            "Z2_i": -0.003193737045395555,
                        },
                        {
                            "i": 400,
                            "Z2_i": -0.003221924688299396,
                        },
                        {
                            "i": 1999,
                            "Z2_i": -0.00317339213692169,
                        },
                    ],
                },
                "A2_array": {
                    "shape": (1, 2000),
                    "A2": [
                        {
                            "i": 0,
                            "A2_i": 0.4992015664173166,
                        },
                        {
                            "i": 400,
                            "A2_i": 0.49919451952471916,
                        },
                        {
                            "i": 1999,
                            "A2_i": 0.4992066526315478,
                        },
                    ],
                },
            },
        },
        {
            "name": "change_weights_check",
            "input": {
                "X": X,
                "parameters": {
                    "W1": np.array(
                        [
                            [-0.00082741, -0.00627001],
                            [-0.00043818, -0.00477218],
                            [0.00899338, -0.00154507],
                        ]
                    ),
                    "b1": np.array([[0.01769627], [0.00483788], [0.01769627]]),
                    "W2": np.array([[-0.01313865, 0.00884622, 0.00483788]]),
                    "b2": np.array([[0.01167882]]),
                },
            },
            "expected": {
                "Z1_array": {
                    "shape": (3, 2000),
                    "Z1": [
                        {
                            "i": 0,
                            "j": 0,
                            "Z1_i_j": -0.005126396781157443,
                        },
                        {
                            "i": 1,
                            "j": 1999,
                            "Z1_i_j": -0.03094074954823146,
                        },
                        {
                            "i": 0,
                            "j": 100,
                            "Z1_i_j": -0.04103063929597483,
                        },
                    ],
                },
                "A1_array": {
                    "shape": (3, 2000),
                    "A1": [
                        {
                            "i": 0,
                            "j": 0,
                            "A1_i_j": 0.4987184036113995,
                        },
                        {
                            "i": 1,
                            "j": 1999,
                            "A1_i_j": 0.49226542964777215,
                        },
                        {
                            "i": 0,
                            "j": 100,
                            "A1_i_j": 0.4897437790093911,
                        },
                    ],
                },
                "Z2_array": {
                    "shape": (1, 2000),
                    "Z2": [
                        {
                            "i": 0,
                            "Z2_i": 0.012018360374017639,
                        },
                        {
                            "i": 400,
                            "Z2_i": 0.012033400685020897,
                        },
                        {
                            "i": 1999,
                            "Z2_i": 0.01208014064812657,
                        },
                    ],
                },
                "A2_array": {
                    "shape": (1, 2000),
                    "A2": [
                        {
                            "i": 0,
                            "A2_i": 0.5030045539285305,
                        },
                        {
                            "i": 400,
                            "A2_i": 0.5030083138703372,
                        },
                        {
                            "i": 1999,
                            "A2_i": 0.5030199984364742,
                        },
                    ],
                },
            },
        },
        {
            "name": "change_dataset_check",
            "input": {
                "X": np.array([[0, 1, 0, 0, 1], [0, 0, 0, 0, 1]]),
                "parameters": {
                    "W1": np.array(
                        [[-0.00082741, -0.00627001], [-0.00043818, -0.00477218]]
                    ),
                    "b1": np.zeros((n_h, 1)),
                    "W2": np.array([[-0.01313865, 0.00884622]]),
                    "b2": np.zeros((n_y, 1)),
                },
            },
            "expected": {
                "Z1_array": {
                    "shape": (2, 5),
                    "Z1": [
                        {
                            "i": 0,
                            "j": 0,
                            "Z1_i_j": 0.0,
                        },
                        {
                            "i": 1,
                            "j": 4,
                            "Z1_i_j": -0.00521036,
                        },
                        {
                            "i": 0,
                            "j": 4,
                            "Z1_i_j": -0.00709742,
                        },
                    ],
                },
                "A1_array": {
                    "shape": (2, 5),
                    "A1": [
                        {
                            "i": 0,
                            "j": 0,
                            "A1_i_j": 0.5,
                        },
                        {
                            "i": 1,
                            "j": 4,
                            "A1_i_j": 0.49869741294686865,
                        },
                        {
                            "i": 0,
                            "j": 4,
                            "A1_i_j": 0.49822565244831607,
                        },
                    ],
                },
                "Z2_array": {
                    "shape": (1, 5),
                    "Z2": [
                        {
                            "i": 0,
                            "Z2_i": -0.002146215,
                        },
                        {
                            "i": 1,
                            "Z2_i": -0.0021444662967103198,
                        },
                        {
                            "i": 4,
                            "Z2_i": -0.00213442544018122,
                        },
                    ],
                },
                "A2_array": {
                    "shape": (1, 5),
                    "A2": [
                        {
                            "i": 0,
                            "A2_i": 0.4994634464559578,
                        },
                        {
                            "i": 1,
                            "A2_i": 0.4994638836312772,
                        },
                        {
                            "i": 4,
                            "A2_i": 0.49946639384253705,
                        },
                    ],
                },
            },
        },
    ]

    for test_case in test_cases:
        result_A2, result_cache = target_forward_propagation(
            test_case["input"]["X"], test_case["input"]["parameters"]
        )

        try:
            assert (
                result_cache["Z1"].shape == test_case["expected"]["Z1_array"]["shape"]
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["Z1_array"]["shape"],
                    "got": result_cache["Z1"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the array Z1. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        for test_case_i_j in test_case["expected"]["Z1_array"]["Z1"]:
            i = test_case_i_j["i"]
            j = test_case_i_j["j"]

            try:
                assert np.isclose(
                    result_cache["Z1"][i, j], test_case_i_j["Z1_i_j"], atol=1e-12
                )
                successful_cases += 1

            except:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": test_case_i_j["Z1_i_j"],
                        "got": result_cache["Z1"][i, j],
                    }
                )
                print(
                    f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of Z1 for X = \n{test_case['input']['X']}\nTest for i = {i}, j = {j}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
                )

        try:
            assert (
                result_cache["A1"].shape == test_case["expected"]["A1_array"]["shape"]
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["A1_array"]["shape"],
                    "got": result_cache["A1"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the array A1. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        for test_case_i_j in test_case["expected"]["A1_array"]["A1"]:
            i = test_case_i_j["i"]
            j = test_case_i_j["j"]

            try:
                assert np.isclose(
                    result_cache["A1"][i, j], test_case_i_j["A1_i_j"], atol=1e-12
                )
                successful_cases += 1

            except:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": test_case_i_j["A1_i_j"],
                        "got": result_cache["A1"][i, j],
                    }
                )
                print(
                    f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of A1 for X = \n{test_case['input']['X']}\nTest for i = {i}, j = {j}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
                )

        try:
            assert (
                result_cache["Z2"].shape == test_case["expected"]["Z2_array"]["shape"]
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["Z2_array"]["shape"],
                    "got": result_cache["Z2"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the array Z2. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        for test_case_i in test_case["expected"]["Z2_array"]["Z2"]:
            i = test_case_i["i"]

            try:
                assert np.isclose(
                    result_cache["Z2"][0, i], test_case_i["Z2_i"], atol=1e-12
                )
                successful_cases += 1

            except:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": test_case_i["Z2_i"],
                        "got": result_cache["Z2"][0, i],
                    }
                )
                print(
                    f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of Z2. Test for i = {i}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
                )

        try:
            assert result_A2.shape == test_case["expected"]["A2_array"]["shape"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["A2_array"]["shape"],
                    "got": result_A2.shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the array A2. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        for test_case_i in test_case["expected"]["A2_array"]["A2"]:
            i = test_case_i["i"]

            try:
                assert np.isclose(result_A2[0, i], test_case_i["A2_i"], atol=1e-12)
                successful_cases += 1

            except:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": test_case_i["A2_i"],
                        "got": result_A2[0, i],
                    }
                )
                print(
                    f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of A2. Test for i = {i}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
                )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_compute_cost(target_compute_cost, input_A2):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "A2": input_A2,
                "Y": Y,
            },
            "expected": {
                "cost": 0.6931477703826823,
            },
        },
        {
            "name": "extra_check",
            "input": {
                "A2": np.array([[0.64, 0.60, 0.35, 0.15, 0.95]]),
                "Y": np.array([[0.58, 0.01, 0.42, 0.24, 0.99]]),
            },
            "expected": {
                "cost": 0.5901032749748385,
            },
        },
    ]

    for test_case in test_cases:
        result = target_compute_cost(test_case["input"]["A2"], test_case["input"]["Y"])

        try:
            assert np.allclose(result, test_case["expected"]["cost"], atol=1e-12)
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["cost"],
                    "got": result,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of compute_cost. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_update_parameters(target_update_parameters):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "parameters": {
                    "W1": np.array(
                        [[0.01788628, 0.0043651], [0.00096497, -0.01863493]]
                    ),
                    "b1": np.zeros((n_h, 1)),
                    "W2": np.array([[-0.00277388, -0.00354759]]),
                    "b2": np.zeros((n_y, 1)),
                },
                "grads": {
                    "dW1": np.array(
                        [
                            [-1.49856632e-05, 1.67791519e-05],
                            [-2.12394543e-05, 2.43895135e-05],
                        ]
                    ),
                    "db1": np.array([[5.11207671e-07], [7.06236219e-07]]),
                    "dW2": np.array([[-0.00032641, -0.0002606]]),
                    "db2": np.array([[-0.00078732]]),
                },
                "learning_rate": 1.2,
            },
            "expected": {
                "parameters": {
                    "W1": np.array(
                        [[0.01790426, 0.00434497], [0.00099046, -0.0186642]]
                    ),
                    "b1": np.array([[-6.13449205e-07], [-8.47483463e-07]]),
                    "W2": np.array([[-0.00238219, -0.00323487]]),
                    "b2": np.array([[0.00094478]]),
                },
            },
        },
        {
            "name": "extra_check",
            "input": {
                "parameters": {
                    "W1": np.array(
                        [
                            [-0.00082741, -0.00627001],
                            [-0.00043818, -0.00477218],
                            [0.00899338, -0.00154507],
                        ]
                    ),
                    "b1": np.array([[0.01769627], [0.00483788], [0.01769627]]),
                    "W2": np.array([[-0.01313865, 0.00884622, 0.00483788]]),
                    "b2": np.array([[0.01167882]]),
                },
                "grads": {
                    "dW1": np.array(
                        [
                            [-7.56054712e-05, 8.48587435e-05],
                            [5.05322772e-05, -5.72665231e-05],
                            [-0.00588594e-05, -0.00873882e-05],
                        ]
                    ),
                    "db1": np.array(
                        [[1.68002224e-06], [-1.14292837e-06], [0.00029714e-06]]
                    ),
                    "dW2": np.array([[-0.0002246, -0.00023206, -0.02248258]]),
                    "db2": np.array([[-0.000521]]),
                },
                "learning_rate": 0.1,
            },
            "expected": {
                "parameters": {
                    "W1": np.array(
                        [
                            [-0.00081985, -0.0062785],
                            [-0.00044323, -0.00476645],
                            [0.00899339, -0.00154506],
                        ]
                    ),
                    "b1": np.array([[0.0176961], [0.00483799], [0.01769627]]),
                    "W2": np.array([[-0.01311619, 0.00886943, 0.00708614]]),
                    "b2": np.array([[0.01173092]]),
                },
            },
        },
    ]

    for test_case in test_cases:
        result = target_update_parameters(
            test_case["input"]["parameters"],
            test_case["input"]["grads"],
            test_case["input"]["learning_rate"],
        )

        try:
            assert result["W1"].shape == test_case["expected"]["parameters"]["W1"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["parameters"]["W1"].shape,
                    "got": result["W1"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the output array W1. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result["W1"], test_case["expected"]["parameters"]["W1"], atol=1e-06
            )
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["parameters"]["W1"],
                    "got": result["W1"],
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output array W1. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

        try:
            assert result["b1"].shape == test_case["expected"]["parameters"]["b1"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["parameters"]["b1"].shape,
                    "got": result["b1"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the output array b1. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result["b1"], test_case["expected"]["parameters"]["b1"], atol=1e-06
            )
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["parameters"]["b1"],
                    "got": result["b1"],
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output array b1. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

        try:
            assert result["W2"].shape == test_case["expected"]["parameters"]["W2"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["parameters"]["W2"].shape,
                    "got": result["W2"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the output array W2. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result["W2"], test_case["expected"]["parameters"]["W2"], atol=1e-06
            )
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["parameters"]["W2"],
                    "got": result["W2"],
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output array W2. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

        try:
            assert result["b2"].shape == test_case["expected"]["parameters"]["b2"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["parameters"]["b2"].shape,
                    "got": result["b2"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the output array b2. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result["b2"], test_case["expected"]["parameters"]["b2"], atol=1e-06
            )
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["parameters"]["b2"],
                    "got": result["b2"],
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output array b2. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_nn_model(target_nn_model):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "X": X,
                "Y": Y,
                "n_h": 2,
                "num_iterations": 3000,
                "learning_rate": 1.2,
            },
            "expected": {
                "W1": np.zeros(
                    (n_h, n_x)
                ),  # no check of the actual values in the unit tests
                "b1": np.zeros(
                    (n_h, 1)
                ),  # no check of the actual values in the unit tests
                "W2": np.zeros(
                    (n_y, n_h)
                ),  # no check of the actual values in the unit tests
                "b2": np.zeros(
                    (n_y, 1)
                ),  # no check of the actual values in the unit tests
            },
        },
        {
            "name": "extra_check",
            "input": {
                "X": np.array([[0, 1, 0, 0, 1], [0, 0, 0, 0, 1]]),
                "Y": np.array([[0, 0, 0, 0, 1]]),
                "n_h": 3,
                "num_iterations": 100,
                "learning_rate": 0.1,
            },
            "expected": {
                "W1": np.zeros(
                    (3, 2)
                ),  # no check of the actual values in the unit tests
                "b1": np.zeros(
                    (3, 1)
                ),  # no check of the actual values in the unit tests
                "W2": np.zeros(
                    (1, 3)
                ),  # no check of the actual values in the unit tests
                "b2": np.zeros(
                    (1, 1)
                ),  # no check of the actual values in the unit tests
            },
        },
    ]

    for test_case in test_cases:

        result = target_nn_model(
            test_case["input"]["X"],
            test_case["input"]["Y"],
            test_case["input"]["n_h"],
            test_case["input"]["num_iterations"],
            test_case["input"]["learning_rate"],
            False,
        )

        try:
            assert result["W1"].shape == test_case["expected"]["W1"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["W1"].shape,
                    "got": result["W1"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the weights matrix W1. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result["b1"].shape == test_case["expected"]["b1"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b1"].shape,
                    "got": result["b1"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the bias vector b1. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result["W2"].shape == test_case["expected"]["W2"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["W2"].shape,
                    "got": result["W2"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the weights matrix W2. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result["b2"].shape == test_case["expected"]["b2"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b2"].shape,
                    "got": result["b2"].shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the bias vector b2. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_predict(target_predict):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "X": np.array([[2, 8, 2, 8], [2, 8, 8, 2]]),
                "parameters": {
                    "W1": np.array(
                        [[2.14274251, -1.93155541], [2.20268789, -2.1131799]]
                    ),
                    "b1": np.array([[-4.83079243], [6.2845223]]),
                    "W2": np.array([[-7.21370685, 7.0898022]]),
                    "b2": np.array([[-3.48755239]]),
                },
            },
            "expected": {
                "predictions": np.array([[True, True, False, False]]),
            },
        },
        {
            "name": "extra_check",
            "input": {
                "X": np.array([[0, 10, 0, 0, 10], [0, 0, 0, 0, 10]]),
                "parameters": {
                    "W1": np.array(
                        [
                            [2.15345603, -2.02993877],
                            [2.24191569, -1.89471923],
                            [1.02971382, -2.24825777],
                        ]
                    ),
                    "b1": np.array([[6.29905582], [-4.80909975], [3.26776186]]),
                    "W2": np.array([[7.07457688, -7.23061969, 1.01318344]]),
                    "b2": np.array([[-3.50971507]]),
                },
            },
            "expected": {
                "predictions": np.array([[True, False, True, True, True]]),
            },
        },
    ]

    for test_case in test_cases:

        result = target_predict(
            test_case["input"]["X"], test_case["input"]["parameters"]
        )

        try:
            assert result.shape == test_case["expected"]["predictions"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["predictions"].shape,
                    "got": result.shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of the output array. Input: X = \n{test_case['input']['X']},\nparameters = {test_case['input']['parameters']}. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result, test_case["expected"]["predictions"], atol=1e-06)
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["predictions"],
                    "got": result,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output array. Input: X = \n{test_case['input']['X']},\nparameters = {test_case['input']['parameters']}. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")
