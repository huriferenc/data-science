import numpy as np


def test_matrix(target_P, target_X0, target_X1):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "expected": {
                "P": np.array(
                    [
                        [0, 0.75, 0.35, 0.25, 0.85],
                        [0.15, 0, 0.35, 0.25, 0.05],
                        [0.15, 0.15, 0, 0.25, 0.05],
                        [0.15, 0.05, 0.05, 0, 0.05],
                        [0.55, 0.05, 0.25, 0.25, 0],
                    ]
                ),
                "X0": np.array([[0], [0], [0], [1], [0]]),
            },
        },
    ]

    for test_case in test_cases:

        try:
            assert target_P.shape == test_case["expected"]["P"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["P"].shape,
                    "got": target_P.shape,
                }
            )
            print(
                f"Wrong shape of matrix P. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(
                np.diagonal(target_P), np.diagonal(test_case["expected"]["P"])
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": np.nonzero(
                        np.logical_not(
                            np.isclose(
                                np.diagonal(target_P),
                                np.diagonal(test_case["expected"]["P"]),
                            )
                        )
                    ),
                    "got": sum(target_P),
                }
            )
            print(f"Wrong matrix P. \nCheck the diagonal elements.")

        try:
            assert np.allclose(sum(target_P), sum(test_case["expected"]["P"]))
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": np.nonzero(
                        np.logical_not(
                            np.isclose(sum(target_P), sum(test_case["expected"]["P"]))
                        )
                    ),
                    "got": sum(target_P),
                }
            )
            print(
                f"Wrong matrix P. \nCheck the elements in the column {failed_cases[-1].get('expected')[0][0] + 1}."
            )

        try:
            assert target_X0.shape == test_case["expected"]["X0"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["X0"].shape,
                    "got": target_X0.shape,
                }
            )
            print(
                f"Wrong shape of vector X0. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(target_X0, test_case["expected"]["X0"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": np.nonzero(
                        np.logical_not(
                            np.isclose(target_X0, test_case["expected"]["X0"])
                        )
                    ),
                    "got": target_X0,
                }
            )
            print(
                f"Wrong array X0.\nCheck element {failed_cases[-1].get('expected')[0][0] + 1} in the vector X0."
            )

        expected_X1 = np.matmul(target_P, target_X0)

        try:
            assert target_X1.shape == expected_X1.shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_X1.shape,
                    "got": target_X1.shape,
                }
            )
            print(
                f"Wrong shape of vector X1. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(target_X1, expected_X1)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_X1,
                    "got": target_X1,
                }
            )
            print(
                f"Wrong vector X1. Check if matrix multiplication was performed correctly."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_check_eigenvector(target_T):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "P": np.array(
                    [
                        [0, 0.75, 0.35, 0.25, 0.85],
                        [0.15, 0, 0.35, 0.25, 0.05],
                        [0.15, 0.15, 0, 0.25, 0.05],
                        [0.15, 0.05, 0.05, 0, 0.05],
                        [0.55, 0.05, 0.25, 0.25, 0],
                    ]
                ),
            },
        },
        {
            "name": "extra_check",
            "input": {
                "P": np.array(
                    [
                        [2, 3],
                        [2, 1],
                    ]
                ),
            },
        },
    ]

    for test_case in test_cases:

        X_inf = np.linalg.eig(test_case["input"]["P"])[1][:, 0]

        target_X_check = target_T(test_case["input"]["P"], X_inf)
        expected_X_check = test_case["input"]["P"] @ X_inf

        try:
            assert target_X_check.shape == expected_X_check.shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_X_check.shape,
                    "got": target_X_check.shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of output matrix in the check_eigenvector function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(target_X_check, expected_X_check)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_X_check,
                    "got": target_X_check,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output matrix in the check_eigenvector function. Check if matrix multiplication was performed correctly."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_center_data(target_center_data):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": np.load("./support_files/imgs_flatten.npy"),
            "expected": np.load("./support_files/expected_centered_data.npy"),
        },
    ]

    for test_case in test_cases:
        try:
            target_result = target_center_data(test_case["input"])

        except Exception as e:
            print(f"There was an error evaluating the function. \nError: {e}")
            return

        try:
            assert isinstance(target_result, np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": np.ndarray,
                    "got": type(target_result),
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output type on the center_data function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(target_result, test_case["expected"], atol=1e-2)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": target_result,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output on the center_data function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_cov_matrix(target_cov_mat):
    successful_cases = 0
    failed_cases = []
    test_cases = [
        {
            "name": "default_check",
            "input": np.load("./support_files/expected_centered_data.npy"),
            "expected": np.load("./support_files/expected_cov_mat.npy"),
        },
    ]

    for test_case in test_cases:
        try:
            target_result = target_cov_mat(test_case["input"])
        except Exception as e:
            print(f"There was an error evaluating the function. \nError: {e}")
            return

        try:
            assert isinstance(target_result, np.ndarray)
            successful_cases += 1
        except:

            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": np.ndarray,
                    "got": type(target_result),
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output type on the get_cov_matrix function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert target_result.shape == test_case["expected"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"].shape,
                    "got": target_result.shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output shape on the get_cov_matrix function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(target_result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": target_result,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output on the get_cov_matrix function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_check_PCA(target_pca):
    successful_cases = 0
    failed_cases = []
    test_cases = [
        {
            "name": "default_check_2",
            "input": {
                "X": np.load("./support_files/expected_centered_data.npy"),
                "eigenvecs": np.load("./support_files/expected_eigvecs.npy"),
                "k": 2,
            },
            "expected": np.load("./support_files/expected_pca2.npy"),
        },
        {
            "name": "default_check_12",
            "input": {
                "X": np.load("./support_files/expected_centered_data.npy"),
                "eigenvecs": np.load("./support_files/expected_eigvecs.npy"),
                "k": 12,
            },
            "expected": np.load("./support_files/expected_pca12.npy"),
        },
    ]

    for test_case in test_cases:
        try:
            target_result = target_pca(**test_case["input"])
        except Exception as e:
            print(f"There was an error evaluating the function. \nError: {e}")
            return

        try:
            assert isinstance(target_result, np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": np.ndarray,
                    "got": type(target_result),
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output type on the perform_PCA function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert target_result.shape == test_case["expected"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"].shape,
                    "got": target_result.shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output shape on the perform_PCA function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(target_result, test_case["expected"], atol=1e-8)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": target_result,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output on the perform_PCA function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


np.random.seed(179)

np.random.uniform(low=10, high=20, size=10)


def test_get_stats(target):

    test_cases = [
        {
            "name": "test_case_1",
            "input": np.array(
                [
                    19.50455046,
                    12.90527347,
                    18.68267129,
                    17.85301249,
                    13.71774122,
                    13.91098702,
                    18.16410518,
                    16.63597096,
                    18.57756755,
                    17.41476787,
                ]
            ),
            "expected": (10, 16.73666475089095, 2.365698170445931),
        },
        {
            "name": "test_case_2",
            "input": np.array(
                [
                    49.99704465,
                    49.26508329,
                    48.15882412,
                    51.59740797,
                    49.05896027,
                    46.01397403,
                    45.29696906,
                    48.63119845,
                    45.52315415,
                    43.58113295,
                    43.58810372,
                    48.93753677,
                    51.91145255,
                    52.23945901,
                    42.47087567,
                    42.50170719,
                    42.44115578,
                    46.40779799,
                    40.57215065,
                    48.43968513,
                ]
            ),
            "expected": (20, 46.83168366960498, 3.501190844170854),
        },
        {
            "name": "test_case_3",
            "input": np.array(
                [
                    1.70923536,
                    0.39026122,
                    4.04785046,
                    2.34032621,
                    2.88279857,
                    0.37808171,
                    3.46175874,
                    24.92570901,
                    0.27035019,
                    2.35627335,
                    5.66096993,
                    3.00928774,
                    5.69682927,
                    2.86099122,
                    16.65419663,
                    3.65581194,
                    1.56435499,
                    30.57348257,
                    5.08099823,
                    2.71325506,
                    0.7967606,
                    5.55042057,
                    3.37970955,
                    4.70478162,
                    0.55352832,
                    12.62701788,
                    0.68942262,
                    3.01225193,
                    2.7348847,
                    7.91641966,
                ]
            ),
            "expected": (30, 5.406600662367174, 7.053067681261619),
        },
    ]

    successful_cases = 0
    failed_cases = []

    for test_case in test_cases:

        try:
            output = target(test_case["input"])
            check_array = [
                np.isclose(x, y) for x, y in zip(output, test_case["expected"])
            ]
        except Exception as e:
            print(
                f"\033[91mUnit test broken in {test_case['name']} due to an Exception being thrown. Input was:\n\t{test_case['input']}\nException is: {e}.\nAborting. "
            )
            return

        # Check if outputs are numbers
        for value in output:
            if not isinstance(value, (int, float)):
                print(
                    f"\033[91mIncorrect data type for {'n' if output.index(value) == 0 else 'x' if output.index(value) == 1 else 's'}. Expected output is a float or integer, but got {type(value)}.\nAborting test."
                )
                return

        # Check for n
        if not check_array[0]:
            failed_cases.append(
                {
                    "name": f"n value for {test_case['name']}",
                    "expected": test_case["expected"][0],
                    "got": output[0],
                }
            )
            print(
                f"Wrong n value for test case {failed_cases[-1]['name']}. The input was:\n{test_case['input']}.\nExpected n = {failed_cases[-1]['expected']}, but got n = {failed_cases[-1]['got']}."
            )
        else:
            successful_cases += 1
        # Check for x
        if not check_array[1]:
            failed_cases.append(
                {
                    "name": f"x value for {test_case['name']}",
                    "expected": test_case["expected"][1],
                    "got": output[1],
                }
            )
            print(
                f"Wrong x value for test case {failed_cases[-1]['name']}. The input was:\n{test_case['input']}.\nExpected x = {failed_cases[-1]['expected']}, but got x = {failed_cases[-1]['got']}."
            )
        else:
            successful_cases += 1
        # Check for s
        if not check_array[2]:
            failed_cases.append(
                {
                    "name": f"s value for {test_case['name']}",
                    "expected": test_case["expected"][2],
                    "got": output[2],
                }
            )
            print(
                f"Wrong s value for test case {failed_cases[-1]['name']}. The input was:\n{test_case['input']}.\nExpected s = {failed_cases[-1]['expected']}, but got s = {failed_cases[-1]['got']}. Do not forget to pass the argument ddof with the proper value!"
            )
        else:
            successful_cases += 1

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_degrees_of_freedom(target):

    test_cases = [
        {
            "name": "test_case_1",
            "input": {"n_v": 5906, "s_v": 46.13, "n_c": 4442, "s_c": 44.45},
            "expected": 9742.194110899103,
        },
        {
            "name": "test_case_2",
            "input": {"n_v": 5261, "s_v": 36.39, "n_c": 2537, "s_c": 54.62},
            "expected": 3657.064062203471,
        },
        {
            "name": "test_case_3",
            "input": {"n_v": 5804, "s_v": 40.89, "n_c": 3098, "s_c": 39.12},
            "expected": 6569.922250340089,
        },
        {
            "name": "test_case_4",
            "input": {"n_v": 2029, "s_v": 42.62, "n_c": 4081, "s_c": 41.52},
            "expected": 3955.588594226483,
        },
    ]

    successful_cases = 0
    failed_cases = []

    for test_case in test_cases:

        try:
            output = target(**test_case["input"])
            check_solution = np.isclose(output, test_case["expected"])
        except Exception as e:
            print(
                f"\033[91mUnit test broken in {test_case['name']} due to an Exception being thrown. Input was:\n\t{test_case['input']}\nException is: {e}.\nAborting. "
            )
            return

        if not check_solution:
            failed_cases.append(
                {
                    "name": f"n value for {test_case['name']}",
                    "expected": test_case["expected"],
                    "got": output,
                }
            )
            print(
                f"Wrong value for test case {failed_cases[-1]['name']}. The input values were:\n{test_case['input']}.\nExpected {failed_cases[-1]['expected']}, but got {failed_cases[-1]['got']}."
            )
        else:
            successful_cases += 1

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_t_value(target):

    test_cases = [
        {
            "name": "test_case_1",
            "input": {
                "n_v": 5776,
                "x_v": 23.38294493263588,
                "s_v": 38.92,
                "n_c": 2139,
                "s_c": 42.88,
                "x_c": 41.890689354074816,
            },
            "expected": -17.473698745465104,
        },
        {
            "name": "test_case_2",
            "input": {
                "n_v": 4187,
                "x_v": 26.131926712301706,
                "s_v": 44.35,
                "n_c": 4566,
                "s_c": 47.56,
                "x_c": 35.456317192552234,
            },
            "expected": -9.49119497926876,
        },
        {
            "name": "test_case_3",
            "input": {
                "n_v": 5866,
                "x_v": 39.14918250716274,
                "s_v": 41.06,
                "n_c": 2419,
                "s_c": 38.36,
                "x_c": 41.01944898537004,
            },
            "expected": -1.9761484013170285,
        },
        {
            "name": "test_case_4",
            "input": {
                "n_v": 4600,
                "x_v": 30.641116701676623,
                "s_v": 40.83,
                "n_c": 5542,
                "s_c": 49.07,
                "x_c": 15.063310320827487,
            },
            "expected": 17.450508434704457,
        },
    ]

    successful_cases = 0
    failed_cases = []

    for test_case in test_cases:

        try:
            output = target(**test_case["input"])
            check_solution = np.isclose(output, test_case["expected"])
        except Exception as e:
            print(
                f"\033[91mUnit test broken in {test_case['name']} due to an Exception being thrown. Input was:\n\t{test_case['input']}\nException is: {e}.\nAborting. "
            )
            return

        if not check_solution:
            failed_cases.append(
                {
                    "name": f"n value for {test_case['name']}",
                    "expected": test_case["expected"],
                    "got": output,
                }
            )
            print(
                f"Wrong value for test case {failed_cases[-1]['name']}. The input values were:\n{test_case['input']}.\nExpected {failed_cases[-1]['expected']}, but got {failed_cases[-1]['got']}."
            )
        else:
            successful_cases += 1

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_p_value(target):

    test_cases = [
        {
            "name": "test_case_1",
            "input": {"d": 3546, "t_value": 1.121289949626803},
            "expected": 0.13112019466699698,
        },
        {
            "name": "test_case_2",
            "input": {"d": 3469, "t_value": 2.694950876114083},
            "expected": 0.003536921187750286,
        },
        {
            "name": "test_case_3",
            "input": {"d": 3550, "t_value": 1.1728775488141598},
            "expected": 0.12046180346625257,
        },
        {
            "name": "test_case_4",
            "input": {"d": 2681, "t_value": 1.5534967853750998},
            "expected": 0.060211263665166714,
        },
    ]

    successful_cases = 0
    failed_cases = []

    for test_case in test_cases:

        try:
            output = target(**test_case["input"])
            check_solution = np.isclose(output, test_case["expected"])
        except Exception as e:
            print(
                f"\033[91mUnit test broken in {test_case['name']} due to an Exception being thrown. Input was:\n\t{test_case['input']}\nException is: {e}.\nAborting. "
            )
            return

        if not check_solution:
            failed_cases.append(
                {
                    "name": f"n value for {test_case['name']}",
                    "expected": test_case["expected"],
                    "got": output,
                }
            )
            print(
                f"Wrong value for test case {failed_cases[-1]['name']}. The input values were:\n{test_case['input']}.\nExpected {failed_cases[-1]['expected']}, but got {failed_cases[-1]['got']}."
            )
        else:
            successful_cases += 1

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_make_decision(target):

    test_cases = [
        {
            "name": "test_case_1",
            "input": {
                "X_v": np.array([34, 27, 32, 23, 31, 26, 29, 37, 37, 23]),
                "X_c": np.array([24, 23, 34, 35, 23, 30, 21, 20, 29, 21]),
                "alpha": 0.09,
            },
            "expected": "Reject H_0",
        },
        {
            "name": "test_case_2",
            "input": {
                "X_v": np.array([38, 27, 36, 28, 26, 33, 34, 30, 31, 27]),
                "X_c": np.array([27, 28, 26, 28, 31, 36, 20, 21, 39, 22]),
                "alpha": 0.01,
            },
            "expected": "Do not reject H_0",
        },
        {
            "name": "test_case_3",
            "input": {
                "X_v": np.array([32, 37, 30, 28, 32, 34, 26, 34, 31, 32]),
                "X_c": np.array([34, 34, 28, 21, 23, 25, 30, 31, 32, 34]),
                "alpha": 0.01,
            },
            "expected": "Do not reject H_0",
        },
        {
            "name": "test_case_4",
            "input": {
                "X_v": np.array([29, 35, 32, 20, 32, 30, 21, 23, 26, 32]),
                "X_c": np.array([25, 24, 28, 23, 28, 24, 33, 31, 39, 29]),
                "alpha": 0.08,
            },
            "expected": "Do not reject H_0",
        },
    ]

    successful_cases = 0
    failed_cases = []

    for test_case in test_cases:

        try:
            output = target(**test_case["input"])
            check_solution = output == test_case["expected"]
        except Exception as e:
            print(
                f"\033[91mUnit test broken in {test_case['name']} due to an Exception being thrown. Input was:\n\t{test_case['input']}\nException is: {e}.\nAborting. "
            )
            return

        if not check_solution:
            failed_cases.append(
                {
                    "name": f"n value for {test_case['name']}",
                    "expected": test_case["expected"],
                    "got": output,
                }
            )
            print(
                f"Wrong value for test case {failed_cases[-1]['name']}. The input values were:\n{test_case['input']}.\nExpected {failed_cases[-1]['expected']}, but got {failed_cases[-1]['got']}."
            )
        else:
            successful_cases += 1

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")
