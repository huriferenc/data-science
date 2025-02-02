# +
import jax.numpy as np
import pandas as pd
from math import isclose
import joblib

# Variables for the default_check test cases.
prices_A = np.array(
    [
        104.0,
        108.0,
        101.0,
        104.0,
        102.0,
        105.0,
        114.0,
        102.0,
        105.0,
        101.0,
        109.0,
        103.0,
        93.0,
        98.0,
        92.0,
        97.0,
        96.0,
        94.0,
        97.0,
        93.0,
        99.0,
        93.0,
        98.0,
        94.0,
        93.0,
        92.0,
        96.0,
        98.0,
        98.0,
        93.0,
        97.0,
        102.0,
        103.0,
        100.0,
        100.0,
        104.0,
        100.0,
        103.0,
        104.0,
        101.0,
        102.0,
        100.0,
        102.0,
        108.0,
        107.0,
        107.0,
        103.0,
        109.0,
        108.0,
        108.0,
    ],
    dtype=np.float32,
)
prices_B = np.array(
    [
        76.0,
        76.0,
        84.0,
        79.0,
        81.0,
        84.0,
        90.0,
        93.0,
        93.0,
        99.0,
        98.0,
        96.0,
        94.0,
        104.0,
        101.0,
        102.0,
        104.0,
        106.0,
        105.0,
        103.0,
        106.0,
        104.0,
        113.0,
        115.0,
        114.0,
        124.0,
        119.0,
        115.0,
        112.0,
        111.0,
        106.0,
        107.0,
        108.0,
        108.0,
        102.0,
        104.0,
        101.0,
        101.0,
        100.0,
        103.0,
        106.0,
        100.0,
        97.0,
        98.0,
        90.0,
        92.0,
        92.0,
        99.0,
        94.0,
        91.0,
    ],
    dtype=np.float32,
)


# -


def test_load_and_convert_data(target_A, target_B):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "expected": {
                "prices_A": prices_A,
                "prices_B": prices_B,
            },
        },
    ]

    for test_case in test_cases:

        try:
            assert type(target_A) == type(test_case["expected"]["prices_A"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": type(test_case["expected"]["prices_A"]),
                    "got": type(target_A),
                }
            )
            print(
                f"prices_A has incorrect type. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert type(target_B) == type(test_case["expected"]["prices_B"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": type(test_case["expected"]["prices_B"]),
                    "got": type(target_B),
                }
            )
            print(
                f"prices_B has incorrect type. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            # Check only one element - no need to check all array.
            assert type(target_A[0].item()) == type(
                test_case["expected"]["prices_A"][0].item()
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": type(test_case["expected"]["prices_A"][0].item()),
                    "got": type(target_A[0].item()),
                }
            )
            print(
                f"Elements of prices_A array have incorrect type. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            # Check only one element - no need to check all array.
            assert type(target_B[0].item()) == type(
                test_case["expected"]["prices_B"][0].item()
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": type(test_case["expected"]["prices_B"][0].item()),
                    "got": type(target_B[0].item()),
                }
            )
            print(
                f"Elements of prices_B array have incorrect type. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert target_A.shape == test_case["expected"]["prices_A"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["prices_A"].shape,
                    "got": target_A.shape,
                }
            )
            print(
                f"Wrong shape of prices_A array. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert target_B.shape == test_case["expected"]["prices_B"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["prices_B"].shape,
                    "got": target_B.shape,
                }
            )
            print(
                f"Wrong shape of prices_B array. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(target_A, test_case["expected"]["prices_A"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["prices_A"],
                    "got": target_A,
                }
            )
            print(
                f"Wrong array prices_A. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(target_B, test_case["expected"]["prices_B"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["prices_B"],
                    "got": target_B,
                }
            )
            print(
                f"Wrong array prices_B. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_f_of_omega(target_f_of_omega):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {"omega": 0, "pA": prices_A, "pB": prices_B},
            "expected": {
                "f_of_omega": prices_B,
            },
        },
        {
            "name": "extra_check_1",
            "input": {"omega": 0.2, "pA": prices_A, "pB": prices_B},
            "expected": {
                "f_of_omega": prices_A * 0.2 + prices_B * (1 - 0.2),
            },
        },
        {
            "name": "extra_check_2",
            "input": {"omega": 0.8, "pA": prices_A, "pB": prices_B},
            "expected": {
                "f_of_omega": prices_A * 0.8 + prices_B * (1 - 0.8),
            },
        },
        {
            "name": "extra_check_3",
            "input": {"omega": 1, "pA": prices_A, "pB": prices_B},
            "expected": {
                "f_of_omega": prices_A,
            },
        },
    ]

    for test_case in test_cases:
        result = target_f_of_omega(
            test_case["input"]["omega"],
            test_case["input"]["pA"],
            test_case["input"]["pB"],
        )

        try:
            assert result.shape == test_case["expected"]["f_of_omega"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["f_of_omega"].shape,
                    "got": result.shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of f_of_omega output for omega = {test_case['input']['omega']}. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result, test_case["expected"]["f_of_omega"], atol=1e-5)
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["f_of_omega"],
                    "got": result,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of f_of_omega for omega = {test_case['input']['omega']}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_L_of_omega_array(target_L_of_omega_array):
    successful_cases = 0
    failed_cases = []

    # Not all of the values of the output array will be checked - only some of them.
    # In graders all of the output array gets checked.
    test_cases = [
        {
            "name": "default_check",
            "input": {
                "omega_array": np.linspace(0, 1, 1001, endpoint=True),
                "pA": prices_A,
                "pB": prices_B,
            },
            "expected": {
                "shape": (1001,),
                "L_of_omega_array": [
                    {
                        "i": 0,
                        "L_of_omega": 110.72,
                    },
                    {
                        "i": 1000,
                        "L_of_omega": 27.48,
                    },
                    {
                        "i": 400,
                        "L_of_omega": 28.051199,
                    },
                ],
            },
        },
        {
            "name": "extra_check",
            "input": {
                "omega_array": np.linspace(0, 1, 11, endpoint=True),
                "pA": prices_A,
                "pB": prices_B,
            },
            "expected": {
                "shape": (11,),
                "L_of_omega_array": [
                    {
                        "i": 0,
                        "L_of_omega": 110.72,
                    },
                    {
                        "i": 11,
                        "L_of_omega": 27.48,
                    },
                    {
                        "i": 5,
                        "L_of_omega": 17.67,
                    },
                ],
            },
        },
    ]

    for test_case in test_cases:
        result = target_L_of_omega_array(
            test_case["input"]["omega_array"],
            test_case["input"]["pA"],
            test_case["input"]["pB"],
        )

        try:
            assert result.shape == test_case["expected"]["shape"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["shape"],
                    "got": result.shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of L_of_omega_array output. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        for test_case_i in test_case["expected"]["L_of_omega_array"]:
            i = test_case_i["i"]

            try:
                assert isclose(result[i], test_case_i["L_of_omega"], abs_tol=1e-5)
                successful_cases += 1

            except:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": test_case_i["L_of_omega"],
                        "got": result[i],
                    }
                )
                print(
                    f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of L_of_omega_array for omega_array = \n{test_case['input']['omega_array']}\nTest for index i = {i}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
                )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_dLdOmega_of_omega_array(target_dLdOmega_of_omega_array):
    successful_cases = 0
    failed_cases = []

    # Not all of the values of the output array will be checked - only some of them.
    # In graders all of the output array gets checked.
    test_cases = [
        {
            "name": "default_check",
            "input": {
                "omega_array": np.linspace(0, 1, 1001, endpoint=True),
                "pA": prices_A,
                "pB": prices_B,
            },
            "expected": {
                "shape": (1001,),
                "dLdOmega_of_omega_array": [
                    {
                        "i": 0,
                        "dLdOmega_of_omega": -288.96,
                    },
                    {
                        "i": 1000,
                        "dLdOmega_of_omega": 122.47999,
                    },
                    {
                        "i": 400,
                        "dLdOmega_of_omega": -124.38398,
                    },
                ],
            },
        },
        {
            "name": "extra_check",
            "input": {
                "omega_array": np.linspace(0, 1, 11, endpoint=True),
                "pA": prices_A,
                "pB": prices_B,
            },
            "expected": {
                "shape": (11,),
                "dLdOmega_of_omega_array": [
                    {
                        "i": 0,
                        "dLdOmega_of_omega": -288.96,
                    },
                    {
                        "i": 11,
                        "dLdOmega_of_omega": 122.47999,
                    },
                    {
                        "i": 5,
                        "dLdOmega_of_omega": -83.240036,
                    },
                ],
            },
        },
    ]

    for test_case in test_cases:
        result = target_dLdOmega_of_omega_array(
            test_case["input"]["omega_array"],
            test_case["input"]["pA"],
            test_case["input"]["pB"],
        )

        try:
            assert result.shape == test_case["expected"]["shape"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["shape"],
                    "got": result.shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of dLdOmega_of_omega_array output. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        for test_case_i in test_case["expected"]["dLdOmega_of_omega_array"]:
            i = test_case_i["i"]

            try:
                assert isclose(
                    result[i], test_case_i["dLdOmega_of_omega"], abs_tol=1e-2
                )
                successful_cases += 1

            except:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": test_case_i["dLdOmega_of_omega"],
                        "got": result[i],
                    }
                )
                print(
                    f"Test case \"{failed_cases[-1].get('name')}\". Wrong output of dLdOmega_of_omega_array for omega_array = \n{test_case['input']['omega_array']}\nTest for index i = {i}. \n\tExpected: \n{failed_cases[-1].get('expected')}\n\tGot: \n{failed_cases[-1].get('got')}"
                )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_get_word_frequency(target):
    successful_cases = 0
    failed_cases = []

    test_cases = joblib.load("test_cases_get_word_frequency.joblib")

    for test_case in test_cases:

        output = target(**test_case["input"])
        expected = test_case["expected"]

        ## Check same length
        if len(output) != len(expected):
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": len(expected),
                    "got": len(output),
                }
            )
            print(
                f"Wrong output dictionary size for {failed_cases[-1]['name']}. Expected: {failed_cases[-1]['expected']}. Got: {failed_cases[-1]['got']}"
            )
            continue

        ## Check if keys are the same
        keys_truth = set(expected.keys())
        keys_output = set(output.keys())
        if not (keys_truth.issubset(keys_output) and keys_output.issubset(keys_truth)):
            failed_cases.append(
                {"name": test_case["name"], "expected": keys_truth, "got": keys_output}
            )
            print(
                f"Wrong output dictionary keys for {failed_cases[-1]['name']}. Expected: {failed_cases[-1]['expected']}. Got: {failed_cases[-1]['got']}"
            )
            continue

        ## Check if, for every key, the counting is the same
        for key in output.keys():
            if len(output[key]) != len(expected[key]):
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": len(expected[key]),
                        "got": len(output[key]),
                    }
                )
                print(
                    f"Wrong output dictionary size for word {key} in {failed_cases[-1]['name']}. Expected: {failed_cases[-1]['expected']}. Got: {failed_cases[-1]['got']}"
                )
                break

            if output[key] != expected[key]:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": expected[key],
                        "got": output[key],
                    }
                )
                print(
                    f"Wrong counting for word {key} in {failed_cases[-1]['name']}. Expected: {failed_cases[-1]['expected']}. Got: {failed_cases[-1]['got']}"
                )
                break
        successful_cases += 1

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_prob_word_given_class(target, word_frequency, class_frequency):

    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "test_case_0",
            "input": {"word": "compile", "cls": "ham"},
            "expected": 0.0025951557093425604,
        },
        {
            "name": "test_case_1",
            "input": {"word": "doc", "cls": "spam"},
            "expected": 0.0026929982046678637,
        },
        {
            "name": "test_case_2",
            "input": {"word": "nights", "cls": "ham"},
            "expected": 0.003748558246828143,
        },
        {
            "name": "test_case_3",
            "input": {"word": "attached", "cls": "spam"},
            "expected": 0.008976660682226212,
        },
        {
            "name": "test_case_4",
            "input": {"word": "hook", "cls": "ham"},
            "expected": 0.0025951557093425604,
        },
        {
            "name": "test_case_5",
            "input": {"word": "projector", "cls": "spam"},
            "expected": 0.0017953321364452424,
        },
        {
            "name": "test_case_6",
            "input": {"word": "also", "cls": "ham"},
            "expected": 0.2577854671280277,
        },
        {
            "name": "test_case_7",
            "input": {"word": "requirements", "cls": "spam"},
            "expected": 0.018850987432675045,
        },
        {
            "name": "test_case_8",
            "input": {"word": "dietary", "cls": "ham"},
            "expected": 0.0008650519031141869,
        },
        {
            "name": "test_case_9",
            "input": {"word": "equipment", "cls": "spam"},
            "expected": 0.02064631956912029,
        },
        {
            "name": "test_case_10",
            "input": {"word": "staying", "cls": "ham"},
            "expected": 0.008362168396770472,
        },
        {
            "name": "test_case_11",
            "input": {"word": "find", "cls": "spam"},
            "expected": 0.09425493716337523,
        },
        {
            "name": "test_case_12",
            "input": {"word": "reserve", "cls": "ham"},
            "expected": 0.019319492502883506,
        },
        {
            "name": "test_case_13",
            "input": {"word": "several", "cls": "spam"},
            "expected": 0.04039497307001795,
        },
        {
            "name": "test_case_14",
            "input": {"word": "university", "cls": "ham"},
            "expected": 0.13408304498269896,
        },
        {
            "name": "test_case_15",
            "input": {"word": "shirley", "cls": "spam"},
            "expected": 0.0017953321364452424,
        },
        {
            "name": "test_case_16",
            "input": {"word": "ca", "cls": "ham"},
            "expected": 0.03460207612456748,
        },
        {
            "name": "test_case_17",
            "input": {"word": "enron", "cls": "spam"},
            "expected": 0.0008976660682226212,
        },
        {
            "name": "test_case_18",
            "input": {"word": "thanks", "cls": "ham"},
            "expected": 0.41205305651672436,
        },
        {
            "name": "test_case_19",
            "input": {"word": "soon", "cls": "spam"},
            "expected": 0.04039497307001795,
        },
    ]

    for test_case in test_cases:

        output = target(
            word_frequency=word_frequency,
            class_frequency=class_frequency,
            **test_case["input"],
        )
        expected = test_case["expected"]

        if not np.isclose(output, expected):
            failed_cases.append(
                {"name": test_case["name"], "expected": expected, "got": output}
            )
            print(
                f"Wrong value for P({test_case['input']['word']} | spam = {test_case['input']['spam']}) in {failed_cases[-1]['name']}. Expected: {failed_cases[-1]['expected']}. Got: {failed_cases[-1]['got']}"
            )
            continue
        successful_cases += 1

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_prob_email_given_class(target, word_frequency, class_frequency):

    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "test_case_0",
            "input": {
                "treated_email": [
                    "ca",
                    "enron",
                    "find",
                    "projector",
                    "compile",
                    "find",
                    "attached",
                    "staying",
                    "soon",
                ],
                "cls": "ham",
            },
            "expected": 3.3983894210489835e-13,
        },
        {
            "name": "test_case_1",
            "input": {
                "treated_email": [
                    "doc",
                    "soon",
                    "university",
                    "nights",
                    "attached",
                    "nights",
                    "equipment",
                    "hook",
                ],
                "cls": "spam",
            },
            "expected": 7.069258091965318e-19,
        },
        {
            "name": "test_case_2",
            "input": {
                "treated_email": [
                    "thanks",
                    "ca",
                    "university",
                    "enron",
                    "university",
                    "several",
                ],
                "cls": "ham",
            },
            "expected": 9.77525599231039e-06,
        },
        {
            "name": "test_case_3",
            "input": {
                "treated_email": ["projector", "also", "also", "ca", "hook"],
                "cls": "spam",
            },
            "expected": 6.013747036672614e-10,
        },
        {
            "name": "test_case_4",
            "input": {
                "treated_email": [
                    "dietary",
                    "find",
                    "thanks",
                    "staying",
                    "shirley",
                    "dietary",
                    "attached",
                    "thanks",
                ],
                "cls": "ham",
            },
            "expected": 3.4470299679032404e-12,
        },
        {
            "name": "test_case_5",
            "input": {
                "treated_email": [
                    "several",
                    "find",
                    "staying",
                    "staying",
                    "also",
                    "ca",
                    "university",
                    "equipment",
                ],
                "cls": "spam",
            },
            "expected": 4.397549817075224e-15,
        },
        {
            "name": "test_case_6",
            "input": {
                "treated_email": [
                    "projector",
                    "reserve",
                    "attached",
                    "staying",
                    "university",
                    "hook",
                    "staying",
                    "dietary",
                ],
                "cls": "ham",
            },
            "expected": 2.717031873714673e-16,
        },
        {
            "name": "test_case_7",
            "input": {
                "treated_email": [
                    "thanks",
                    "attached",
                    "thanks",
                    "equipment",
                    "also",
                    "staying",
                    "several",
                    "staying",
                ],
                "cls": "spam",
            },
            "expected": 4.973982358381553e-15,
        },
        {
            "name": "test_case_8",
            "input": {
                "treated_email": [
                    "compile",
                    "dietary",
                    "requirements",
                    "shirley",
                    "several",
                    "nights",
                    "doc",
                    "hook",
                    "thanks",
                ],
                "cls": "ham",
            },
            "expected": 2.698275888443421e-16,
        },
        {
            "name": "test_case_9",
            "input": {
                "treated_email": [
                    "enron",
                    "hook",
                    "staying",
                    "staying",
                    "doc",
                    "equipment",
                ],
                "cls": "spam",
            },
            "expected": 1.444102224707258e-16,
        },
    ]

    for test_case in test_cases:

        got = target(
            word_frequency=word_frequency,
            class_frequency=class_frequency,
            **test_case["input"],
        )
        expected = test_case["expected"]

        if not np.isclose(got, expected):
            failed_cases.append(
                {"name": test_case["name"], "expected": expected, "got": got}
            )
            print(
                f"Wrong value for email = {test_case['input']['treated_email']} and spam = {test_case['input']['spam']} in {failed_cases[-1]['name']}. Expected: {failed_cases[-1]['expected']}. Got: {failed_cases[-1]['got']}"
            )
            continue
        successful_cases += 1

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_naive_bayes(target, word_frequency, class_frequency):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "test_case_0",
            "input": {
                "treated_email": [
                    "ca",
                    "enron",
                    "find",
                    "projector",
                    "compile",
                    "find",
                    "attached",
                    "staying",
                    "soon",
                ]
            },
            "expected": 0,
        },
        {
            "name": "test_case_1",
            "input": {
                "treated_email": [
                    "doc",
                    "soon",
                    "university",
                    "nights",
                    "attached",
                    "nights",
                    "equipment",
                    "hook",
                ]
            },
            "expected": 0,
        },
        {
            "name": "test_case_2",
            "input": {
                "treated_email": [
                    "thanks",
                    "ca",
                    "university",
                    "enron",
                    "university",
                    "several",
                ]
            },
            "expected": 0,
        },
        {
            "name": "test_case_3",
            "input": {"treated_email": ["projector", "also", "also", "ca", "hook"]},
            "expected": 0,
        },
        {
            "name": "test_case_4",
            "input": {
                "treated_email": [
                    "dietary",
                    "find",
                    "thanks",
                    "staying",
                    "shirley",
                    "dietary",
                    "attached",
                    "thanks",
                ]
            },
            "expected": 0,
        },
        {
            "name": "test_case_5",
            "input": {
                "treated_email": [
                    "several",
                    "find",
                    "staying",
                    "staying",
                    "also",
                    "ca",
                    "university",
                    "equipment",
                ]
            },
            "expected": 0,
        },
        {
            "name": "test_case_6",
            "input": {
                "treated_email": [
                    "projector",
                    "reserve",
                    "attached",
                    "staying",
                    "university",
                    "hook",
                    "staying",
                    "dietary",
                ]
            },
            "expected": 0,
        },
        {
            "name": "test_case_7",
            "input": {
                "treated_email": [
                    "thanks",
                    "attached",
                    "thanks",
                    "equipment",
                    "also",
                    "staying",
                    "several",
                    "staying",
                ]
            },
            "expected": 0,
        },
        {
            "name": "test_case_8",
            "input": {
                "treated_email": [
                    "compile",
                    "dietary",
                    "requirements",
                    "shirley",
                    "several",
                    "nights",
                    "doc",
                    "hook",
                    "thanks",
                ]
            },
            "expected": 0,
        },
        {
            "name": "test_case_9",
            "input": {
                "treated_email": [
                    "enron",
                    "hook",
                    "staying",
                    "staying",
                    "doc",
                    "equipment",
                ]
            },
            "expected": 0,
        },
    ]

    for test_case in test_cases:

        got = target(
            word_frequency=word_frequency,
            class_frequency=class_frequency,
            **test_case["input"],
        )
        expected = test_case["expected"]

        if not np.isclose(got, expected):
            failed_cases.append(
                {"name": test_case["name"], "expected": expected, "got": got}
            )
            print(
                f"Wrong decision for email = {test_case['input']['treated_email']} in {failed_cases[-1]['name']}. Expected: {failed_cases[-1]['expected']}. Got: {failed_cases[-1]['got']}"
            )
            continue
        successful_cases += 1

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")
