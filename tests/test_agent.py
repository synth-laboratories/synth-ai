import unittest
from typing import List, Union

from pydantic import BaseModel

from zyk.lms.core.main import LM


class UnitTestDict(BaseModel):
    test_description: str
    input_names: List[str]
    input_types: List[str]
    stringified_input_values: List[str]
    assertion_condition: str
    assertion_type: str  # Consider using Literal for specific assertion types


class ActionArgument(BaseModel):
    key: str
    value: Union[str, int, float, bool, UnitTestDict]


class ReAct(BaseModel):
    reasoning: str
    action_name: str
    action_args: List[ActionArgument]  # Dict#[str, Dict]


system = """
<System Message> 
# Premise
You are a software engineer
Here is some information about this setting
<Setting Information>
You are working to solve a computer science problem. You will need to submit a solution to the problem, which will be tested against a suite of hidden unit tests.
</Setting Information>
<Actions Available>
<edit_submission>
<action_context>
Edit the submission code. Use this when you want to make changes to the current solution.
</action_context>
<action_arg_spec>
{'first_line': <class 'int'>, 'last_line': <class 'int'>, 'new_code': <class 'str'>}
</action_arg_spec>
<action_description>
Edit the submission code
</action_description>

</edit_submission>
<add_submission>
<action_context>
Add the submission code. Use this when you want to start from scratch with a new solution.
</action_context>
<action_arg_spec>
{'submission': <class 'str'>}
</action_arg_spec>
<action_description>
Add the submission code
</action_description>

</add_submission>
<add_unit_test>
<action_context>
Add a unit test. The unit test information you submit must be in the format of a BCBUnitTest: 

class BCBUnitTest(BaseModel):
    test_description: str
    input_names: List[str]
    input_types: List[str]
    input_values: List[Any]
    assertion_condition: str
    assertion_type: Literal["assertTrue", "assertRaises"] = "assertTrue"


 It will be parsed via BCBUnitTest(**unit_test_dict)



# Some various notes:
1. If an input should be of a type defined by a specific package, add the package name/alias to the type. E.g. "np.ndarray" or "pd.DataFrame". You still should fully define the value for the input_value field e.g. "pd.DataFrame({'a': [1, 2, 3]})"

2. Unit tests will be compiled from the BCBUnitTest class as follows:
    A. For AssertTrue type tests, the test will be compiled as follows:
    ```python
    def test_case(self):
        # {{self.test_description}}

        {{defs}}
        result = {{function_name}}(**{{{{args}}}}})
        self.{{self.assertion_type}}({{self.assertion_condition}})
    ```
    B. For AssertRaises type tests, the test will be compiled as follows:

    ```python
    def test_case(self):
        # {{self.test_description}}
        {{defs}}
        with self.{{self.assertion_type}}({{self.assertion_condition}}):
            {{function_name}}(**{{{{args}}}}})
    ```

    Provide information accordingly.

</action_context>
<action_arg_spec>
{'unit_test_name': <class 'str'>, 'unit_test_dict': typing.Dict}
</action_arg_spec>
<action_description>
Add a unit test
</action_description>

</add_unit_test>
<remove_unit_test>
<action_context>
Remove a unit test
</action_context>
<action_arg_spec>
{'unit_test_name': <class 'str'>}
</action_arg_spec>
<action_description>
Remove a unit test
</action_description>

</remove_unit_test>
<test_submission>
<action_context>
Test the submission
</action_context>
<action_arg_spec>
{}
</action_arg_spec>
<action_description>
Test the submission
</action_description>

</test_submission>
<submit_solution>
<action_context>
Submit the solution
</action_context>
<action_arg_spec>
{}
</action_arg_spec>
<action_description>
Submit the solution
</action_description>

</submit_solution>

</Actions Available>
You'll be given your past actions/thoughts, along with recent raw observations from the environment
The environment one step in the past is your current environment.

# Objective
Please complete the problem by drafting a solution, creating unit tests, improving the solution, and submitting the solution.

# Constraints
You will be given a code_prompt_for_answer, which contains imports and the function signature. Your solution must comprise code that can be appended to code_prompt_for_answer and run as a single script.
 
"""

user = """
<User Message> 
# Recent Actions / Thoughts

# Recent Observations
<1 environment step(s) in the past>{'action_result': None, 'environment_state': {'question': 'import pandas as pd\nimport numpy as np\n\n# Constants\nCOLUMNS = [\'column1\', \'column2\', \'column3\', \'column4\', \'column5\']\n\ndef task_func(df, dct):\n    '''\n    Replace certain values in a DataFrame with a dictionary mapping and calculate the Pearson correlation coefficient between each pair of columns.\n\n    Parameters:\n    df (DataFrame): The input DataFrame, containing numeric or categorical data.\n    dct (dict): A dictionary for replacing values in df, where keys are existing values and values are new values.\n\n    Returns:\n    DataFrame: A DataFrame with the correlation coefficients between each pair of columns. The format of the DataFrame is a square matrix with column and index labels matching the columns of the input DataFrame.\n    \n    Requirements:\n    - pandas\n    - numpy\n    \n    Note:\n    - This function operates on DataFrames containing numeric or categorical data that can be replaced with numeric values, as correlation calculations require numeric data.\n    - This function using pearson method to calculate the correlation matrix.\n    \n    Raises:\n    - This function will raise a ValueError is input df is not a DataFrame.\n        \n    Example:\n    >>> df = pd.DataFrame({\'A\': [1, 2, 3], \'B\': [4, 5, 6]})\n    >>> dct = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60}\n    >>> correlation_matrix = task_func(df, dct)\n    >>> correlation_matrix.shape == (2, 2)\n    True\n    >>> np.allclose(correlation_matrix, np.array([[1.0, 1.0], [1.0, 1.0]]))\n    True\n    '''\n', 'code_prompt_for_answer': "import pandas as pd\nimport numpy as np\n# Constants\nCOLUMNS = ['column1', 'column2', 'column3', 'column4', 'column5']\ndef task_func(df, dct):\n", 'unit_tests_you_have_written': {}, 'current_solution': ''}}</1 environment step(s) in the past>

Your next actions / thought:
"""


###

hard_system = """
# Premise
You are a software engineer
Here is some information about this setting
<Setting Information>
You are working to solve a computer science problem. You will need to submit a solution to the problem, which will be tested against a suite of hidden unit tests.
</Setting Information>
<Actions Available>
<edit_submission>
<action_context>
Edit the submission code. Use this when you want to make changes to the current solution.
</action_context>
<action_arg_spec>
{'first_line': <class 'int'>, 'last_line': <class 'int'>, 'new_code': <class 'str'>}
</action_arg_spec>
<action_description>
Edit the submission code
</action_description>

</edit_submission>
<add_submission>
<action_context>
Add the submission code. Use this when you want to start from scratch with a new solution.
</action_context>
<action_arg_spec>
{'submission': <class 'str'>}
</action_arg_spec>
<action_description>
Add the submission code
</action_description>

</add_submission>
<add_unit_test>
<action_context>
Add a unit test. The unit test information you submit must be in the format of a BCBUnitTest: 

class BCBUnitTest(BaseModel):
    test_description: str
    input_names: List[str]
    input_types: List[str]
    input_values: List[Any]
    assertion_condition: str
    assertion_type: Literal["assertTrue", "assertRaises"] = "assertTrue"


 It will be parsed via BCBUnitTest(**unit_test_dict)



# Some various notes:
1. If an input should be of a type defined by a specific package, add the package name/alias to the type. E.g. "np.ndarray" or "pd.DataFrame". You still should fully define the value for the input_value field e.g. "pd.DataFrame({'a': [1, 2, 3]})"

2. Unit tests will be compiled from the BCBUnitTest class as follows:
    A. For AssertTrue type tests, the test will be compiled as follows:
    ```python
    def test_case(self):
        # {{self.test_description}}

        {{defs}}
        result = {{function_name}}(**{{{{args}}}}})
        self.{{self.assertion_type}}({{self.assertion_condition}})
    ```
    B. For AssertRaises type tests, the test will be compiled as follows:

    ```python
    def test_case(self):
        # {{self.test_description}}
        {{defs}}
        with self.{{self.assertion_type}}({{self.assertion_condition}}):
            {{function_name}}(**{{{{args}}}}})
    ```

    Provide information accordingly.

</action_context>
<action_arg_spec>
{'unit_test_name': <class 'str'>, 'unit_test_dict': typing.Dict}
</action_arg_spec>
<action_description>
Add a unit test
</action_description>

</add_unit_test>
<remove_unit_test>
<action_context>
Remove a unit test
</action_context>
<action_arg_spec>
{'unit_test_name': <class 'str'>}
</action_arg_spec>
<action_description>
Remove a unit test
</action_description>

</remove_unit_test>
<test_submission>
<action_context>
Test the submission
</action_context>
<action_arg_spec>
{}
</action_arg_spec>
<action_description>
Test the submission
</action_description>

</test_submission>
<submit_solution>
<action_context>
Submit the solution
</action_context>
<action_arg_spec>
{}
</action_arg_spec>
<action_description>
Submit the solution
</action_description>

</submit_solution>

</Actions Available>
You'll be given your past actions/thoughts, along with recent raw observations from the environment
The environment one step in the past is your current environment.

# Objective
Please complete the problem by drafting a solution, creating unit tests, improving the solution, and submitting the solution.

# Constraints
You will be given a code_prompt_for_answer, which contains imports and the function signature. Your solution must comprise code that can be appended to code_prompt_for_answer and run as a single script.
 

<User Message> 
# Recent Actions / Thoughts

# Recent Observations
<1 environment step(s) in the past>{'action_result': None, 'environment_state': {'question': 'import pandas as pd\nimport numpy as np\n\n# Constants\nCOLUMNS = [\'column1\', \'column2\', \'column3\', \'column4\', \'column5\']\n\ndef task_func(df, dct):\n   '''\n    Replace certain values in a DataFrame with a dictionary mapping and calculate the Pearson correlation coefficient between each pair of columns.\n\n    Parameters:\n    df (DataFrame): The input DataFrame, containing numeric or categorical data.\n    dct (dict): A dictionary for replacing values in df, where keys are existing values and values are new values.\n\n    Returns:\n    DataFrame: A DataFrame with the correlation coefficients between each pair of columns. The format of the DataFrame is a square matrix with column and index labels matching the columns of the input DataFrame.\n    \n    Requirements:\n    - pandas\n    - numpy\n    \n    Note:\n    - This function operates on DataFrames containing numeric or categorical data that can be replaced with numeric values, as correlation calculations require numeric data.\n    - This function using pearson method to calculate the correlation matrix.\n    \n    Raises:\n    - This function will raise a ValueError is input df is not a DataFrame.\n        \n    Example:\n    >>> df = pd.DataFrame({\'A\': [1, 2, 3], \'B\': [4, 5, 6]})\n    >>> dct = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60}\n    >>> correlation_matrix = task_func(df, dct)\n    >>> correlation_matrix.shape == (2, 2)\n    True\n    >>> np.allclose(correlation_matrix, np.array([[1.0, 1.0], [1.0, 1.0]]))\n    True\n    '''\n', 'code_prompt_for_answer': "import pandas as pd\nimport numpy as np\n# Constants\nCOLUMNS = ['column1', 'column2', 'column3', 'column4', 'column5']\ndef task_func(df, dct):\n", 'unit_tests_you_have_written': {}, 'current_solution': ''}}</1 environment step(s) in the past>

Your next actions / thought:  

Structured output: reasoning="I need to implement the function 'task_func' that replaces values in a DataFrame based on a dictionary and calculates the Pearson correlation coefficient between the columns. I will also ensure to handle the case where the input is not a DataFrame by raising a ValueError." action_name='edit_submission' action_args=[ActionArgument(key='first_line', value=4), ActionArgument(key='last_line', value=4), ActionArgument(key='new_code', value="    if not isinstance(df, pd.DataFrame):\n        raise ValueError('Input must be a DataFrame')\n    df.replace(dct, inplace=True)\n    return df.corr(method='pearson')")]
<System Message> 
# Premise
You are a software engineer
Here is some information about this setting
<Setting Information>
You are working to solve a computer science problem. You will need to submit a solution to the problem, which will be tested against a suite of hidden unit tests.
</Setting Information>
<Actions Available>
<edit_submission>
<action_context>
Edit the submission code. Use this when you want to make changes to the current solution.
</action_context>
<action_arg_spec>
{'first_line': <class 'int'>, 'last_line': <class 'int'>, 'new_code': <class 'str'>}
</action_arg_spec>
<action_description>
Edit the submission code
</action_description>

</edit_submission>
<add_submission>
<action_context>
Add the submission code. Use this when you want to start from scratch with a new solution.
</action_context>
<action_arg_spec>
{'submission': <class 'str'>}
</action_arg_spec>
<action_description>
Add the submission code
</action_description>

</add_submission>
<add_unit_test>
<action_context>
Add a unit test. The unit test information you submit must be in the format of a BCBUnitTest: 

class BCBUnitTest(BaseModel):
    test_description: str
    input_names: List[str]
    input_types: List[str]
    input_values: List[Any]
    assertion_condition: str
    assertion_type: Literal["assertTrue", "assertRaises"] = "assertTrue"


 It will be parsed via BCBUnitTest(**unit_test_dict)



# Some various notes:
1. If an input should be of a type defined by a specific package, add the package name/alias to the type. E.g. "np.ndarray" or "pd.DataFrame". You still should fully define the value for the input_value field e.g. "pd.DataFrame({'a': [1, 2, 3]})"

2. Unit tests will be compiled from the BCBUnitTest class as follows:
    A. For AssertTrue type tests, the test will be compiled as follows:
    ```python
    def test_case(self):
        # {{self.test_description}}

        {{defs}}
        result = {{function_name}}(**{{{{args}}}}})
        self.{{self.assertion_type}}({{self.assertion_condition}})
    ```
    B. For AssertRaises type tests, the test will be compiled as follows:

    ```python
    def test_case(self):
        # {{self.test_description}}
        {{defs}}
        with self.{{self.assertion_type}}({{self.assertion_condition}}):
            {{function_name}}(**{{{{args}}}}})
    ```

    Provide information accordingly.

</action_context>
<action_arg_spec>
{'unit_test_name': <class 'str'>, 'unit_test_dict': typing.Dict}
</action_arg_spec>
<action_description>
Add a unit test
</action_description>

</add_unit_test>
<remove_unit_test>
<action_context>
Remove a unit test
</action_context>
<action_arg_spec>
{'unit_test_name': <class 'str'>}
</action_arg_spec>
<action_description>
Remove a unit test
</action_description>

</remove_unit_test>
<test_submission>
<action_context>
Test the submission
</action_context>
<action_arg_spec>
{}
</action_arg_spec>
<action_description>
Test the submission
</action_description>

</test_submission>
<submit_solution>
<action_context>
Submit the solution
</action_context>
<action_arg_spec>
{}
</action_arg_spec>
<action_description>
Submit the solution
</action_description>

</submit_solution>

</Actions Available>
You'll be given your past actions/thoughts, along with recent raw observations from the environment
The environment one step in the past is your current environment.

# Objective
Please complete the problem by drafting a solution, creating unit tests, improving the solution, and submitting the solution.

# Constraints
You will be given a code_prompt_for_answer, which contains imports and the function signature. Your solution must comprise code that can be appended to code_prompt_for_answer and run as a single script.
"""

hard_user = """
# Recent Actions / Thoughts
<1 reasoning step(s) in the past>reasoning="I need to implement the function 'task_func' that replaces values in a DataFrame based on a dictionary and calculates the Pearson correlation coefficient between the columns. I will also ensure to handle the case where the input is not a DataFrame by raising a ValueError." action_name='edit_submission' action_args=[ActionArgument(key='first_line', value=4), ActionArgument(key='last_line', value=4), ActionArgument(key='new_code', value="    if not isinstance(df, pd.DataFrame):\n        raise ValueError('Input must be a DataFrame')\n    df.replace(dct, inplace=True)\n    return df.corr(method='pearson')")]</1 reasoning step(s) in the past>
# Recent Observations
<1 environment step(s) in the past>success=True result='Edited submission successfully'</1 environment step(s) in the past>
<2 environment step(s) in the past>{'action_result': None, 'environment_state': {'question': 'import pandas as pd\nimport numpy as np\n\n# Constants\nCOLUMNS = [\'column1\', \'column2\', \'column3\', \'column4\', \'column5\']\n\ndef task_func(df, dct):\n    '''\n    Replace certain values in a DataFrame with a dictionary mapping and calculate the Pearson correlation coefficient between each pair of columns.\n\n    Parameters:\n    df (DataFrame): The input DataFrame, containing numeric or categorical data.\n    dct (dict): A dictionary for replacing values in df, where keys are existing values and values are new values.\n\n    Returns:\n    DataFrame: A DataFrame with the correlation coefficients between each pair of columns. The format of the DataFrame is a square matrix with column and index labels matching the columns of the input DataFrame.\n    \n    Requirements:\n    - pandas\n    - numpy\n    \n    Note:\n    - This function operates on DataFrames containing numeric or categorical data that can be replaced with numeric values, as correlation calculations require numeric data.\n    - This function using pearson method to calculate the correlation matrix.\n    \n    Raises:\n    - This function will raise a ValueError is input df is not a DataFrame.\n        \n    Example:\n    >>> df = pd.DataFrame({\'A\': [1, 2, 3], \'B\': [4, 5, 6]})\n    >>> dct = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60}\n    >>> correlation_matrix = task_func(df, dct)\n    >>> correlation_matrix.shape == (2, 2)\n    True\n    >>> np.allclose(correlation_matrix, np.array([[1.0, 1.0], [1.0, 1.0]]))\n    True\n    '''\n', 'code_prompt_for_answer': "import pandas as pd\nimport numpy as np\n# Constants\nCOLUMNS = ['column1', 'column2', 'column3', 'column4', 'column5']\ndef task_func(df, dct):\n", 'unit_tests_you_have_written': {}, 'current_solution': ''}}</2 environment step(s) in the past>

Your next actions / thought:
"""


class TestLMStructuredOutputs(unittest.TestCase):
    # ... existing code ...

    @classmethod
    def setUpClass(cls):
        # Initialize LMs for both forced_json and stringified_json modes
        cls.lm_forced_json = LM(
            model_name="gpt-4o-mini",
            formatting_model_name="gpt-4o-mini",
            temperature=0.0,
            max_retries="Few",
            structured_output_mode="forced_json",
        )
        cls.lm_stringified_json = LM(
            model_name="gpt-4o-mini",
            formatting_model_name="gpt-4o-mini",
            temperature=0.0,
            max_retries="Few",
            structured_output_mode="stringified_json",
        )

    def test_sync_react_response_content(self):
        system_message = system

        user_message = user

        for lm in [self.lm_forced_json, self.lm_stringified_json]:
            with self.subTest(
                mode=lm.structured_output_handler.handler.structured_output_mode
            ):
                result = lm.respond_sync(
                    system_message=system_message,
                    user_message=user_message,
                    response_model=ReAct,
                )
                self.assertIsInstance(result, ReAct)
                self.assertIsInstance(result.reasoning, str)
                self.assertIsInstance(result.action_name, str)
                self.assertIsInstance(result.action_args, list)
                for arg in result.action_args:
                    self.assertIsInstance(arg, ActionArgument)
                    self.assertIsInstance(arg.key, str)
                    # self.assertIsInstance(arg.value, str)

    def test_sync_react_response_hard_content(self):
        system_message = hard_system

        user_message = hard_user

        for lm in [self.lm_forced_json, self.lm_stringified_json]:
            with self.subTest(
                mode=lm.structured_output_handler.handler.structured_output_mode
            ):
                result = lm.respond_sync(
                    system_message=system_message,
                    user_message=user_message,
                    response_model=ReAct,
                )
                self.assertIsInstance(result, ReAct)
                self.assertIsInstance(result.reasoning, str)
                self.assertIsInstance(result.action_name, str)
                self.assertIsInstance(result.action_args, list)
                for arg in result.action_args:
                    self.assertIsInstance(arg, ActionArgument)
                    self.assertIsInstance(arg.key, str)
                    # self.assertIsInstance(arg.value, str)


# use non-trivial fallback?

if __name__ == "__main__":
    # Create an instance of the test class
    test_instance = TestLMStructuredOutputs()

    # Set up the class (this would normally be done by unittest)
    test_instance.setUpClass()

    # Run the test methods
    test_instance.test_sync_react_response_content()
    test_instance.test_sync_react_response_hard_content()

    print("All tests completed.")
