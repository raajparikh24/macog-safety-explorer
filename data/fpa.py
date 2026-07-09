"""
Module containing the `FacilityProfileAnalyzer` object class used to perform 
facility profile analyses.

Created: 6/6/2023 by Tariq Shihadah
"""


# =============================================================================
# IMPORT DEPENDENCIES
# =============================================================================

from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mapclassify import NaturalBreaks
from tooles.utils.utils import format_numbers_in_string


# =============================================================================
# DEFINE CLASS
# =============================================================================

class FacilityProfileAnalyzer(object):
    """
    Object class for performing facility profile analysis using the decision 
    tree regression machine learning algorithm.

    The input dataframe should contain multiple categorical, numerical, or 
    binary attribute fields representing independent variables which can be 
    used to distinguish and subset facilities, as well as a target variable 
    field with numeric values that will be optimized by the model.

    Parameters
    ----------
    df : pd.DataFrame or subtype
        Input data to build the model with and to apply the model to.
    target : label
        Label within the input dataframe for the column containing the 
        dependent variable data which will be used to train the model.
    categorical, numerical, binary : list of labels, default []
        Labels within the input dataframe for columns containing categorical, 
        numerical, or binary independent variable data which will be used to 
        train the model. At least one label should be included between all 
        three lists.
    weights : label, optional
        Label within the input dataframe for a numerical column containing 
        values which should be used to weight each record during fitting. For 
        analysis of segments with variable lengths, this should be a column of  
        segment lengths.
    exclude : list, optional
        List of categories within categorical columns which should ignored 
        by the model.
    query : str, optional
        Query to be applied to the input dataframe which will filter down the 
        data prior to training.
    """

    def __init__(
        self,
        df,
        target,
        categorical=[],
        numerical=[],
        binary=[],
        weights=None,
        exclude=None,
        query=None,
        **kwargs
    ):
        # Validate input
        self.df = df
        self.query = query
        self.target = target
        self.categorical = categorical
        self.numerical = numerical
        self.binary = binary
        self.weights = weights
        self.exclude = exclude

    def __repr__(self):
        text = f"""\
Facility Profile Analyzer
=========================
- Status: {'Not fit' if (self._model is None) else 'Fit'}

- Target variable label: {self.target}

- Categorical labels:    {self.categorical}
- Numerical labels:      {self.numerical}
- Binary labels:         {self.binary}
"""
        return text

    @property
    def df(self):
        if self._query is None:
            return self._df
        else:
            return self._df[self._query_mask]

    @df.setter
    def df(self, obj):
        # Validate dataframe
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(
                "Input data must be pd.DataFrame.")
        self._df = obj

    @property
    def df_full(self):
        return self._df

    @property
    def query(self):
        return self._query
    
    @query.setter
    def query(self, obj):
        # Evaluate query
        if not obj is None:
            try:
                query_mask = self._df.eval(obj)
            except:
                raise ValueError(f"Unable to evaluate data query:\n{obj}")
        else:
            query_mask = np.ones(self._df.shape[0], dtype=bool)
        # Log query
        self._query = obj
        self._query_mask = query_mask

    @property
    def exclude(self):
        return self._exclude
    
    @exclude.setter
    def exclude(self, obj):
        # Validate
        if not isinstance(obj, list):
            raise TypeError("Input `exclude` parameter must be a list.")
        self._exclude = obj

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, label):
        # Validate column label
        self._validate_column_label(label)
        # Validate numeric data
        try:
            assert pd.api.types.is_numeric_dtype(self._df[label])
        except:
            raise ValueError(
                "Selected target variable column must contain numeric data.")
        self._target = label
        # Reset y data
        self._y = None

    @property
    def categorical(self):
        return self._categorical

    @categorical.setter
    def categorical(self, labels):
        # Validate column labels
        self._validate_column_labels(labels)
        self._categorical = labels
        self.reset_model()
        
    @property
    def numerical(self):
        return self._numerical

    @numerical.setter
    def numerical(self, labels):
        # Validate column labels
        self._validate_column_labels(labels)
        self._numerical = labels
        self.reset_model()
                
    @property
    def binary(self):
        return self._binary

    @binary.setter
    def binary(self, labels):
        # Validate column labels
        self._validate_column_labels(labels)
        self._binary = labels
        self.reset_model()

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, label):
        # Validate column labels
        if not label is None:
            self._validate_column_label(label)
        self._weights = label
        self.reset_model()

    @property
    def sample_weights(self):
        """
        Weights associated with all samples within the input dataframe which 
        will be used to normalize the relative influence of each sample.
        """
        if not self.weights is None:
            return self.df[self.weights].values
        else:
            return np.ones(self.df.shape[0])

    @property
    def x(self):
        """
        Independent model variables used as X values to fit the model. Honors 
        data query.
        """
        # Generate data from independent model variable columns
        return self.prepare_data()[self._query_mask]
    
    @property
    def x_full(self):
        """
        Independent model variables used as X values to fit the model. Ignores 
        data query.
        """
        # Generate data from independent model variable columns
        return self.prepare_data()

    @property
    def y(self):
        """
        Dependent model variables used as Y values to fit the model.
        """
        # Generate data from dependent model variable column
        return self.df[self._target]

    @property
    def y_full(self):
        """
        Dependent model variables used as Y values to fit the model.
        """
        # Generate data from dependent model variable column
        return self.df_full[self._target]

    @property
    def variable_names(self):
        return np.array(self._categorical + self._numerical + self._binary)

    @property
    def feature_names(self):
        return np.array(self.x.columns)

    @property
    def feature_types(self):
        num_numerical = len(self._numerical)
        num_binary = len(self._binary)
        num_categorical = self.x.shape[1] - num_numerical - num_binary
        return np.array(
            ['categorical'] * num_categorical + 
            ['numerical'] * num_numerical + 
            ['binary'] * num_binary
        )

    @property
    def feature_variables(self):
        feature_variables = []
        for i in self._categorical:
            for _ in range(len(self.df[i][~self.df[i].isin(self.exclude)].unique())):
                feature_variables.append(i)
        for i in self._numerical:
            feature_variables.append(i)
        for i in self._binary:
            feature_variables.append(i)
        return np.array(feature_variables)

    @property
    def model(self):
        return self._model

    def _validate_column_label(self, label):
        # Check for valid input label
        if not label in self._df.columns:
            raise ValueError(
                f"Input column label `{label}` is not present in the input "
                "dataframe.")

    def _validate_column_labels(self, labels):
        # Validate all input labels
        [self._validate_column_label(label) for label in labels]

    def _create_dummies(self, df=None):
        # Select dataframe if provided
        if df is None:
            df = self.df
        elif not isinstance(df, pd.DataFrame):
            raise TypeError("Input data must be a pd.DataFrame instance.")
        # Iterate through categorical columns to create dummies
        if len(self._categorical) > 0:
            dummies = []
            for label in self._categorical:
                dummy = pd.get_dummies(df[label])
                dummy = dummy.drop(columns=self.exclude, errors='ignore')
                dummies.append(dummy)
            dummies = pd.concat(dummies, axis=1, join='outer')
        # If no categorical columns, return empty dataframe
        else:
            dummies = pd.DataFrame(index=df.index)
        # Enforce text column labels
        dummies.columns = [str(label) for label in dummies.columns]
        return dummies

    @property
    def default_response_behaviour_dict(self):
        default_response_behaviour_dict = {
            "inverted_list": True,
            "format_numbers": None,
            "binary_recode": None,
        }
        return default_response_behaviour_dict

    def _get_node_responses(self):
        """
        Return a list of left and right answers to node questions.
        """
        # Retrieve node-level information
        node_leaves = self.model.tree_.children_left < 0
        node_feature_names = np.where(
            ~node_leaves, self.feature_names[self.model.tree_.feature], np.nan)
        node_types = np.where(
            ~node_leaves, self.feature_types[self.model.tree_.feature], np.nan)
        node_thresholds = self.model.tree_.threshold
        node_feature_variables = np.where(
            ~node_leaves, self.feature_variables[self.model.tree_.feature], np.nan)
        node_thresholds = self.model.tree_.threshold
        
        # Describe node question responses
        node_responses = [] # list of form [left, right]
        for i, is_leaf in enumerate(node_leaves):
            # Check for leaf
            if is_leaf:
                node_responses.append([
                    {
                        'text': 'NA',
                        'feature': 'NA',
                        'variable': 'NA',
                        'numeric_value': 'NA',
                        'numeric_dir': 'NA',
                        'response': 'NA',
                        'type': 'NA'
                    },
                    {
                        'text': 'NA',
                        'feature': 'NA',
                        'variable': 'NA',
                        'numeric_value': 'NA',
                        'numeric_dir': 'NA',
                        'response': 'NA',
                        'type': 'NA'
                    },
                ])
                continue
            # Define questions
            if node_types[i] == 'categorical':
                node_responses.append([
                    {
                        'text': f'Is Not: {node_feature_names[i]}',
                        'feature': node_feature_names[i],
                        'variable': node_feature_variables[i],
                        'numeric_value': 'NA',
                        'numeric_dir': 'NA',
                        'response': False,
                        'type': 'categorical'
                    },
                    {
                        'text': f'Is: {node_feature_names[i]}',
                        'feature': node_feature_names[i],
                        'variable': node_feature_variables[i],
                        'numeric_value': 'NA',
                        'numeric_dir': 'NA',
                        'response': True,
                        'type': 'categorical'
                    }
                ])
            elif node_types[i] == 'numerical':
                node_responses.append([
                    {
                        'text': f'{node_feature_names[i]} ≤ {node_thresholds[i]:.3f}',
                        'feature': node_feature_names[i],
                        'variable': node_feature_variables[i],
                        'numeric_value': f'{node_thresholds[i]:.3f}',
                        'numeric_dir': '≤',
                        'response': f'≤ {node_thresholds[i]:.3f}',
                        'type': 'numerical'
                    },
                    {
                        'text': f'{node_feature_names[i]} > {node_thresholds[i]:.3f}',
                        'feature': node_feature_names[i],
                        'variable': node_feature_variables[i],
                        'numeric_value': f'{node_thresholds[i]:.3f}',
                        'numeric_dir': '>',
                        'response': f'> {node_thresholds[i]:.3f}',
                        'type': 'numerical'
                    }
                ])
            else: # binary
                node_responses.append([
                    {
                        'text': 'Is Not: {node_feature_names[i]}',
                        'feature': node_feature_names[i],
                        'variable': node_feature_variables[i],
                        'numeric_value': 'NA',
                        'numeric_dir': 'NA',
                        'response': False,
                        'type': 'binary'
                    },
                    {
                        'text': 'Is: {node_feature_names[i]}',
                        'feature': node_feature_names[i],
                        'variable': node_feature_variables[i],
                        'numeric_value': 'NA',
                        'numeric_dir': 'NA',
                        'response': True,
                        'type': 'binary'
                    }
                ])
        return node_responses

    def _get_node_response_path(self, node_id):
        """
        Describe the selected node in readable text in terms of the decisions 
        made to arrive at the node.
        """
        # Initialize node responses
        node_responses = self._get_node_responses()
        node_path = [node_id]
        node_response_path = []
        
        # Recursively search through tree structure for path
        while True:
            
            # Check left children
            search = self.model.tree_.children_left==node_id
            if search.sum() > 0:
                # Update node ID
                node_id = search.argmax()
                node_path.append(node_id)
                # Get question response
                node_response_path.append(node_responses[node_id][0]) # Left response
                continue
                
            # Check right children
            search = self.model.tree_.children_right==node_id
            if search.sum() > 0:
                # Update node ID
                node_id = search.argmax()
                node_path.append(node_id)
                # Get question response
                node_response_path.append(node_responses[node_id][1]) # Right response
                continue
        
            # No remaining matches, break
            break
        return node_response_path[::-1]

    def _get_all_node_response_paths(self):
        """
        Generate node response paths for all nodes in the tree model.
        """
        # Iterate through all nodes
        paths = []
        for node_id in range(self.model.tree_.node_count):
            paths.append(self._get_node_response_path(node_id))
        return paths

    def _clean_responses(self, responses):
        """
        Cleans up node responses for easier human parsing.
        """
        # make clean return list
        return_list = []
        numerical_responses_list = []
        # Go through the responses and see if they are numerical
        for i in range(len(responses)):
            # If they are numerical, add them to a list that will go to a cleaning function
            if responses[i]["type"] == "numerical":
                numerical_responses_list.append(responses[i])
            # if the the type is not numerical, just return as is
            else:
                return_list.append(responses[i])
        # if there are numerical, clean them
        if len(numerical_responses_list) > 0:
            numeric_return_list = self._clean_numerical_responses(
                numerical_responses_list
            )
            return_list = return_list + numeric_return_list

        return return_list

    def _clean_numerical_responses(self, numerical_responses):
        """
        Cleans up numeric type node responses for easier human parsing.
        """
        # make clean return list
        numeric_return_list = []
        # get the numerical variables
        numericals = [i["variable"] for i in numerical_responses]
        # make a list that will track if some variables have multiple instances
        multi_numerics = []
        # for each numerical, see if there are multiple instances of them
        for numerical in numericals:
            # if there is only one instance of the numerical, add it back to the the return list as is
            if numericals.count(numerical) == 1:
                numeric_return_list.append(
                    [
                        numerical_responses[i]
                        for i in range(len(numerical_responses))
                        if numerical_responses[i]["variable"] == numerical
                    ][0]
                )
            # if there are, make a list of them
            else:
                if numerical not in multi_numerics:
                    multi_numerics.append(numerical)
        # for ones with multiple instances, clean them up separately
        if len(multi_numerics) > 0:
            for multi_numeric in multi_numerics:
                multi_list = [
                    numerical_responses[i]
                    for i in range(len(numerical_responses))
                    if numerical_responses[i]["variable"] == multi_numeric
                ]
                numeric_return_list.append(self._clean_multiple_numerical(multi_list))
        return numeric_return_list

    def _clean_multiple_numerical(self, multi_numerical_responses):
        """
        Cleans up numeric type nodes with multiple responses for the same variable for easier human parsing.
        """
        # get the directions
        numeric_dirs = [
            multi_numerical_responses[i]["numeric_dir"]
            for i in range(len(multi_numerical_responses))
        ]
        # get the unique directions
        numeric_dirs_unique = list(set(numeric_dirs))
        # for instances where there's only one direction, get the min/max
        if len(numeric_dirs_unique) == 1:
            # for less than, get the min of all the values
            if numeric_dirs_unique[0] == "≤":
                new_value = min(
                    [
                        multi_numerical_responses[i]["numeric_value"]
                        for i in range(len(multi_numerical_responses))
                    ]
                )
                new_dir = "≤"
            if numeric_dirs_unique[0] == ">":
            # for greater than, get the max of all the values
                new_value = {
                    max(
                        [
                            multi_numerical_responses[i]["numeric_value"]
                            for i in range(len(multi_numerical_responses))
                        ]
                    )
                }
                new_dir = ">"
            # create the new text and response values
            new_text = (
                f"{multi_numerical_responses[0]['feature']} {new_dir} {new_value}"
            )
            new_response = f"{new_dir} {new_value}"
        # for instances where there are two direction, get the range of values
        elif len(numeric_dirs_unique) == 2:
            max_value = max(
                [
                    multi_numerical_responses[i]["numeric_value"]
                    for i in range(len(multi_numerical_responses))
                    if multi_numerical_responses[i]["numeric_dir"] == "≤"
                ]
            )
            min_value = min(
                [
                    multi_numerical_responses[i]["numeric_value"]
                    for i in range(len(multi_numerical_responses))
                    if multi_numerical_responses[i]["numeric_dir"] == ">"
                ]
            )
            new_value = [min_value, max_value]
            new_dir = "-"
            new_text = f"{multi_numerical_responses[0]['feature']} {min_value} {new_dir} {max_value}"
            new_response = f"{min_value} {new_dir} {max_value}"
        else:
            raise ValueError("Haven't figured out how to handle this yet")

        cleaned_response = {
            "text": new_text,
            "feature": multi_numerical_responses[0]["feature"],
            "variable": multi_numerical_responses[0]["variable"],
            "numeric_value": new_value,
            "numeric_dir": new_dir,
            "response": new_response,
            "type": multi_numerical_responses[0]["type"],
        }

        return cleaned_response
            
    def prepare_data(self, df=None):
        """
        Prepare an input dataframe containing fields consistent with the 
        original modeling dataframe to prepare X data which can be run 
        through the model.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Input dataframe containing all fields in the `variable_names` 
            property which will be transformed to match the `feature_names` 
            property with dummies for categorical data. If not provided, 
            the original, unfiltered modeling dataframe will be used.
        """
        # Select dataframe if provided
        if df is None:
            df = self.df_full
        # Generate data from independent model variable columns
        data = pd.concat([
            self._create_dummies(df),
            df[self._numerical],
            df[self._binary]
        ], axis=1, join='outer')
        return data

    def reset_model(self):
        # Reset model
        self._model = None
        # Reset figure
        self._fig = None

    def fit(self, **kwargs):
        """
        Fit the `DecisionTreeRegressor` model with the generated X and Y data. 
        Provided `kwargs` will be passed to the `DecisionTreeRegressor` 
        constructor.
        """
        # Construct regressor
        self._model = DecisionTreeRegressor(**kwargs)
        # Fit data
        self._model.fit(self.x, self.y, sample_weight=self.sample_weights)

    def plot(self, figsize=None, fontsize=None, **kwargs):
        """
        Create a figure visualizing the structure of the fitted decision tree 
        model.
        """
        # Create figure
        self._fig = plt.figure(figsize=figsize)
        ax = self._fig.subplots(1,1)
        # Plot model
        tree.plot_tree(
            self._model,
            ax=ax,
            fontsize=fontsize,
            feature_names=list(self.feature_names),
            **kwargs
        )

    def plot_graphviz(self, fp=None, format='png', **kwargs):
        """
        Create and export a figure visualizing the structure of the fitted 
        decision tree model using the graphviz package and executable (if 
        it is available). If no export filepath is provided, do not save 
        the generated figure.
        """
        # Import minor dependency
        try:
            import graphviz
        except:
            raise ImportError(
                "The `graphviz` package and/or dependencies are not "
                "installed.")

        # Generate the dot data for the figure
        dot_data = tree.export_graphviz(
            self.model, feature_names=self.feature_names, **kwargs)
        graph = graphviz.Source(dot_data)

        # Export the figure
        if not fp is None:
            graph.format = format
            graph.render(fp)
        return graph

    def predict(self, X=None, query='filter', **kwargs):
        """
        Perform a prediction on the input independent variable data using the 
        fitted model. Input data must contain all fields shown in the 
        `feature_names` class property.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Dataframe containing the required independent variable data 
            fields. If not provided, the X data used to generate the model 
            will be used.
        query : {'filter', 'predict', 'null'}
            How to treat filtered data if no X data is provided and a query is 
            present on the analyzer instance.

            Options
            -------
            filter : perform prediction on filtered data set, returning 
                prediction results that are aligned with the filtered 
                dataframe.
            predict : perform prediction on full data set, returning 
                prediction results that are aligned with the full dataframe.
            null : perform prediction on filtered data set, returning 
                prediction results that are aligned with the full dataframe, 
                filling filtered rows with null values.
        """
        # Validate query parameter
        _ops_query = ['filter', 'predict', 'null']
        if not query in _ops_query:
            raise ValueError(
                f"Input `query` parameter must be one of {_ops_query}")
        # Check for input data
        if X is None:
            if query in ['predict', 'null']:
                X = self.x_full
            else:
                X = self.x
        # Perform prediction on the input data
        prediction = self.model.predict(X, **kwargs)
        if query=='null':
            prediction[~self._query_mask] = np.nan
        return prediction
        
    def apply(self, X=None, query='filter', **kwargs):
        """
        Perform a prediction on the input independent variable data using the 
        fitted model, outputting the identified tree leaf node ID. Input data 
        must contain all fields shown in the `feature_names` class property.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Dataframe containing the required independent variable data 
            fields. If not provided, the X data used to generate the model 
            will be used.
        query : {'filter', 'predict', 'null'}
            How to treat filtered data if no X data is provided and a query is 
            present on the analyzer instance.

            Options
            -------
            filter : perform prediction on filtered data set, returning 
                prediction results that are aligned with the filtered 
                dataframe.
            predict : perform prediction on full data set, returning 
                prediction results that are aligned with the full dataframe.
            null : perform prediction on filtered data set, returning 
                prediction results that are aligned with the full dataframe, 
                filling filtered rows with null values.
        """
        # Validate query parameter
        _ops_query = ['filter', 'predict', 'null']
        if not query in _ops_query:
            raise ValueError(
                f"Input `query` parameter must be one of {_ops_query}")
        # Check for input data
        if X is None:
            if query in ['predict', 'null']:
                X = self.x_full
            else:
                X = self.x
        # Perform prediction on the input data
        prediction = self.model.apply(X, **kwargs)
        if query=='null':
            prediction[~self._query_mask] = np.nan
        return prediction
        
    def describe(self, X=None, sep='; ', query='filter', **kwargs):
        """
        Perform a prediction on the input independent variable data using the 
        fitted model, outputting the readable text description of the tree 
        leaf. Input data must contain all fields shown in the `feature_names` 
        class property.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Dataframe containing the required independent variable data 
            fields. If not provided, the X data used to generate the model 
            will be used.
        sep : str, default '; '
            String separator to use to join the individual node response path 
            elements (e.g., decisions).
        query : {'filter', 'predict', 'null'}
            How to treat filtered data if no X data is provided and a query is 
            present on the analyzer instance.

            Options
            -------
            filter : perform prediction on filtered data set, returning 
                prediction results that are aligned with the filtered 
                dataframe.
            predict : perform prediction on full data set, returning 
                prediction results that are aligned with the full dataframe.
            null : perform prediction on filtered data set, returning 
                prediction results that are aligned with the full dataframe, 
                filling filtered rows with 'N/A'.
        """
        # Validate query parameter
        _ops_query = ['filter', 'predict', 'null']
        if not query in _ops_query:
            raise ValueError(
                f"Input `query` parameter must be one of {_ops_query}")
        # Check for input data
        if X is None:
            if query in ['predict', 'null']:
                X = self.x_full
            else:
                X = self.x
        # Prepare response path information
        response_paths = self._get_all_node_response_paths()
        response_paths = np.array(
            [sep.join([i['text'] for i in path]) for path in response_paths]
        )
        # Perform prediction on the input data
        selector = self.model.apply(X, **kwargs)
        prediction = response_paths[selector]
        if query=='null':
            prediction[~self._query_mask] = 'N/A'
        return prediction

    def tabulate(self, X=None, sep='; ', query='filter', **kwargs):
        """
        Perform a prediction on the input independent variable data using the 
        fitted model, outputting the readable text description of the tree 
        leaf. Input data must contain all fields shown in the `feature_names` 
        class property.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Dataframe containing the required independent variable data 
            fields. If not provided, the X data used to generate the model 
            will be used.
        sep : str, default '; '
            String separator to use to join the individual node response path 
            elements (e.g., decisions).
        query : {'filter', 'predict', 'null'}
            How to treat filtered data if no X data is provided and a query is 
            present on the analyzer instance.

            Options
            -------
            filter : perform prediction on filtered data set, returning 
                prediction results that are aligned with the filtered 
                dataframe.
            predict : perform prediction on full data set, returning 
                prediction results that are aligned with the full dataframe.
            null : perform prediction on filtered data set, returning 
                prediction results that are aligned with the full dataframe, 
                filling filtered rows with 'N/A'.
        """
        # Validate query parameter
        _ops_query = ['filter', 'predict', 'null']
        if not query in _ops_query:
            raise ValueError(
                f"Input `query` parameter must be one of {_ops_query}")
        # Check for input data
        if X is None:
            if query in ['predict', 'null']:
                X = self.x_full
            else:
                X = self.x
        # Prepare response path information
        response_paths = self._get_all_node_response_paths()
        response_data = []
        for path in response_paths:
            sub = {var: [] for var in self.variable_names}
            for i in path:
                sub[i['feature']].append(i['response'])
            response_data.append(sub)
        response_paths = np.array(response_data)

        # Perform prediction on the input data
        selector = self.model.apply(X, **kwargs)
        prediction = response_paths[selector]
        if query=='null':
            prediction[~self._query_mask] = 'N/A'
        return prediction

    def tier(
        self, X=None, tiers=['Critical', 'High', 'Medium', 'Low', 'Minimal'], 
        query='filter', **kwargs):
        """
        Perform a prediction on the input independent variable data using the 
        fitted model, outputting tier levels associated with each data record, 
        broken into the provided tiers using the Jenks Natural Breaks 
        algorithm.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Dataframe containing the required independent variable data 
            fields. If not provided, the X data used to generate the model 
            will be used.
        tiers : list, default ['Critical', 'High', 'Medium', 'Low', 'Minimal']
            List of tier names to use to classify the relative predicted model 
            values using Jenks Natural Breaks. Tier names should be listed in 
            descending order.
        query : {'filter', 'predict', 'null'}
            How to treat filtered data if no X data is provided and a query is 
            present on the analyzer instance.

            Options
            -------
            filter : perform prediction on filtered data set, returning 
                prediction results that are aligned with the filtered 
                dataframe.
            predict : perform prediction on full data set, returning 
                prediction results that are aligned with the full dataframe.
            null : perform prediction on filtered data set, returning 
                prediction results that are aligned with the full dataframe, 
                filling filtered rows with 'N/A'.
        """
        # Validate query parameter
        _ops_query = ['filter', 'predict', 'null']
        if not query in _ops_query:
            raise ValueError(
                f"Input `query` parameter must be one of {_ops_query}")
        # Check for input data
        if X is None:
            if query in ['predict', 'null']:
                X = self.x_full
            else:
                X = self.x
        # Perform prediction on the input data
        prediction = self.model.predict(X, **kwargs)
        nb = NaturalBreaks(prediction, len(tiers))
        prediction = np.array(tiers[::-1])[nb.yb]
        if query=='null':
            prediction[~self._query_mask] = np.nan
        return prediction

    def summarize(self, target=None, weights=None, weights_label='Weights', 
                  target_label='Target', sep='; ',
                  tiers=['Critical', 'High', 'Medium', 'Low', 'Minimal']):
        """
        Create a dataframe which summarizes all defined facility profiles and 
        their relative share of the total weights and total target values.

        Parameters
        ----------
        target, weights : labels, optional
            Labels for alternative columns within the target dataframe which 
            should be used to quantify the optimization of predictions. If not 
            provided, will default to using already-available values.
        target_label, weights_label : labels, optional
            Label names to use in the summary table for columns related to the 
            target and weights data being analyzed.
        """
        # Address optional parameters
        if target is None:
            target = self.target
        target_data = self.df[target].values
        if weights is None:
            weights = self.weights
            weights_data = self.df[weights].values
        else:
            weights_data = self.sample_weights
        
        # Create predictions and assignments
        df = pd.DataFrame({
            'Class': self.describe(sep=sep),
            'Prediction': self.predict().round(3),
            'Tier': self.tier(tiers=tiers),
            target_label: target_data.round(3),
            weights_label: weights_data.round(3),
        })
        
        # Summarize by class
        aggregators = {
            'Tier': 'first',
            'Prediction': 'first',
            weights_label: 'sum',
            target_label: 'sum',
        }
        summary = df.groupby('Class').agg(aggregators) \
            .sort_values(by='Prediction', ascending=False)

        # Compute additional metrics
        summary[f'{weights_label} Share'] = \
            (summary[f'{weights_label}'] / summary[f'{weights_label}'].sum()) \
            .apply(lambda x: f'{x:.1%}')
        summary[f'{target_label} Share'] = \
            (summary[f'{target_label}'] / summary[f'{target_label}'].sum()) \
            .apply(lambda x: f'{x:.1%}')
        
        return summary
    
    def response_matrix(self, X=None, query='filter', response_behaviour_dict=None, **kwargs):
        # Validate query parameter
        _ops_query = ['filter', 'predict', 'null']
        if not query in _ops_query:
            raise ValueError(
                f"Input `query` parameter must be one of {_ops_query}")
        # Check for input data
        if X is None:
            if query in ['predict', 'null']:
                X = self.x_full
            else:
                X = self.x
        
        # Deterermine how to handle the responses
        # can also use this is a placeholder argument for future functionality re: ordinal categorical features
        # If not provided, create it with the default args
        if response_behaviour_dict is None:
            response_behaviour_dict = {i:self.default_response_behaviour_dict for i in self.categorical + self.numerical + self.binary}
        else:
            # If dict is provided, make sure that all the colums are accounted for
            for i in self.categorical + self.numerical + self.binary:
                # If the column is not present, add it with the defaults
                if i not in response_behaviour_dict:
                    response_behaviour_dict[i] = self.default_response_behaviour_dict
                # if the colulmn is present, make sure that the keys are present, and if not, add them with the defaults
                else:
                    for key, value in self.default_response_behaviour_dict.items():
                        if key not in response_behaviour_dict[i]:
                            response_behaviour_dict[i][key] = value

        # Prepare response path information
        response_paths = self._get_all_node_response_paths()

        # Make dict of the responses to turn into a data frame
        # for now just using positional index, but might need to adjust
        response_dict = {
            i: self._clean_responses(response_paths[i]) for i in range(len(response_paths))
        }
        
        # make an empty data frame to concat on to
        response_matrix_df = pd.DataFrame()

        # iterate through the response dictionary and make a data frame for class
        for i in response_dict:
            # for each class, make another empty data frame for each response
            class_df = pd.DataFrame()
            # depedning on the type of response, get the value for the df from a different key 
            for j in response_dict[i]:
                if j['type'] == 'numerical':
                    val = j["response"]
                    if response_behaviour_dict[j["variable"]]["format_numbers"] is not None:
                        val = format_numbers_in_string(val, response_behaviour_dict[j["variable"]]["format_numbers"])
                elif j['type'] == 'binary':
                    val = j["response"]
                    if response_behaviour_dict[j["variable"]]["binary_response"] is not None:
                        val = response_behaviour_dict[j["variable"]]["binary_response"][val]
                elif j['type'] == 'categorical':
                    # true, i.e. a certian category, get the feature
                    if j["response"]:
                        val = j["feature"]
                    # false, i.e. not a certain category, get all the other categories (minus exluded ones) and return that
                    elif not j["response"]:
                        # todo: for categorical features that are ordinal, find a way to to get the numeric value and flip the direction
                        # ex: bins = ["0-4", "5-9", "10+"], if the response is "0-4" == False, be able to return "5+" instead of ["5-9", "10+"]
                        if response_behaviour_dict[j["variable"]]["inverted_list"]:
                            uniques=self.df[j["variable"]][~self.df[j["variable"]].isin(self.exclude)].unique()
                            val = "; ".join(list(np.delete(uniques, np.argwhere(uniques==j["feature"]))))
                        else:
                            val = f"Is Not: {j['feature']}"
                # create a new column/row for each response in the class
                feature_df = pd.DataFrame({j["variable"]:val}, index=[i])
                # within the class, concat rows together on the columns
                class_df = pd.concat([class_df, feature_df], axis=1)

            # for the matrix, concat the classes together on the rows
            response_matrix_df = pd.concat([response_matrix_df, class_df], axis=0)
        
        return response_matrix_df