# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pandas as pd

from typing import Optional

from ._result import Result


class MemoryConverter:
    """
    Handles conversion between memory dictionaries and dataframes.

    This class is used at the search level to convert between internal
    memory representations (dictionaries with position tuples as keys)
    and pandas DataFrames for user-facing memory/warm-start functionality.

    Parameters
    ----------
    converter : Converter
        The converter instance that handles position/value/parameter conversions
    """

    def __init__(self, converter):
        self.converter = converter

    def returnNoneIfArgNone(func_):
        """Decorator to return None if any argument is None."""
        def wrapper(self, *args):
            for arg in [*args]:
                if arg is None:
                    return None
            else:
                return func_(self, *args)

        return wrapper

    @returnNoneIfArgNone
    def positions_scores2memory_dict(
        self, positions: Optional[list], scores: Optional[list]
    ) -> Optional[dict]:
        """
        Convert positions and scores to memory dictionary format.

        Parameters
        ----------
        positions : list
            List of position arrays
        scores : list
            List of scores corresponding to positions

        Returns
        -------
        dict
            Memory dictionary with position tuples as keys and Result objects as values
        """
        value_tuple_list = list(map(tuple, positions))
        # Convert scores to Result objects
        result_objects = [Result(float(score), {}) for score in scores]
        memory_dict = dict(zip(value_tuple_list, result_objects))

        return memory_dict

    @returnNoneIfArgNone
    def memory_dict2positions_scores(self, memory_dict: Optional[dict]):
        """
        Convert memory dictionary to positions and scores.

        Parameters
        ----------
        memory_dict : dict
            Memory dictionary with position tuples as keys and Result objects as values

        Returns
        -------
        tuple
            (positions, scores) tuple
        """
        positions = [np.array(pos).astype(int) for pos in list(memory_dict.keys())]
        # Extract scores from Result objects
        scores = [result.score if isinstance(result, Result) else result
                 for result in memory_dict.values()]

        return positions, scores

    @returnNoneIfArgNone
    def dataframe2memory_dict(
        self, dataframe: Optional[pd.DataFrame]
    ) -> Optional[dict]:
        """
        Convert a DataFrame to memory dictionary format.

        Parameters
        ----------
        dataframe : pd.DataFrame
            DataFrame with parameter columns and a 'score' column

        Returns
        -------
        dict
            Memory dictionary with position tuples as keys and Result objects as values
        """
        parameter = set(self.converter.search_space.keys())
        memory_para = set(dataframe.columns)

        if parameter <= memory_para:
            values = list(dataframe[self.converter.para_names].values)
            positions = self.converter.values2positions(values)
            scores = dataframe["score"]

            memory_dict = self.positions_scores2memory_dict(positions, scores)

            return memory_dict
        else:
            missing = parameter - memory_para

            print(
                "\nWarning:",
                '"{}"'.format(*missing),
                "is in search_space but not in memory dataframe",
            )
            print("Optimization run will continue without memory warm start\n")

            return {}

    @returnNoneIfArgNone
    def memory_dict2dataframe(
        self, memory_dict: Optional[dict]
    ) -> Optional[pd.DataFrame]:
        """
        Convert memory dictionary to DataFrame format.

        Parameters
        ----------
        memory_dict : dict
            Memory dictionary with position tuples as keys and Result objects as values

        Returns
        -------
        pd.DataFrame
            DataFrame with parameter columns and a 'score' column
        """
        positions, score = self.memory_dict2positions_scores(memory_dict)
        values = self.converter.positions2values(positions)

        dataframe = pd.DataFrame(values, columns=self.converter.para_names)
        dataframe["score"] = score

        return dataframe
