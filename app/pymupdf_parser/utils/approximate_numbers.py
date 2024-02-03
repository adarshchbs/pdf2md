from app.pymupdf_parser.utils.cluster import moving_avg_cluster_1d
import numpy as np
import pandas as pd
from sortedcollections import SortedDict
from app.typing import Number


class ApproximateNumbersWithThreshold:
    def __init__(self, array: np.ndarray, threshold: Number) -> None:
        """
        It takes an array and a threshold, and returns a dictionary that maps each value in the array to
        the mean of the values in the array that are within the threshold of the value

        :param array: the array to be clustered
        :type array: np.ndarray
        :param threshold: the threshold for the moving average
        :type threshold: Number
        """
        self.threshold = threshold
        self.index = moving_avg_cluster_1d(array, threshold)
        df = pd.DataFrame({"array": array, "index": self.index})
        index_to_mean_dict = df.groupby("index").aggregate(np.mean).to_dict()["array"]
        self.value_to_approximation_dict = SortedDict()
        for value, ind in zip(array, self.index):
            self.value_to_approximation_dict[value] = index_to_mean_dict[ind]

    def approx(self, value: Number, raise_error=True) -> Number:
        """
        If the value is in the dictionary, return it. If it's not, find the closest value in the
        dictionary and return that

        :param value: The value to be approximated
        :type value: Number
        :param raise_error: If True, then if no value is found, an error is raised. If False, then the
        value is returned, defaults to True (optional)
        :return: The value that is closest to the given value.
        """
        if value in self.value_to_approximation_dict:
            return self.value_to_approximation_dict[value]
        lower_index = self.value_to_approximation_dict.bisect_left(value)
        if lower_index == -1:
            lower_index += 1
        just_lower_value_in_dict: Number = self.value_to_approximation_dict.iloc[
            lower_index
        ]
        if np.abs(just_lower_value_in_dict - value) < self.threshold:
            return just_lower_value_in_dict
        upper_index = self.value_to_approximation_dict.bisect_right(value)
        if upper_index == len(self.value_to_approximation_dict):
            upper_index -= 1
        just_upper_value_in_dict = self.value_to_approximation_dict.iloc[upper_index]
        if np.abs(just_upper_value_in_dict - value) < self.threshold:
            return just_upper_value_in_dict

        if raise_error:
            raise ValueError("No value approximate to the given number")
        else:
            return value


if __name__ == "__main__":
    array = np.random.randint(0, 3, 10) + np.random.random(10) / 5
    apx = ApproximateNumbersWithThreshold(array, threshold=0.5)
    approximate_number = [apx.approx(value) for value in array]
    print(f"{array=} \n {approximate_number=}")
