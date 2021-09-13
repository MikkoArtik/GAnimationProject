from typing import NamedTuple
import numpy as np


class ColumnIndex(NamedTuple):
    time: int = 0
    x: int = 1
    y: int = 2
    z: int = 3
    density: int = 4


class Limit(NamedTuple):
    min_val: float
    max_val: float


class InterpolationParameter(NamedTuple):
    time_lim: Limit
    x_lim: Limit
    y_lim: Limit
    cell_size: float
    buffer_distance: float

    @property
    def extent_x_size(self) -> float:
        return self.x_lim.max_val - self.x_lim.min_val

    @property
    def extent_y_size(self) -> float:
        return self.y_lim.max_val - self.y_lim.min_val


class Dataset:
    def __init__(self, path: str, columns: ColumnIndex, delimiter: str,
                 skip_rows_count: int):
        self.__src_data = np.loadtxt(
            path, skiprows=skip_rows_count, delimiter=delimiter,
            usecols=[columns.time, columns.x, columns.y, columns.z,
                     columns.density]
        )
        self.__clear_data = self.get_clear_data()

    @property
    def src_data(self) -> np.ndarray:
        return self.__src_data

    @property
    def clear_data(self):
        return self.__clear_data

    def get_clear_data(self) -> np.ndarray:
        hash_vals = set()
        clear_array = np.zeros(shape=(0, len(ColumnIndex)))
        for time, x, y, z, dens in self.src_data:
            current_hash = hash((time, x, y, z))
            if current_hash not in hash_vals:
                clear_array = np.vstack((clear_array, [time, x, y, z, dens]))
                hash_vals.add(current_hash)
        return clear_array

    def get_selection(self, params: InterpolationParameter) -> np.ndarray:
        arr = self.clear_data
        if not arr.shape[0]:
            return np.array([])

        arr = arr[(arr[ColumnIndex.time] >= params.time_lim.min_val) *
                  (arr[ColumnIndex.time] < params.time_lim.max_val)]

        arr = arr[(arr[ColumnIndex.x] >= params.x_lim.min_val) *
                        (arr[ColumnIndex.x] < params.x_lim.max_val)]

        arr = arr[(arr[ColumnIndex.y] >= params.y_lim.min_val) *
                  (arr[ColumnIndex.y] < params.y_lim.max_val)]
        return arr

    @staticmethod
    def get_random_points_count(params: InterpolationParameter):
        extent_area = params.extent_x_size * params.extent_y_size
        return int(extent_area / params.cell_size ** 2)
