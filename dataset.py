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
    x_lim: Limit
    y_lim: Limit
    time_lim: Limit
    cell_size: float
    buffer_distance: float
    

class Dataset:
    def __init__(self, path: str, columns: ColumnIndex, delimiter: str,
                 skip_rows_count: int):
        data = np.loadtxt(path, skiprows=skip_rows_count, delimiter=delimiter,
                          usecols=[columns.time, columns.x, columns.y,
                                   columns.z, columns.density])
        self.__src_data = data

    @property
    def src_data(self) -> np.ndarray:
        return self.__src_data

    def get_clear_data(self) -> np.ndarray:
        hash_val = set()
        clear_array = np.zeros(shape=(0, len(ColumnIndex)))
        for time, x, y, z, dens in self.src_data:
            if (time, x, y, z) not in hash_val:
                clear_array = np.vstack((clear_array, [time, x, y, z, dens]))
        return clear_array


