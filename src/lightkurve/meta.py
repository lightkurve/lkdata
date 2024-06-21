import numpy as np


class BaseMeta:
    _meta = {}

    def __getitem__(self, key):
        return self._meta[key]

    def __setitem__(self, key, value):
        self._meta[key] = value


class CubeMeta(BaseMeta):
    def __init__(self, cube):
        self.cube = cube
        for time_index in cube.index.names:
            self._meta[time_index] = cube.__getattr__(time_index)
        for loc in cube.columns.names:
            if loc != "series":
                self._meta[loc] = np.unique(cube.__getattr__(loc))
        for key in cube._metadata:
            if key not in cube.index.names + cube.columns.names:
                self._meta[key] = getattr(cube, key)

    def __repr__(self):
        _meta = self._meta
        out = "\nDataCube Attributes, accessible via `object.key`\n"
        max_name_len = max(map(len, self._meta.keys()))
        with np.printoptions(linewidth=79, edgeitems=2, threshold=20):
            for key in self._meta:
                out += f"\t{key.ljust(max_name_len+1)}:\t{self._meta[key]}\n"
            return self.cube.__repr__() + "\n" + out
