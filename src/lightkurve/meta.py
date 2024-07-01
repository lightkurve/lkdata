import numpy as np


class CubeMeta(dict):
    """A meta dictionary created from a datacube with a custom repr

    This meta information is collected from the datacube itself and should not
    be directly modified.
    """

    def __init__(self, cube):
        super().__init__()
        self.cube = cube
        for time_index in cube.index.names:
            self[time_index] = cube.__getattr__(time_index)
        for loc in cube.columns.names:
            if loc != "series":
                self[loc] = np.unique(cube.__getattr__(loc))
        for key in cube._metadata:
            if (key not in cube.index.names + cube.columns.names) and (key[0] != "_"):
                self[key] = getattr(cube, key)

    def __repr__(self):
        out = "\nAttributes accessible via `object.key`\n"
        out += "(displaying only unique values)\n"
        max_name_len = max(map(len, self.keys()))
        with np.printoptions(linewidth=79, edgeitems=2, threshold=20):
            for key in self:
                out += f"\t{key.ljust(max_name_len+1)}:\t{self[key]}\n"
            return out

    def __eq__(self, other):
        if self.keys() != other.keys():
            return False
        eq = []
        for key in self.keys():
            try:
                eq.append(all(self[key] == other[key]))
            except TypeError:
                eq.append(self[key] == other[key])
        return eq
