# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import saps

xp = saps.xp

#poetry run asv run --python=same -v --set-commit-hash $(git rev-parse HEAD) --record-samples

class Foo:
    @property
    def params(self):
        return ([xp], self.data_generators)
    
    param_names = ["xp", "size"]

class TimeSuite(Foo):
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    @property
    def data_generators(self):
        return [0, 1, 100]

    def setup(self, xp, s):
        self.d = {}
        for x in range(s):
            self.d[x] = None

    def time_range(self, xp, s):
        d = self.d
        for key in range(s):
            d[key]


class MemSuite:
    def mem_list(self):
        return [0] * 256
