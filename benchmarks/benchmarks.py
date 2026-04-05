# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

class Foo:
    params = [1, 10, 100]
    param_names = ["size"]

class TimeSuite(Foo):
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    def setup(self, s):
        self.d = {}
        for x in range(s):
            self.d[x] = None

    def time_range(self, s):
        d = self.d
        for key in range(s):
            d[key]


class MemSuite:
    def mem_list(self):
        return [0] * 256
