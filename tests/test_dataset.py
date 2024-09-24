import unittest
import numpy as np
from lkdata.dataset import DataProcessorMixin, DataProducts, ErrorProducts, DataSet
from lkdata.datacube import DataCube, ErrorCube
from lkdata.dataframe import DataFrame, ErrorFrame
from lkdata.dataseries import DataSeries, ErrorSeries


class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessorMixin()
        self.processor._type = "data"
        self.processor.kwargs = {}

    def test_process_input_list(self):
        data = [[1, 2], [3, 4]]
        result = self.processor.process_input(data)
        self.assertIsInstance(result, DataFrame)

    def test_process_input_numpy_array(self):
        data = np.array([[1, 2], [3, 4]])
        result = self.processor.process_input(data)
        self.assertIsInstance(result, DataFrame)

    def test_process_input_data_cube(self):
        data = DataCube(np.array([[[1, 2], [3, 4]]]))
        result = self.processor.process_input(data)
        self.assertIsInstance(result, DataCube)

    def test_process_input_invalid_type(self):
        with self.assertRaises(TypeError):
            self.processor.process_input("invalid")

    def test_check_attrs(self):
        data = DataCube(np.array([[[1, 2], [3, 4]]]))
        self.processor.ntime = 1
        self.processor.nrow = 2
        self.processor.ncol = 2
        self.processor._check_attrs(data)

    def test_check_attrs_mismatch(self):
        data = DataCube(np.array([[[1, 2], [3, 4]]]))
        self.processor.ntime = 2
        with self.assertRaises(ValueError):
            self.processor._check_attrs(data)


class TestDataProducts(unittest.TestCase):
    def setUp(self):
        self.data = np.array([[1, 2], [3, 4]])

    def test_init_with_array(self):
        dp = DataProducts(self.data)
        self.assertIn("flux_data", dp)
        self.assertIsInstance(dp["flux_data"], DataFrame)

    def test_init_with_dict(self):
        dp = DataProducts({"test": self.data})
        self.assertIn("test", dp)
        self.assertIsInstance(dp["test"], DataFrame)

    def test_update(self):
        dp = DataProducts({"test1": self.data})
        dp.update({"test2": self.data})
        self.assertIn("test1", dp)
        self.assertIn("test2", dp)

    def test_getitem_slice(self):
        dp = DataProducts({"test": self.data})
        result = dp[0:1]
        self.assertIsInstance(result["test"], DataFrame)
        np.testing.assert_array_equal(result["test"], self.data[0:1])


class TestErrorProducts(unittest.TestCase):
    def setUp(self):
        self.error = np.array([[0.1, 0.2], [0.3, 0.4]])

    def test_init_with_array(self):
        ep = ErrorProducts(self.error)
        self.assertIn("flux_error", ep)
        self.assertIsInstance(ep["flux_error"], ErrorFrame)

    def test_init_with_dict(self):
        ep = ErrorProducts({"test": self.error})
        self.assertIn("test", ep)
        self.assertIsInstance(ep["test"], ErrorFrame)


class TestBatch(unittest.TestCase):
    def setUp(self):
        self.data = np.array([[1, 2], [3, 4]])
        self.error = np.array([[0.1, 0.2], [0.3, 0.4]])

    def test_init(self):
        batch = DataSet(self.data, self.error)
        self.assertIsInstance(batch.data, DataProducts)
        self.assertIsInstance(batch.error, ErrorProducts)

    def test_cubes_property(self):
        cube_data = np.array([[[1, 2], [3, 4]]])
        cube_error = np.array([[[0.1, 0.2], [0.3, 0.4]]])
        batch = DataSet(
            {"data_cube": DataCube(cube_data)}, {"error_cube": ErrorCube(cube_error)}
        )
        cubes = batch.cubes
        self.assertIn("data_cube", cubes)
        self.assertIn("error_cube", cubes)
        self.assertIsInstance(cubes["data_cube"], DataCube)
        self.assertIsInstance(cubes["error_cube"], ErrorCube)

    def test_frames_property(self):
        batch = DataSet(
            {"data_frame": DataFrame(self.data)},
            {"error_frame": ErrorFrame(self.error)},
        )
        frames = batch.frames
        self.assertIn("data_frame", frames)
        self.assertIn("error_frame", frames)
        self.assertIsInstance(frames["data_frame"], DataFrame)
        self.assertIsInstance(frames["error_frame"], ErrorFrame)

    def test_series_property(self):
        series_data = np.array([1, 2, 3, 4])
        series_error = np.array([0.1, 0.2, 0.3, 0.4])
        batch = DataSet(
            {"data_series": DataSeries(series_data)},
            {"error_series": ErrorSeries(series_error)},
        )
        series = batch.series
        self.assertIn("data_series", series)
        self.assertIn("error_series", series)
        self.assertIsInstance(series["data_series"], DataSeries)
        self.assertIsInstance(series["error_series"], ErrorSeries)

    def test_getitem_string(self):
        batch = DataSet({"data": self.data}, {"error": self.error})
        self.assertIsInstance(batch["data"], DataFrame)
        self.assertIsInstance(batch["error"], ErrorFrame)

    # def test_getitem_slice(self):
    #     batch = DataSet({"data": self.data}, {"error": self.error})
    #     sliced_batch = batch[0:1]
    #     self.assertIsInstance(sliced_batch, DataSet)
    #     np.testing.assert_array_equal(sliced_batch.data["data"], self.data[0:1])
    #     np.testing.assert_array_equal(sliced_batch.error["error"], self.error[0:1])

    def test_getitem_invalid_key(self):
        batch = DataSet({"data": self.data}, {"error": self.error})
        with self.assertRaises(ValueError):
            batch["invalid_key"]

    def test_repr(self):
        batch = DataSet({"data": self.data}, {"error": self.error})
        repr_str = repr(batch)
        self.assertIn("data", repr_str)
        self.assertIn("error", repr_str)
