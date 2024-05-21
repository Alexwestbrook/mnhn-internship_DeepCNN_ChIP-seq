import itertools as it

import kMC_sequence_design_v2 as kmc_v2
import numpy as np
import pandas as pd
import pytest


class TestMaxstepLinspace:
    def test_examples(self):
        # all spacings equal
        assert np.all(kmc_v2.maxstep_linspace(0, 8, 2) == np.array([0, 2, 4, 6, 8]))
        # uneven adjusted spacings
        assert np.all(kmc_v2.maxstep_linspace(0, 8, 3) == np.array([0, 3, 5, 8]))
        # spacings much smaller than max_step
        assert np.all(kmc_v2.maxstep_linspace(0, 8, 7) == np.array([0, 4, 8]))
        # start same as stop
        assert np.all(kmc_v2.maxstep_linspace(0, 0, 2) == np.array([0]))


class TestSlicerOnAxis:
    def test_examples(self):
        arr = np.arange(24).reshape(2, 3, 4)
        assert np.all(
            arr[kmc_v2.slicer_on_axis(arr, slice(1, 3), axis=2)]
            == np.array([[[1, 2], [5, 6], [9, 10]], [[13, 14], [17, 18], [21, 22]]])
        )
        assert np.all(
            arr[kmc_v2.slicer_on_axis(arr, slice(2, None), axis=1)]
            == np.array([[[8, 9, 10, 11]], [[20, 21, 22, 23]]])
        )
        assert np.all(
            arr[kmc_v2.slicer_on_axis(arr, slice(None, -1))]
            == np.array([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]])
        )
        assert np.all(
            arr[kmc_v2.slicer_on_axis(arr, [slice(None, -1), slice(1, 3)], axis=[0, 2])]
            == np.array([[[1, 2], [5, 6], [9, 10]]])
        )
        assert np.all(
            arr[kmc_v2.slicer_on_axis(arr, [slice(None, -1), slice(1, 3)])]
            == np.array([[[4, 5, 6, 7], [8, 9, 10, 11]]])
        )


class TestMovingSum:
    tokens = np.array([0, 3, 1, 0, 3, 3, 3, 3, 1, 3])
    one_hot = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=int,
    )

    def test_examples(self):
        assert np.all(
            kmc_v2.moving_sum(np.arange(10), n=2)
            == np.array([1, 3, 5, 7, 9, 11, 13, 15, 17])
        )
        arr = np.arange(24).reshape(2, 3, 4)
        assert np.all(
            kmc_v2.moving_sum(arr, n=2, axis=-1)
            == np.array(
                [
                    [[1, 3, 5], [9, 11, 13], [17, 19, 21]],
                    [[25, 27, 29], [33, 35, 37], [41, 43, 45]],
                ]
            )
        )

    def test_1D(self):
        refs = [
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ]
            ),
            np.array(
                [
                    [1, 0, 0, 1],
                    [0, 1, 0, 1],
                    [1, 1, 0, 0],
                    [1, 0, 0, 1],
                    [0, 0, 0, 2],
                    [0, 0, 0, 2],
                    [0, 0, 0, 2],
                    [0, 1, 0, 1],
                    [0, 1, 0, 1],
                ]
            ),
            np.array(
                [
                    [1, 1, 0, 1],
                    [1, 1, 0, 1],
                    [1, 1, 0, 1],
                    [1, 0, 0, 2],
                    [0, 0, 0, 3],
                    [0, 0, 0, 3],
                    [0, 1, 0, 2],
                    [0, 1, 0, 2],
                ]
            ),
            np.array([[2, 2, 0, 6]]),
        ]
        for n, ref in zip([1, 2, 3, 10], refs):
            assert np.all(
                kmc_v2.moving_sum(self.one_hot[:, 0], n) == ref[:, 0]
            )  # default on 1D
            for axis in [None, 0, -1]:
                assert np.all(
                    kmc_v2.moving_sum(self.one_hot[:, 0], n, axis=axis) == ref[:, 0]
                )  # all axis values on 1D
            assert np.all(
                kmc_v2.moving_sum(self.one_hot, n, axis=0) == ref
            )  # axis=0 on 2D

    def test_2D(self):
        refs = [
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ]
            ),
            np.array(
                [
                    [1, 0, 0],
                    [0, 0, 1],
                    [1, 1, 0],
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1],
                    [1, 1, 0],
                    [0, 0, 1],
                ]
            ),
            np.array(
                [
                    [1, 0],
                    [0, 1],
                    [1, 1],
                    [1, 0],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [1, 1],
                    [0, 1],
                ]
            ),
            np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]),
        ]
        for n, ref in zip(range(1, 5), refs):
            assert np.all(
                kmc_v2.moving_sum(self.one_hot, n, axis=-1) == ref
            )  # axis=-1 on 2D

    def test_flattened(self):
        refs = [
            np.array(
                [
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                ]
            ),
            np.array(
                [
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                ]
            ),
            np.array(
                [
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    2,
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                ]
            ),
            np.array([10]),
        ]
        for n, ref in zip([1, 2, 3, 40], refs):
            assert np.all(kmc_v2.moving_sum(self.one_hot, n) == ref)  # axis=None on 2D

    def test_exceptions(self):
        with pytest.raises(ValueError):
            kmc_v2.moving_sum(self.one_hot[:, 0], 11)
            kmc_v2.moving_sum(self.one_hot, 5, axis=-1)
            kmc_v2.moving_sum(self.one_hot, 41)
