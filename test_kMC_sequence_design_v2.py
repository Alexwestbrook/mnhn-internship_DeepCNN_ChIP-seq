import kMC_sequence_design_v2 as kmc_v2
import numpy as np
import pytest


class TestMaxstepLinspace:
    def test_examples(self):
        # all spacings equal
        assert np.all(kmc_v2.maxstep_linspace(0, 8, 2) == np.array([0, 2, 4, 6, 8]))
        # uneven adjusted spacings
        assert np.all(kmc_v2.maxstep_linspace(0, -8, 3) == np.array([0, -3, -5, -8]))
        # spacings much smaller than max_step
        assert np.all(kmc_v2.maxstep_linspace(-4.0, 4.0, 7.0) == np.array([-4, 0, 4]))
        # start same as stop
        assert np.all(kmc_v2.maxstep_linspace(0, 0, 2) == np.array([0]))

    def test_exceptions(self):
        with pytest.raises(ValueError):
            # negative max_step
            kmc_v2.maxstep_linspace(0, 8, -2)
            # non integers
            kmc_v2.maxstep_linspace(0, 8.2, 2)


class TestSlicerOnAxis:
    def test_examples(self):
        arr = np.arange(24).reshape(2, 3, 4)
        # simple slicer
        assert np.all(
            arr[kmc_v2.slicer_on_axis(arr, slice(1, 3), axis=-1)]
            == np.array([[[1, 2], [5, 6], [9, 10]], [[13, 14], [17, 18], [21, 22]]])
        )
        # unknown axis parameter
        for axis, res in enumerate([arr[1:], arr[:, 1:], arr[:, :, 1:]]):
            assert np.all(
                arr[kmc_v2.slicer_on_axis(arr, slice(1, None), axis=axis)] == res
            )
        # no axis parameter
        assert np.all(
            arr[kmc_v2.slicer_on_axis(arr, slice(None, -1))]
            == np.array([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]])
        )
        # multiple slices and axis
        assert np.all(
            arr[kmc_v2.slicer_on_axis(arr, [slice(None, -1), slice(1, 3)], axis=[0, 2])]
            == np.array([[[1, 2], [5, 6], [9, 10]]])
        )
        # multiple slices without axis parameter
        assert np.all(
            arr[kmc_v2.slicer_on_axis(arr, [slice(None, -1), slice(1, 3)])]
            == np.array([[[4, 5, 6, 7], [8, 9, 10, 11]]])
        )
        # single slice on multiple axis
        assert np.all(
            arr[kmc_v2.slicer_on_axis(arr, slice(1, None), axis=[1, 2])]
            == np.array([[[5, 6, 7], [9, 10, 11]], [[17, 18, 19], [21, 22, 23]]])
        )

    def test_exceptions(self):
        arr = np.arange(24).reshape(2, 3, 4)
        with pytest.raises(IndexError):
            # axis out of bounds
            kmc_v2.slicer_on_axis(arr, slice(1, -1), axis=4)
            kmc_v2.slicer_on_axis(arr, slice(1, -1), axis=-4)
        with pytest.raises(ValueError):
            # iterable slice, integer axis
            kmc_v2.slicer_on_axis(arr, [slice(1, -1)], axis=1)
            # different number of slices and axis
            kmc_v2.slicer_on_axis(arr, [slice(1, -1), slice(2, 3)], axis=[1, 0, 2])
            kmc_v2.slicer_on_axis(arr, [slice(1, -1), slice(2, 3)], axis=[1])
            # multiple references to same axis
            kmc_v2.slicer_on_axis(arr, [slice(1, -1), slice(2, 3)], axis=[1, 1])
            kmc_v2.slicer_on_axis(arr, [slice(1, -1), slice(2, 3)], axis=[1, -2])


class TestMovingSum:
    tokens = np.array([0, 3, 1, 0, 3, 3, 3, 3, 1, 3])
    one_hot = kmc_v2.np_idx_to_one_hot(tokens)

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
        with pytest.raises(IndexError):
            # axis out of bounds
            kmc_v2.moving_sum(self.one_hot, 2, axis=2)
            kmc_v2.moving_sum(self.one_hot, 2, axis=-2)
        with pytest.raises(ValueError):
            # n negative or 0
            kmc_v2.moving_sum(self.one_hot, 0, axis=0)
            kmc_v2.moving_sum(self.one_hot, -1, axis=0)
            # n too big
            kmc_v2.moving_sum(self.one_hot[:, 0], 11)
            kmc_v2.moving_sum(self.one_hot, 5, axis=-1)
            kmc_v2.moving_sum(self.one_hot, 41)


class TestSlidingGC:
    tokens1 = np.array([[3, 2, 2, 1, 1, 0, 0, 0, 0, 3]])
    tokens2 = np.array(
        [
            [3, 2, 2, 1, 1, 0],
            [0, 0, 0, 3, 2, 3],
            [2, 2, 3, 2, 2, 2],
            [2, 3, 1, 3, 2, 0],
            [1, 3, 2, 0, 3, 2],
        ]
    )
    tokens = [tokens1, tokens2]
    one_hots = [kmc_v2.np_idx_to_one_hot(token) for token in tokens]

    def test_examples(self):
        ref1 = np.array([[0.75, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0]])
        ref2 = np.array(
            [
                [0.75, 1.0, 0.75],
                [0.0, 0.25, 0.25],
                [0.75, 0.75, 0.75],
                [0.5, 0.5, 0.5],
                [0.5, 0.25, 0.5],
            ]
        )
        refs = [ref1, ref2]
        for i, ref in enumerate(refs):
            assert np.all(kmc_v2.sliding_GC(self.tokens[i], 4, form="token") == ref)
            assert np.all(kmc_v2.sliding_GC(self.one_hots[i], 4, form="one_hot") == ref)
