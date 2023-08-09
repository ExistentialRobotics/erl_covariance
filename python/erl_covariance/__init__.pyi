from enum import IntEnum
from typing import overload

import numpy as np
import numpy.typing as npt
from erl_common.yaml import YamlableBase

__all__ = [
    "Covariance",
    "OrnsteinUhlenbeck",
    "Matern32",
    "RadialBiasFunction",
    "RationalQuadratic",
    "CustomKernelV1",
    "CustomKernelV2",
    "CustomKernelV3",
    "CustomKernelV4",
]

class Covariance:
    class Type(IntEnum):
        kOrnsteinUhlenbeck = 0
        kMatern32 = 1
        kRadialBiasFunction = 2
        kRationalQuadratic = 3
        kCustomKernelV1 = 4
        kCustomKernelV2 = 5
        kCustomKernelV3 = 6
        kCustomKernelV4 = 7
        kUnknown = 8

    class Setting(YamlableBase):
        type: Covariance.Type
        alpha: float
        scale: float
        parallel: bool
        scale_mix: float
        weights: npt.NDArray[np.float64]

        @overload
        def __init__(self: Covariance.Setting): ...
        @overload
        def __init__(self: Covariance.Setting, type_: Covariance.Type): ...

    def __init__(self: Covariance, setting: Covariance.Setting): ...
    @property
    def setting(self: Covariance) -> Setting: ...
    @overload
    def compute_ktrain(self: Covariance, mat_x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    @overload
    def compute_ktrain(
        self: Covariance, mat_x: npt.NDArray[np.float64], vec_sigma_y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def compute_ktest(
        self: Covariance, mat_x1: npt.NDArray[np.float64], mat_x2: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def compute_ktrain_with_gradient(
        self: Covariance, mat_x: npt.NDArray[np.float64], vec_grad_flags: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def compute_ktrain_with_gradient(
        self: Covariance,
        mat_x: npt.NDArray[np.float64],
        vec_grad_flags: npt.NDArray[np.bool_],
        vec_sigma_x: npt.NDArray[np.float64],
        vec_sigma_y: npt.NDArray[np.float64],
        vec_sigma_grad: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]: ...
    def compute_ktest_with_gradient(
        self: Covariance,
        mat_x1: npt.NDArray[np.float64],
        vec_grad_flags: npt.NDArray[np.bool_],
        mat_x2: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]: ...

class OrnsteinUhlenbeck(Covariance):
    def __init__(self: OrnsteinUhlenbeck, setting: Covariance.Setting = None): ...

class Matern32(Covariance):
    def __init__(self: Matern32, setting: Covariance.Setting = None): ...

class RadialBiasFunction(Covariance):
    def __init__(self: RadialBiasFunction, setting: Covariance.Setting = None): ...

class RationalQuadratic(Covariance):
    def __init__(self: RationalQuadratic, setting: Covariance.Setting = None): ...

class CustomKernelV1(Covariance):
    def __init__(self: CustomKernelV1, setting: Covariance.Setting = None): ...

class CustomKernelV2(Covariance):
    def __init__(self: CustomKernelV2, setting: Covariance.Setting = None): ...

class CustomKernelV3(Covariance):
    def __init__(self: CustomKernelV3, setting: Covariance.Setting = None): ...

class CustomKernelV4(Covariance):
    def __init__(self: CustomKernelV4, setting: Covariance.Setting = None): ...
