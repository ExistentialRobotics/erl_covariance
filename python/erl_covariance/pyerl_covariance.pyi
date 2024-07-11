from enum import IntEnum
from typing import overload

import numpy as np
import numpy.typing as npt
from erl_common.yaml import YamlableBase

__all__ = [
    "Covariance",
    "OrnsteinUhlenbeck_1D",
    "OrnsteinUhlenbeck_2D",
    "OrnsteinUhlenbeck_3D",
    "OrnsteinUhlenbeck_xD",
    "Matern32_1D",
    "Matern32_2D",
    "Matern32_3D",
    "Matern32_xD",
    "RadialBiasFunction_1D",
    "RadialBiasFunction_2D",
    "RadialBiasFunction_3D",
    "RadialBiasFunction_xD",
    "RationalQuadratic_1D",
    "RationalQuadratic_2D",
    "RationalQuadratic_3D",
    "RationalQuadratic_xD",
    "CustomKernelV1",
    "CustomKernelV2",
    "CustomKernelV3",
    "CustomKernelV4",
]

class Covariance:
    class Setting(YamlableBase):
        x_dim: int
        alpha: float
        scale: float
        scale_mix: float
        weights: npt.NDArray[np.float64]

        def __init__(self: Covariance.Setting): ...

    def __init__(self: Covariance, setting: Covariance.Setting): ...
    @property
    def setting(self: Covariance) -> Setting: ...
    @overload
    def compute_ktrain(
        self: Covariance, mat_x: npt.NDArray[np.float64], num_samples: int
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def compute_ktrain(
        self: Covariance, mat_x: npt.NDArray[np.float64], vec_var_y: npt.NDArray[np.float64], num_samples: int
    ) -> npt.NDArray[np.float64]: ...
    def compute_ktest(
        self: Covariance,
        mat_x1: npt.NDArray[np.float64],
        num_samples1: int,
        mat_x2: npt.NDArray[np.float64],
        num_samples2: int,
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def compute_ktrain_with_gradient(
        self: Covariance, mat_x: npt.NDArray[np.float64], num_samples: int, vec_grad_flags: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def compute_ktrain_with_gradient(
        self: Covariance,
        mat_x: npt.NDArray[np.float64],
        num_samples: int,
        vec_grad_flags: npt.NDArray[np.bool_],
        vec_var_x: npt.NDArray[np.float64],
        vec_var_y: npt.NDArray[np.float64],
        vec_var_grad: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]: ...
    def compute_ktest_with_gradient(
        self: Covariance,
        mat_x1: npt.NDArray[np.float64],
        num_samples1: int,
        vec_grad1_flags: npt.NDArray[np.bool_],
        mat_x2: npt.NDArray[np.float64],
        num_samples2: int,
    ) -> npt.NDArray[np.float64]: ...

class OrnsteinUhlenbeck_1D(Covariance):
    def __init__(self: OrnsteinUhlenbeck_1D, setting: Covariance.Setting = None): ...

class OrnsteinUhlenbeck_2D(Covariance):
    def __init__(self: OrnsteinUhlenbeck_2D, setting: Covariance.Setting = None): ...

class OrnsteinUhlenbeck_3D(Covariance):
    def __init__(self: OrnsteinUhlenbeck_3D, setting: Covariance.Setting = None): ...

class OrnsteinUhlenbeck_xD(Covariance):
    def __init__(self: OrnsteinUhlenbeck_xD, setting: Covariance.Setting = None): ...

class Matern32_1D(Covariance):
    def __init__(self: Matern32_1D, setting: Covariance.Setting = None): ...

class Matern32_2D(Covariance):
    def __init__(self: Matern32_2D, setting: Covariance.Setting = None): ...

class Matern32_3D(Covariance):
    def __init__(self: Matern32_3D, setting: Covariance.Setting = None): ...

class Matern32_xD(Covariance):
    def __init__(self: Matern32_xD, setting: Covariance.Setting = None): ...

class RadialBiasFunction_1D(Covariance):
    def __init__(self: RadialBiasFunction_1D, setting: Covariance.Setting = None): ...

class RadialBiasFunction_2D(Covariance):
    def __init__(self: RadialBiasFunction_2D, setting: Covariance.Setting = None): ...

class RadialBiasFunction_3D(Covariance):
    def __init__(self: RadialBiasFunction_3D, setting: Covariance.Setting = None): ...

class RadialBiasFunction_xD(Covariance):
    def __init__(self: RadialBiasFunction_xD, setting: Covariance.Setting = None): ...

class RationalQuadratic_1D(Covariance):
    def __init__(self: RationalQuadratic_1D, setting: Covariance.Setting = None): ...

class RationalQuadratic_2D(Covariance):
    def __init__(self: RationalQuadratic_2D, setting: Covariance.Setting = None): ...

class RationalQuadratic_3D(Covariance):
    def __init__(self: RationalQuadratic_3D, setting: Covariance.Setting = None): ...

class RationalQuadratic_xD(Covariance):
    def __init__(self: RationalQuadratic_xD, setting: Covariance.Setting = None): ...

class CustomKernelV1(Covariance):
    def __init__(self: CustomKernelV1, setting: Covariance.Setting = None): ...

class CustomKernelV2(Covariance):
    def __init__(self: CustomKernelV2, setting: Covariance.Setting = None): ...

class CustomKernelV3(Covariance):
    def __init__(self: CustomKernelV3, setting: Covariance.Setting = None): ...

class CustomKernelV4(Covariance):
    def __init__(self: CustomKernelV4, setting: Covariance.Setting = None): ...
