from typing import Callable, Iterator, List, Tuple

import numpy as np
from numpy.typing import NDArray

from tensorslow.tensor import Tensor

DifferentiableFunction = Callable[..., Tensor]
DifferentiableFunctionOnNP = Callable[..., NDArray]
X0 = List[Tensor]
Derivatives = List[NDArray]
DerivativeApproximator = Callable[[DifferentiableFunction, X0], Derivatives]


class CentredDifferenceApproximator:
    h = 1e-8

    def __init__(self) -> None:
        self.f: DifferentiableFunctionOnNP
        self.x_0: X0
        self.x_0_as_numpy: List[NDArray]

    def set_function(self, f: DifferentiableFunction, x_0: X0) -> None:
        self.f = CentredDifferenceApproximator._modify_for_numpy(f)
        self.x_0 = x_0
        self.x_0_as_numpy = [x.value for x in self.x_0]

    def centred_difference(self) -> Derivatives:
        derivatives = []

        # convert to arrays
        for x in self.x_0_as_numpy:
            df_dx = self._calc_df_dx(x)
            derivatives.append(df_dx)
        return derivatives

    @staticmethod
    def _iterate_multi_index(x: NDArray) -> Iterator[Tuple[np.signedinteger, ...]]:
        yield from [np.unravel_index(k, x.shape) for k in range(x.size)]

    @staticmethod
    def _modify_for_numpy(f: DifferentiableFunction) -> DifferentiableFunctionOnNP:
        def f_on_np(*args: NDArray) -> NDArray:
            return f(*[Tensor(x) for x in args]).value
        return f_on_np

    def _call_f_x_modified(
        self, x: NDArray, indice: Tuple[np.signedinteger, ...], update: NDArray
    ) -> NDArray:
        temp = x[indice]
        x[indice] = update[indice]
        f_x_p = self.f(*self.x_0_as_numpy)
        x[indice] = temp
        return f_x_p

    def _calc_df_dx(self, x: NDArray) -> NDArray:
        df_dx: List[NDArray] = []
        x_m = x - CentredDifferenceApproximator.h
        x_p = x + CentredDifferenceApproximator.h

        for indice in CentredDifferenceApproximator._iterate_multi_index(x):
            # get additive part
            f_x_p = self._call_f_x_modified(x, indice, x_p)

            # get subtractive part
            f_x_m = self._call_f_x_modified(x, indice, x_m)

            # calc derivative
            derivative = (f_x_p - f_x_m) / (2 * CentredDifferenceApproximator.h)
            df_dx.append(derivative)
            out_shape = derivative.shape

        return np.reshape(np.stack(df_dx), x.shape + out_shape)
