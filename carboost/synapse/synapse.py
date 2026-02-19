from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .synapse_utils import (
    get_conv,
    get_conv2,
    get_estimators,
    get_estimators_delta,
    get_impulse,
    get_regions_probabilities,
    integ,
)


class SynapseBase:
    """
    A Base class to contain information to calculate synapse distribution. The details of the variables are given below.

    car_name: name of the CAR receptor used (for book keeping)
    hinge_length: hinge sequence length  (for book keeping)
    target_type: ["dynamic","rigid"] | required for the synapse calculation
    target_kde: KDE density for target receptor
    car_kdes: KDE density for the CAR receptor
    method_name: ["af2rave","rMSA_AF2"] | for book keeping
    x_car: the support array for the CAR KDE density
    x_target: the support array for the CAR KDE density
    max_val_car: max value of the CAR hinge (extended form, required for probability calculation)
    max_val_target: max value of the target (extended form, required for probability calculation)
    offset_value: Offset values coming from scFv or the ordered domain of target
    """

    def __init__(
        self,
        car_name: str,
        hinge_length: str | int,
        target_type: str,
        target_kde: np.ndarray | None,
        car_kdes: np.ndarray | Sequence[np.ndarray],
        method_name: str,
        x_car: np.ndarray,
        x_target: np.ndarray | None,
        max_val_car: float,
        max_val_target: float | None,
        offset_value: float):

        self.car_name = car_name
        self.hinge_length = str(hinge_length)
        self.target_type = target_type.lower()
        self.method_name = method_name
        self.offset_value = float(offset_value)
        self.max_val_car = float(max_val_car)
        self.max_val_target = (
            None if max_val_target is None else float(max_val_target)
        )
        self.x_car = np.asarray(x_car, dtype=float)
        self.x_target = None if x_target is None else np.asarray(x_target, dtype=float)
        self.target_kde = None if target_kde is None else np.asarray(target_kde, dtype=float)
        self.car_kdes = self._standardize_car_kdes(car_kdes)

        if self.target_type == "dynamic":
            if self.max_val_target is None:
                raise ValueError(
                    "Dynamic target_type requires max_val_target."
                )
            self.max_val = self.max_val_car + self.max_val_target + self.offset_value
        elif self.target_type == "rigid":
            if self.max_val_target is not None:
                raise ValueError("Rigid target_type requires max_val_target to be None.")
            self.max_val = self.max_val_car + self.offset_value
        else:
            raise ValueError("target_type must be either 'dynamic' or 'rigid'.")

        self.region_split = [0.0, 12.5, 14.5, self.max_val]
        self._validate_inputs()

    @staticmethod
    def _standardize_car_kdes(car_kdes: np.ndarray | Sequence[np.ndarray]):
        if isinstance(car_kdes, np.ndarray):
            if car_kdes.ndim == 1:
                return [car_kdes.astype(float)]
            if car_kdes.ndim == 2:
                return [row.astype(float) for row in car_kdes]
            raise ValueError("car_kdes numpy array must be 1D or 2D.")
        return [np.asarray(kde, dtype=float) for kde in car_kdes]

    def _validate_inputs(self):
        if self.target_type not in {"dynamic", "rigid"}:
            raise ValueError("target_type must be either 'dynamic' or 'rigid'.")
        if self.x_car.ndim != 1:
            raise ValueError("x_car must be a 1D array.")
        if len(self.car_kdes) == 0:
            raise ValueError("car_kdes cannot be empty.")

        if self.target_type == "dynamic":
            if self.target_kde is None or self.x_target is None:
                raise ValueError("Dynamic target_type requires target_kde and x_target.")
            if self.x_target.ndim != 1:
                raise ValueError("x_target must be a 1D array for dynamic target_type.")
            if self.target_kde.ndim != 1:
                raise ValueError("target_kde must be a 1D array for dynamic target_type.")
            if self.target_kde.shape[0] != self.x_target.shape[0]:
                raise ValueError("target_kde and x_target must have the same length.")
        else:
            if self.target_kde is not None:
                raise ValueError("Rigid target_type does not accept target_kde.")

        for i, car_kde in enumerate(self.car_kdes):
            if car_kde.ndim != 1:
                raise ValueError(f"car_kdes[{i}] must be a 1D array.")
            if car_kde.shape[0] != self.x_car.shape[0]:
                raise ValueError(f"car_kdes[{i}] and x_car must have the same length.")


class IMDistanceResult:
    """A simple container to access intermembrane distance distribution"""

    def __init__(self,car_name, im_distance_kdes, x_im):
        self.car_name = car_name 
        self.im_distance_kdes = im_distance_kdes
        self.x_im = x_im


class IMDistanceCalculator:
    """Inter membrane distance density calculator using SynapseBase class."""

    def __init__(self, synapse: SynapseBase):
        self.synapse = synapse

    def compute(self):
        x_car = self.synapse.x_car
        x_target = self.synapse.x_target 
        im_distance_kdes: list[np.ndarray] = []
        x_im: np.ndarray | None = None

        if self.synapse.target_type == "dynamic":
            # # Resample onto x_car so convolution operates on a single consistent grid.
            # # get_conv/get_conv2 do not perform general unequal-grid convolution.
            # target_resampled = np.interp(
            #     x_ref,
            #     self.synapse.x_target,
            #     self.synapse.target_kde,
            #     left=0.0,
            #     right=0.0,
            # )
            # target_resampled /= integ(target_resampled, x_ref)
            offset_impulse = get_impulse(x_car, self.synapse.offset_value)
        else:
            target_impulse = get_impulse(x_car, self.synapse.offset_value)

        for car_kde in self.synapse.car_kdes:
            car_norm = np.asarray(car_kde, dtype=float)
            car_norm /= integ(car_norm, x_car)
            target_norm = np.asarray(self.synapse.target_kde, dtype=float)
            target_norm /= integ(target_norm, x_target) 
            if self.synapse.target_type == "dynamic":
                y_conv, x_conv = get_conv2(target_norm, car_norm, x_car, x_target)
                y_conv /= integ(y_conv, x_conv)
                y_shifted, x_shifted = get_conv2(y_conv, offset_impulse, x_conv, x_car)
                y_shifted /= integ(y_shifted, x_shifted)
                im_distance_kdes.append(y_shifted)
                x_im = x_shifted
            else:
                y_conv, x_conv = get_conv(target_impulse, car_norm, x_car, x_car.shape[0])
                y_conv /= integ(y_conv, x_conv)
                im_distance_kdes.append(y_conv)
                x_im = x_conv

        if x_im is None:
            raise ValueError("No CAR KDE values available for IM-distance computation.")

        return IMDistanceResult(car_name=self.synapse.car_name,im_distance_kdes=im_distance_kdes, x_im=x_im)


class PhiValueCalculator:
    """Phi value estimator that uses SynapseBase and IMDistanceResult classes."""

    def __init__(self, synapse: SynapseBase, im_result: IMDistanceResult):
        self.synapse = synapse
        self.im_result = im_result

    def compute(self):
        region_probabilities = [
            get_regions_probabilities(prob, self.synapse.region_split, self.im_result.x_im)
            for prob in self.im_result.im_distance_kdes
        ]
        probabilities_region = np.array(region_probabilities, dtype=float)
        probabilities_avg = np.mean(probabilities_region, axis=0)
        probabilities_delta = np.std(probabilities_region, axis=0) / np.sqrt(
            len(probabilities_region)
        )

        dG1, dG2, phi = get_estimators(probabilities_avg.tolist())
        dG1_err, dG2_err, phi_err = get_estimators_delta(
            probabilities_avg.tolist(), probabilities_delta.tolist()
        )

        return {
            "car_name": self.synapse.car_name,
            "hinge_length": self.synapse.hinge_length,
            "target_type": self.synapse.target_type,
            "method_name": self.synapse.method_name,
            "phi": phi,
            "dG1": dG1,
            "dG2": dG2,
            "phi_err": phi_err,
            "dG1_err": dG1_err,
            "dG2_err": dG2_err,
            "probabilities_region": probabilities_region,
            "probabilities_avg": probabilities_avg,
            "probabilities_delta": probabilities_delta,
        }


def run_carboost_pipeline(synapse: SynapseBase):
    """
    End-to-end function to go through carboost calculations.

    1) Compute IM-distance KDEs from user-provided CAR/target KDEs.
    2) Compute phi values from the IM-distance densities.
    """

    im_calc = IMDistanceCalculator(synapse=synapse)
    im_result = im_calc.compute()

    phi_calc = PhiValueCalculator(synapse=synapse, im_result=im_result)
    return phi_calc.compute()
