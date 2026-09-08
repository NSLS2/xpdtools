"""Numerical analysis for X-ray and UV-Vis optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


@dataclass(frozen=True)
class PLClassification:
    """Classification and detected non-LED peaks for one PL spectrum."""

    is_good: bool
    peak_indices: NDArray[np.intp]
    peak_heights: NDArray[np.float64]
    peak_wavelength_nm: float


def _nearest_index(values: NDArray[np.float64], target: float) -> int:
    """Return the index whose value is nearest to ``target``."""
    return int(np.abs(values - target).argmin())


def classify_pl(
    wavelength: ArrayLike,
    intensity: ArrayLike,
    *,
    key_height: float = 2000,
    height: float = 30,
    distance: int = 30,
    c2_c3: bool = False,
    threshold: tuple[float, float, float] = (560, 100000, 200000),
    integration_bounds: tuple[float, float, float] = (340, 400, 800),
) -> PLClassification:
    """Classify a fluorescence spectrum using the production PL policy.

    Peaks below 400 nm are treated as LED emission and excluded. The optional
    integral policy compares the PL-band integral against the LED-band integral.
    """
    x = np.asarray(wavelength, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("wavelength and intensity must be equal-length 1D arrays")

    peaks, properties = find_peaks(y, height=height, distance=distance)
    pl_mask = x[peaks] >= 400
    pl_peaks = np.asarray(peaks[pl_mask], dtype=np.intp)
    pl_heights = np.asarray(properties["peak_heights"][pl_mask], dtype=float)
    if pl_peaks.size == 0:
        return PLClassification(False, pl_peaks, pl_heights, float("nan"))

    top = int(np.argmax(pl_heights))
    top_intensity = float(pl_heights[top])
    top_wavelength = float(x[pl_peaks[top]])
    is_good = top_intensity >= key_height

    if is_good and c2_c3:
        led_start, pl_start, pl_end = (
            _nearest_index(x, bound) for bound in integration_bounds
        )
        led_integral = float(integrate.simpson(y[led_start:pl_start]))
        pl_integral = float(integrate.simpson(y[pl_start:pl_end]))
        peak_difference = pl_integral - led_integral
        split_wavelength, low_threshold, high_threshold = threshold
        if top_wavelength < split_wavelength:
            is_good = peak_difference >= low_threshold
        elif top_wavelength > split_wavelength:
            is_good = peak_difference >= high_threshold

    return PLClassification(
        is_good,
        pl_peaks,
        pl_heights,
        top_wavelength,
    )


def _prepare_spectra(
    wavelength: ArrayLike,
    spectra: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return wavelength rows and a two-dimensional spectra array."""
    values = np.asarray(spectra, dtype=float)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("spectra must contain at least one one-dimensional event")

    wavelengths = np.asarray(wavelength, dtype=float)
    if wavelengths.ndim == 1:
        if wavelengths.shape[0] != values.shape[1]:
            raise ValueError("wavelength and spectra lengths do not match")
        wavelengths = np.broadcast_to(wavelengths, values.shape)
    elif wavelengths.ndim == 2:
        if wavelengths.shape != values.shape:
            raise ValueError("wavelength and spectra shapes do not match")
    else:
        raise ValueError("wavelength must be one- or two-dimensional")

    return wavelengths, np.nan_to_num(values, nan=0.0)


def _select_spectra(
    wavelength: ArrayLike,
    spectra: ArrayLike,
    wavelength_range: tuple[float, float],
    percent_range: tuple[float, float],
    *,
    weighted: bool,
) -> NDArray[np.float64]:
    """Select spectra by inclusive percentiles of a wavelength-window score."""
    low_wavelength, high_wavelength = wavelength_range
    low_percent, high_percent = percent_range
    if low_wavelength > high_wavelength:
        raise ValueError("wavelength_range must be ordered")
    if not 0 <= low_percent <= high_percent <= 100:
        raise ValueError("percent_range must be ordered within [0, 100]")

    wavelengths, values = _prepare_spectra(wavelength, spectra)
    scores = np.empty(values.shape[0], dtype=float)
    for index, (x_row, y_row) in enumerate(zip(wavelengths, values, strict=True)):
        mask = (x_row >= low_wavelength) & (x_row <= high_wavelength)
        if not np.any(mask):
            raise ValueError("wavelength_range contains no spectrum samples")
        window = y_row[mask]
        scores[index] = (
            float(np.mean(window * x_row[mask])) if weighted else float(np.max(window))
        )

    lower, upper = np.percentile(scores, percent_range)
    return values[(scores >= lower) & (scores <= upper)]


def select_pl_spectra(
    wavelength: ArrayLike,
    spectra: ArrayLike,
    *,
    wavelength_range: tuple[float, float] = (400, 800),
    percent_range: tuple[float, float] = (30, 100),
) -> NDArray[np.float64]:
    """Select PL events by peak intensity within the configured wavelength window."""
    return _select_spectra(
        wavelength,
        spectra,
        wavelength_range,
        percent_range,
        weighted=False,
    )


def select_absorbance_spectra(
    wavelength: ArrayLike,
    spectra: ArrayLike,
    *,
    wavelength_range: tuple[float, float] = (210, 700),
    percent_range: tuple[float, float] = (15, 85),
) -> NDArray[np.float64]:
    """Select absorbance events by wavelength-weighted mean intensity."""
    return _select_spectra(
        wavelength,
        spectra,
        wavelength_range,
        percent_range,
        weighted=True,
    )


def _gaussian(
    x: NDArray[np.float64], amplitude: float, center: float, sigma: float
) -> NDArray[np.float64]:
    """Evaluate a one-peak Gaussian profile."""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2))


def fit_pl_spectrum(
    wavelength: ArrayLike,
    intensity: ArrayLike,
    classification: PLClassification,
) -> tuple[float, float, float, float]:
    """Fit one Gaussian and return peak, FWHM, integral, and coefficient of fit."""
    if not classification.is_good or classification.peak_indices.size == 0:
        raise ValueError("a good PL classification with at least one peak is required")

    x_all = np.asarray(wavelength, dtype=float)
    y_all = np.asarray(intensity, dtype=float)
    if x_all.ndim != 1 or y_all.ndim != 1 or x_all.shape != y_all.shape:
        raise ValueError("wavelength and intensity must be equal-length 1D arrays")

    fit_mask = (x_all >= 400) & (x_all <= 800)
    x = x_all[fit_mask]
    y = y_all[fit_mask]
    if x.size < 3:
        raise ValueError("PL fit window must contain at least three samples")

    total = float(np.sum(y))
    if total == 0 or not np.isfinite(total):
        raise ValueError("PL fit window has no finite signal")
    mean = float(np.sum(x * y) / total)
    sigma = float(np.sqrt(np.sum(np.abs(y) * (x - mean) ** 2) / total))

    strongest = int(np.argmax(classification.peak_heights))
    peak_index = int(classification.peak_indices[strongest])
    initial_guess = [float(y_all[peak_index]), float(x_all[peak_index]), sigma]
    try:
        fitted, _ = curve_fit(
            _gaussian,
            x,
            y,
            p0=initial_guess,
            bounds=((0, float(x[0]), 0), (float(np.max(y) * 1.15), 1000, np.inf)),
            maxfev=100000,
        )
    except (RuntimeError, ValueError):
        fitted, _ = curve_fit(
            _gaussian,
            x,
            y,
            p0=initial_guess,
            bounds=(-np.inf, np.inf),
            maxfev=1000000,
        )

    peak = float(fitted[1])
    fitted_sigma = abs(float(fitted[2]))
    fitted_y = _gaussian(x, *fitted)
    r2_mask = (x >= peak - 3 * fitted_sigma) & (x <= peak + 3 * fitted_sigma)
    observed = y[r2_mask]
    predicted = fitted_y[r2_mask]
    if observed.size < 2:
        raise ValueError("PL fit contains too few samples for R-squared")
    residual_sum = float(np.sum((observed - predicted) ** 2))
    total_sum = float(np.sum((observed - np.mean(observed)) ** 2))
    if total_sum == 0:
        raise ValueError("PL fit data are constant")
    r_squared = 1 - residual_sum / total_sum
    pl_integral = float(integrate.simpson(y))
    return peak, 2.355 * fitted_sigma, pl_integral, r_squared


def _fit_baseline(
    wavelength: NDArray[np.float64],
    absorbance: NDArray[np.float64],
    wavelength_range: tuple[float, float],
) -> NDArray[np.float64]:
    """Fit a line over one baseline wavelength range."""
    start = _nearest_index(wavelength, wavelength_range[0])
    stop = _nearest_index(wavelength, wavelength_range[1])
    if start > stop:
        start, stop = stop, start
    x = wavelength[start:stop]
    y = absorbance[start:stop]
    if x.size < 2 or x[0] == x[-1]:
        raise ValueError("absorbance baseline range must contain two samples")
    slope = float((y[-1] - y[0]) / (x[-1] - x[0]))
    intercept = float(np.mean(y))
    fitted, _ = curve_fit(
        lambda values, m, b: values * m + b,
        x,
        y,
        p0=(slope, intercept),
        maxfev=10000,
    )
    return np.asarray(fitted, dtype=float)


def correct_absorbance(
    wavelength: ArrayLike,
    spectra: ArrayLike,
    *,
    percent_range: tuple[float, float] = (10, 70),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Percentile-filter, average, and baseline-correct absorbance spectra."""
    wavelengths, _ = _prepare_spectra(wavelength, spectra)
    selected = select_absorbance_spectra(
        wavelength,
        spectra,
        percent_range=percent_range,
    )
    averaged = np.mean(selected, axis=0)
    x = np.asarray(wavelengths[0], dtype=float)
    short_wavelength = _fit_baseline(x, averaged, (205, 240))
    long_wavelength = _fit_baseline(x, averaged, (750, 950))
    baseline = (
        long_wavelength
        if abs(short_wavelength[0]) >= abs(long_wavelength[0])
        else short_wavelength
    )
    return x, averaged - (baseline[0] * x + baseline[1])


def calculate_plqy(
    absorbance_sample: float,
    pl_integral_sample: float,
    refractive_index_solvent: float,
    *,
    reference_type: str,
    absorbance_reference: float,
    pl_integral_reference: float,
    refractive_index_reference: float,
    plqy_reference: float,
) -> float:
    """Calculate PLQY relative to a fluorescein or quinine reference."""
    with np.errstate(divide="ignore", invalid="ignore"):
        integral_ratio = np.divide(pl_integral_sample, pl_integral_reference)
        refractive_index_ratio = (
            np.divide(refractive_index_solvent, refractive_index_reference) ** 2
        )
        if reference_type == "fluorescein":
            absorbance_ratio = np.divide(
                1 - 10 ** (-absorbance_reference),
                1 - 10 ** (-absorbance_sample),
            )
        elif reference_type == "quinine":
            absorbance_ratio = np.divide(absorbance_reference, absorbance_sample)
        else:
            raise ValueError("reference_type must be either 'fluorescein' or 'quinine'")
        return float(
            plqy_reference * integral_ratio * absorbance_ratio * refractive_index_ratio
        )


def pearson_profile(
    r_exp: ArrayLike,
    g_exp: ArrayLike,
    r_ref: ArrayLike,
    g_ref: ArrayLike,
    *,
    r_range: tuple[float, float] = (2.0, 20.0),
) -> float:
    """Correlate a reference PDF profile onto the experimental radial grid."""
    experimental_r = np.asarray(r_exp, dtype=float)
    experimental_g = np.asarray(g_exp, dtype=float)
    reference_r = np.asarray(r_ref, dtype=float)
    reference_g = np.asarray(g_ref, dtype=float)
    arrays = (experimental_r, experimental_g, reference_r, reference_g)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("PDF profile inputs must be one-dimensional")
    if experimental_r.shape != experimental_g.shape:
        raise ValueError("experimental PDF arrays must have equal lengths")
    if reference_r.shape != reference_g.shape:
        raise ValueError("reference PDF arrays must have equal lengths")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("PDF profile inputs must contain only finite values")

    low, high = r_range
    mask = (experimental_r >= low) & (experimental_r <= high)
    if np.count_nonzero(mask) < 2 or reference_r.size < 2:
        raise ValueError("not enough PDF points to compute Pearson correlation")
    radial = experimental_r[mask]
    observed = experimental_g[mask]
    order = np.argsort(reference_r)
    interpolated = np.interp(radial, reference_r[order], reference_g[order])
    if np.ptp(observed) == 0 or np.ptp(interpolated) == 0:
        raise ValueError("constant PDF profiles have undefined correlation")
    correlation = float(np.corrcoef(observed, interpolated)[0, 1])
    if not np.isfinite(correlation):
        raise ValueError("PDF profile correlation is not finite")
    return correlation
