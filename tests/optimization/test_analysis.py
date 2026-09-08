from __future__ import annotations

import numpy as np
import pytest

from xpd_tools.optimization.analysis import (
    PLClassification,
    calculate_plqy,
    classify_pl,
    correct_absorbance,
    fit_pl_spectrum,
    pearson_profile,
    select_absorbance_spectra,
    select_pl_spectra,
)


def test_classify_pl_suppresses_led_and_accepts_height_boundary() -> None:
    wavelength = np.arange(300.0, 801.0)
    intensity = np.zeros_like(wavelength)
    intensity[65] = 100_000
    intensity[360] = 2_000

    result = classify_pl(wavelength, intensity)

    assert result.is_good
    assert result.peak_wavelength_nm == 660
    np.testing.assert_array_equal(result.peak_indices, [360])
    np.testing.assert_array_equal(result.peak_heights, [2_000])


def test_classify_pl_reports_no_peak() -> None:
    wavelength = np.arange(300.0, 801.0)

    result = classify_pl(wavelength, np.zeros_like(wavelength))

    assert not result.is_good
    assert result.peak_indices.size == 0
    assert result.peak_heights.size == 0
    assert np.isnan(result.peak_wavelength_nm)


def test_select_pl_spectra_ranks_only_configured_window() -> None:
    wavelength = np.array([300.0, 500.0, 600.0])
    spectra = np.array(
        [
            [1000.0, 1.0, 1.0],
            [0.0, 2.0, 2.0],
            [0.0, 3.0, 3.0],
        ]
    )

    selected = select_pl_spectra(wavelength, spectra, percent_range=(0, 50))

    np.testing.assert_array_equal(selected, spectra[:2])
    assert select_pl_spectra(wavelength, spectra[0]).shape == (1, 3)


def test_select_and_correct_absorbance(wavelength: np.ndarray) -> None:
    baseline = 0.001 * wavelength + 0.2
    outlier = baseline.copy()
    outlier[(wavelength >= 210) & (wavelength <= 700)] += 10

    selected = select_absorbance_spectra(
        wavelength,
        np.vstack([baseline, baseline, outlier]),
        percent_range=(0, 50),
    )
    corrected_x, corrected = correct_absorbance(
        wavelength,
        selected,
        percent_range=(0, 100),
    )

    assert selected.shape == (2, wavelength.size)
    np.testing.assert_array_equal(corrected_x, wavelength)
    np.testing.assert_allclose(corrected, 0, atol=1e-10)


def test_fit_single_gaussian(wavelength: np.ndarray) -> None:
    sigma = 20.0
    intensity = 5000 * np.exp(-((wavelength - 660) ** 2) / (2 * sigma**2))
    classification = classify_pl(wavelength, intensity)

    peak, fwhm, integral, r_squared = fit_pl_spectrum(
        wavelength,
        intensity,
        classification,
    )

    assert peak == pytest.approx(660, abs=1e-6)
    assert fwhm == pytest.approx(2.355 * sigma, rel=1e-6)
    assert integral == pytest.approx(5000 * np.sqrt(2 * np.pi) * sigma, rel=1e-3)
    assert r_squared == pytest.approx(1.0)


def test_fit_requires_good_classification(wavelength: np.ndarray) -> None:
    with pytest.raises(ValueError, match="good PL classification"):
        fit_pl_spectrum(
            wavelength,
            np.zeros_like(wavelength),
            PLClassification(
                False,
                np.array([], dtype=np.intp),
                np.array([], dtype=float),
                float("nan"),
            ),
        )


def test_calculate_plqy_formulas() -> None:
    common = {
        "absorbance_sample": 0.2,
        "pl_integral_sample": 10.0,
        "refractive_index_solvent": 1.5,
        "absorbance_reference": 0.1,
        "pl_integral_reference": 5.0,
        "refractive_index_reference": 1.0,
        "plqy_reference": 0.5,
    }

    assert calculate_plqy(reference_type="quinine", **common) == pytest.approx(1.125)
    assert calculate_plqy(reference_type="fluorescein", **common) == pytest.approx(
        0.5 * 2 * ((1 - 10**-0.1) / (1 - 10**-0.2)) * 1.5**2
    )
    with pytest.raises(ValueError, match="reference_type"):
        calculate_plqy(reference_type="unknown", **common)


def test_pearson_profile_is_finite_and_validates_inputs() -> None:
    radial = np.linspace(0, 25, 251)
    profile = np.sin(radial)

    assert pearson_profile(radial, profile, radial, profile) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="constant"):
        pearson_profile(radial, np.ones_like(radial), radial, profile)
    with pytest.raises(ValueError, match="finite"):
        pearson_profile(radial, np.where(radial == 5, np.nan, profile), radial, profile)
    with pytest.raises(ValueError, match="not enough"):
        pearson_profile(np.array([1.0]), np.array([1.0]), radial, profile)
