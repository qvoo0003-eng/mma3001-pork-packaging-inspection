"""Verify finite differences and analyse Gaussian blur on validation images.

References
----------
MMA3001 Week 7 notebook: sections 7.1.3–7.1.5, 7.1.8, 7.1.11–7.1.12.
MMA3001 Week 7 lecture slides: section 7.1.

Method
------
Centred interior differences have O(h**2) truncation error for smooth functions.
Two-point forward/backward boundary differences have O(h) truncation error.
These are boundary stencils, not imposed physical boundary conditions.

Image analysis uses grayscale intensity in [0, 1] and spacing of one pixel.
Gaussian blur uses sigma 0, 1 and 2 pixels; nonzero sigma uses radius ceil(3*sigma)
and OpenCV BORDER_REFLECT_101 padding. Each blur starts from the original image.

Limitations
-----------
Differentiation amplifies noise through division by grid spacing. Finite pixel
resolution and first-order boundaries limit accuracy. Gaussian kernel size,
sigma and padding affect measured edge strength. Contrast scaling changes
gradient magnitude; brightness changes involving clipping or nonlinear camera
responses can also change it. An ideal additive brightness offset cancels.

Synthetic verification concerns numerical accuracy on known smooth functions.
Image measurements describe edge strength, not derivative error against an
unknown continuous scene. Reduced edge strength does not establish reduced
YOLO accuracy; detector performance requires a separate experiment.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIRECTORY = REPOSITORY_ROOT / "data/cleaned/v1/valid/images"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results/finite_difference"

SIGMAS = (0.0, 1.0, 2.0)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_VALIDATION_IMAGES = 119


def finite_differences(
    values: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate horizontal and vertical derivatives on a uniform 2-D grid.

    Parameters
    ----------
    values : array_like
        Finite real-valued array with at least two rows and two columns.
        Rows follow y; columns follow x.
    dx, dy : float, optional
        Positive finite spacing along columns and rows, respectively.

    Returns
    -------
    derivative_x, derivative_y : ndarray
        Float64 derivatives with the same shape as values. Units are input
        intensity units divided by the corresponding coordinate units.

    Raises
    ------
    ValueError
        If values or spacings are invalid, or derivative arithmetic overflows.

    Notes
    -----
    Interior: (f[i+1] - f[i-1]) / (2*h).
    First sample: (f[1] - f[0]) / h.
    Last sample: (f[-1] - f[-2]) / h.
    """
    spacings = []
    for name, spacing in (("dx", dx), ("dy", dy)):
        if (
            not np.isscalar(spacing)
            or isinstance(spacing, (str, bytes, bool, np.bool_))
            or not np.isrealobj(spacing)
        ):
            raise ValueError(f"{name} must be a positive finite real scalar.")
        try:
            number = float(spacing)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a positive finite scalar.") from exc
        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"{name} must be positive and finite.")
        spacings.append(number)
    dx, dy = spacings

    try:
        if np.iscomplexobj(values):
            raise ValueError("Complex-valued input is unsupported.")
        field = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Input must be a real numeric 2-D array.") from exc

    if field.ndim != 2 or min(field.shape) < 2:
        raise ValueError("Input must have at least two rows and two columns.")
    if not np.all(np.isfinite(field)):
        raise ValueError("Input must contain only finite values.")

    derivative_x = np.empty_like(field)
    derivative_y = np.empty_like(field)

    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            # Horizontal derivative: columns vary in x.
            derivative_x[:, 1:-1] = (
                field[:, 2:] - field[:, :-2]
            ) / (2.0 * dx)
            derivative_x[:, 0] = (field[:, 1] - field[:, 0]) / dx
            derivative_x[:, -1] = (field[:, -1] - field[:, -2]) / dx

            # Vertical derivative: rows vary in y.
            derivative_y[1:-1, :] = (
                field[2:, :] - field[:-2, :]
            ) / (2.0 * dy)
            derivative_y[0, :] = (field[1, :] - field[0, :]) / dy
            derivative_y[-1, :] = (field[-1, :] - field[-2, :]) / dy
    except FloatingPointError as exc:
        raise ValueError("Values or spacings exceed the usable numeric range.") from exc

    return derivative_x, derivative_y


def gradient_magnitude(
    values: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
) -> np.ndarray:
    """Calculate sqrt((dI/dx)**2 + (dI/dy)**2).

    Parameters
    ----------
    values : array_like
        Finite real-valued 2-D field.
    dx, dy : float, optional
        Positive finite grid spacings.

    Returns
    -------
    ndarray
        Nonnegative gradient magnitude at every grid point.

    Raises
    ------
    ValueError
        If input validation fails or the magnitude is not finite.
    """
    derivative_x, derivative_y = finite_differences(values, dx, dy)
    with np.errstate(over="ignore"):
        magnitude = np.hypot(derivative_x, derivative_y)
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("Gradient magnitude exceeds the usable numeric range.")
    return magnitude


def report_check(name: str, passed: bool, detail: str) -> bool:
    """Print an explicit verification outcome.

    Parameters
    ----------
    name : str
        Verification description.
    passed : bool
        Whether the numerical acceptance criterion was satisfied.
    detail : str
        Calculated error, tolerance or other supporting evidence.

    Returns
    -------
    bool
        The supplied outcome.
    """
    print(f"{'PASS' if passed else 'FAIL'} | {name} | {detail}")
    return bool(passed)


def verify() -> int:
    """Verify derivatives using analytical functions and grid refinement.

    Returns
    -------
    int
        Zero if every verification passes, otherwise one.

    Notes
    -----
    Exact-function tolerance is 1e-12. Smooth-function checks use explicit
    finest-grid error limits. Observed orders must be within 0.2 of the
    theoretical orders: two for interiors and one for boundary stencils.
    """
    print("SYNTHETIC NUMERICAL VERIFICATION")
    print("No dataset images or detector predictions are used.\n")
    checks = []
    tolerance = 1e-12

    # Constant function: both derivatives and the magnitude must be zero.
    constant = np.full((7, 9), 4.25)
    gx, gy = finite_differences(constant, dx=0.125, dy=0.2)
    error = max(
        float(np.max(np.abs(gx))),
        float(np.max(np.abs(gy))),
        float(np.max(gradient_magnitude(constant, 0.125, 0.2))),
    )
    checks.append(report_check(
        "Constant function",
        error <= tolerance,
        f"max error={error:.3e}, tolerance={tolerance:.1e}",
    ))

    # Unequal spacings help expose swapped axes or incorrect denominators.
    x = np.arange(9, dtype=float) * 0.125
    y = np.arange(7, dtype=float) * 0.2
    X, Y = np.meshgrid(x, y, indexing="xy")
    linear = 3.0 * X - 2.0 * Y + 1.0
    gx, gy = finite_differences(linear, dx=0.125, dy=0.2)
    error_x = float(np.max(np.abs(gx - 3.0)))
    error_y = float(np.max(np.abs(gy + 2.0)))
    error_g = float(np.max(
        np.abs(gradient_magnitude(linear, 0.125, 0.2) - np.sqrt(13.0))
    ))
    checks.append(report_check(
        "Linear function, including every boundary",
        max(error_x, error_y, error_g) <= tolerance,
        f"x error={error_x:.3e}, y error={error_y:.3e}, "
        f"magnitude error={error_g:.3e}, tolerance={tolerance:.1e}",
    ))

    # For f=x^2+3y^2, one-sided errors are known explicitly.
    quadratic = X**2 + 3.0 * Y**2
    gx, gy = finite_differences(quadratic, dx=0.125, dy=0.2)
    boundary_error = max(
        float(np.max(np.abs(gx[:, 0] - (2.0 * x[0] + 0.125)))),
        float(np.max(np.abs(gx[:, -1] - (2.0 * x[-1] - 0.125)))),
        float(np.max(np.abs(gy[0, :] - (6.0 * y[0] + 3.0 * 0.2)))),
        float(np.max(np.abs(gy[-1, :] - (6.0 * y[-1] - 3.0 * 0.2)))),
    )
    checks.append(report_check(
        "Forward/backward boundary formulas",
        boundary_error <= tolerance,
        f"error against exact one-sided slopes={boundary_error:.3e}, "
        f"tolerance={tolerance:.1e}",
    ))

    # Minimal dimensions and unsigned input must retain negative derivatives.
    small = np.array([[10, 8], [7, 5]], dtype=np.uint8)
    gx, gy = finite_differences(small)
    error = max(
        float(np.max(np.abs(gx + 2.0))),
        float(np.max(np.abs(gy + 3.0))),
    )
    checks.append(report_check(
        "2x2 unsigned input",
        error <= tolerance,
        f"max error={error:.3e}, tolerance={tolerance:.1e}",
    ))

    invalid_cases = [
        (np.ones((1, 3)), 1.0, 1.0),
        (np.ones((3, 1)), 1.0, 1.0),
        (np.ones((2, 2, 3)), 1.0, 1.0),
        (np.full((2, 2), np.nan), 1.0, 1.0),
        (np.ones((2, 2)), 0.0, 1.0),
        (np.ones((2, 2)), 1.0, -1.0),
        (np.ones((2, 2)), np.inf, 1.0),
    ]
    rejected = 0
    for field, dx, dy in invalid_cases:
        try:
            finite_differences(field, dx, dy)
        except ValueError:
            rejected += 1
    checks.append(report_check(
        "Invalid input rejection",
        rejected == len(invalid_cases),
        f"rejected {rejected}/{len(invalid_cases)} invalid inputs",
    ))

    # Smooth analytical field:
    # f=sin(x)+cos(y), df/dx=cos(x), df/dy=-sin(y).
    # Nonzero endpoints avoid special cancellations hiding boundary errors.
    print("\nSmooth function: f(x,y)=sin(x)+cos(y), x,y in [0.2, 1.2]")
    print("Errors are maximum absolute errors; interiors exclude outer pixels.")
    print("   n          h       interior_x   interior_y   boundary_x   boundary_y")
    error_rows = []

    for n in (17, 33, 65, 129):
        coordinates = np.linspace(0.2, 1.2, n)
        h = float(coordinates[1] - coordinates[0])
        X, Y = np.meshgrid(coordinates, coordinates, indexing="xy")
        field = np.sin(X) + np.cos(Y)
        gx, gy = finite_differences(field, h, h)
        error_x = np.abs(gx - np.cos(X))
        error_y = np.abs(gy + np.sin(Y))

        errors = [
            float(np.max(error_x[1:-1, 1:-1])),
            float(np.max(error_y[1:-1, 1:-1])),
            float(np.max(error_x[:, [0, -1]])),
            float(np.max(error_y[[0, -1], :])),
        ]
        error_rows.append(errors)
        print(
            f"{n:4d}  {h:10.6f}  "
            + "  ".join(f"{value:11.4e}" for value in errors)
        )

    errors = np.asarray(error_rows)
    checks.append(report_check(
        "Smooth analytical derivative accuracy, finest grid",
        bool(
            np.all(np.isfinite(errors))
            and np.max(errors[-1, :2]) <= 2e-5
            and np.max(errors[-1, 2:]) <= 1e-2
        ),
        f"interior max={np.max(errors[-1, :2]):.3e} <= 2e-5; "
        f"boundary max={np.max(errors[-1, 2:]):.3e} <= 1e-2",
    ))

    # Each refinement halves h, hence p=log2(E(h)/E(h/2)).
    # Zero/nonfinite errors cannot establish an observed convergence order.
    orders = np.full((3, 4), np.nan)
    valid_ratios = (
        np.isfinite(errors[:-1])
        & np.isfinite(errors[1:])
        & (errors[:-1] > 0)
        & (errors[1:] > 0)
    )
    orders[valid_ratios] = np.log2(
        errors[:-1][valid_ratios] / errors[1:][valid_ratios]
    )

    for column, (name, expected_order) in enumerate((
        ("Horizontal centred convergence", 2.0),
        ("Vertical centred convergence", 2.0),
        ("Horizontal boundary convergence", 1.0),
        ("Vertical boundary convergence", 1.0),
    )):
        observed = orders[:, column]
        passed = bool(
            np.all(np.isfinite(observed))
            and np.all(np.diff(errors[:, column]) < 0)
            and np.all(np.abs(observed - expected_order) <= 0.2)
        )
        checks.append(report_check(
            name,
            passed,
            f"observed orders={np.array2string(observed, precision=4)}; "
            f"required {expected_order:.1f} +/- 0.2 with decreasing errors",
        ))

    success = all(checks)
    print(f"\nOverall: {'PASS' if success else 'FAIL'} "
          f"({sum(checks)}/{len(checks)} checks passed)")
    print("These checks verify synthetic derivatives, not real-image accuracy.")
    return 0 if success else 1


def read_grayscale(path: Path) -> np.ndarray:
    """Load an 8-bit image as normalized floating-point grayscale.

    Parameters
    ----------
    path : Path
        Image filename. NumPy reads bytes to support Windows Unicode paths.

    Returns
    -------
    ndarray
        Float64 grayscale image in [0, 1].

    Raises
    ------
    OSError
        If the image cannot be read.
    ValueError
        If decoding, image dimensions, bit depth or channel count is unsupported.
    """
    import cv2

    encoded = np.fromfile(path, dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError("Image file is empty.")

    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("OpenCV could not decode the image.")
    if image.dtype != np.uint8:
        raise ValueError("Expected an 8-bit image; other bit depths are unsupported.")

    if image.ndim == 2:
        grayscale = image
    elif image.ndim == 3 and image.shape[2] == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Expected grayscale or three-channel BGR image.")

    if min(grayscale.shape) < 2:
        raise ValueError("Image must have at least two rows and two columns.")

    return grayscale.astype(np.float64) / 255.0


def analyze() -> int:
    """Measure original and blurred validation-image edge strength.

    Returns
    -------
    int
        Zero after complete processing; one if any files were skipped.

    Raises
    ------
    FileNotFoundError
        If the validation directory is absent.
    ValueError
        If no supported images can be analysed.
    OSError
        If output files cannot be written.

    Notes
    -----
    Each successful image contributes all three blur levels.
    Summary means weight images equally. Standard deviations use ddof=0.
    Relative change is 100*(mean_sigma - mean_original)/mean_original.
    It is left blank when the original mean is zero.
    """
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not VALIDATION_DIRECTORY.is_dir():
        raise FileNotFoundError(
            f"Validation image directory not found: {VALIDATION_DIRECTORY}"
        )

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)

    print("REAL VALIDATION-IMAGE ANALYSIS")
    print(f"Input: {VALIDATION_DIRECTORY}")
    print(f"NumPy {np.__version__}; OpenCV {cv2.__version__}")
    print("Intensity: [0, 1]; dx=dy=1 pixel; all pixels included in each mean.")
    print("Blur: sigma=0,1,2; kernels=none,7x7,13x13; BORDER_REFLECT_101.")

    files = sorted(
        (path for path in VALIDATION_DIRECTORY.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(VALIDATION_DIRECTORY).as_posix(),
    )
    image_files = []
    skipped = 0
    validation_root = VALIDATION_DIRECTORY.resolve()

    for path in files:
        if not path.resolve().is_relative_to(validation_root):
            print(f"WARNING: Skipping path outside validation directory: {path}")
            skipped += 1
        elif path.suffix.lower() not in IMAGE_SUFFIXES:
            print(f"WARNING: Unsupported file skipped: {path.name}")
            skipped += 1
        else:
            image_files.append(path)

    if len(image_files) != EXPECTED_VALIDATION_IMAGES:
        print(
            f"WARNING: Found {len(image_files)} supported validation files; "
            f"expected {EXPECTED_VALIDATION_IMAGES}."
        )
    if not image_files:
        raise ValueError("No supported validation images were found.")

    metric_rows = []
    scores_by_image = []

    for path in image_files:
        filename = path.relative_to(VALIDATION_DIRECTORY).as_posix()
        try:
            original = read_grayscale(path)
            scores = []
            for sigma in SIGMAS:
                if sigma == 0:
                    image = original
                else:
                    radius = int(np.ceil(3.0 * sigma))
                    kernel_size = 2 * radius + 1
                    image = cv2.GaussianBlur(
                        original,
                        (kernel_size, kernel_size),
                        sigmaX=sigma,
                        sigmaY=sigma,
                        borderType=cv2.BORDER_REFLECT_101,
                    )
                score = float(np.mean(gradient_magnitude(image, dx=1.0, dy=1.0)))
                scores.append(score)

        except (OSError, ValueError, cv2.error) as exc:
            print(f"WARNING: Skipping unreadable/unsupported image {filename}: {exc}")
            skipped += 1
            continue

        # Append only after all blur levels succeed, preserving paired samples.
        scores_by_image.append(scores)
        metric_rows.extend(
            (filename, sigma, score)
            for sigma, score in zip(SIGMAS, scores)
        )

    if not scores_by_image:
        raise ValueError("No images were successfully analysed; no outputs written.")

    scores = np.asarray(scores_by_image, dtype=np.float64)
    means = np.mean(scores, axis=0)
    standard_deviations = np.std(scores, axis=0, ddof=0)
    count = scores.shape[0]
    baseline = float(means[0])

    summary_rows = []
    print("\n sigma   images      mean gradient       population SD    change (%)")
    for index, sigma in enumerate(SIGMAS):
        relative_change = (
            100.0 * (float(means[index]) - baseline) / baseline
            if baseline > 0
            else None
        )
        summary_rows.append((
            sigma,
            count,
            float(means[index]),
            float(standard_deviations[index]),
            "" if relative_change is None else relative_change,
        ))
        change_text = "undefined" if relative_change is None else f"{relative_change:.3f}"
        print(
            f"{sigma:6.1f}   {count:6d}   {means[index]:16.8f}   "
            f"{standard_deviations[index]:16.8f}   {change_text}"
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIRECTORY / "per_image_metrics.csv"
    summary_path = OUTPUT_DIRECTORY / "summary.csv"
    plot_path = OUTPUT_DIRECTORY / "blur_comparison.png"

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_filename", "blur_sigma", "mean_gradient_magnitude"))
        writer.writerows(metric_rows)

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "blur_sigma",
            "n_images",
            "mean_gradient_magnitude",
            "std_gradient_magnitude",
            "relative_change_percent_vs_sigma0",
        ))
        writer.writerows(summary_rows)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    try:
        axis.plot(SIGMAS, means, marker="o", linewidth=2, color="tab:blue")
        axis.set_xticks(SIGMAS)
        axis.set_xlabel("Gaussian blur sigma (pixels)")
        axis.set_ylabel("Mean gradient magnitude\n(normalized intensity / pixel)")
        axis.set_title(f"Validation-image edge strength under blur (n={count})")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(plot_path, dpi=160)
    finally:
        plt.close(figure)

    print(f"\nSuccessfully analysed: {count}; skipped files: {skipped}")
    for path in (metrics_path, summary_path, plot_path):
        print(f"Saved: {path}")
    print(
        "Negative relative change means lower measured edge strength. "
        "It does not establish lower YOLO detection accuracy."
    )
    print("Synthetic verification is separate; run --mode verify.")
    return 1 if skipped else 0


def main() -> int:
    """Run the selected verification or validation-image analysis mode.

    Returns
    -------
    int
        Zero on success, one on failure/incomplete analysis, or 130 on interruption.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("verify", "analyze"))
    args = parser.parse_args()

    try:
        return verify() if args.mode == "verify" else analyze()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())