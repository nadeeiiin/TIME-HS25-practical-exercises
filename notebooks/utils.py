from typing import Optional, Union, Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from arch.unitroot import ADF, KPSS, PhillipsPerron, ZivotAndrews
from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanSquaredError
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.naive import NaiveForecaster
from sktime.split import temporal_train_test_split, ExpandingWindowSplitter
from sktime.utils.plotting import plot_interval
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import acovf


def get_figure(
        nrows: int = 1, ncols: int = 1, figsize: tuple[int, int] = (12, 4), grid: bool = True, **kwargs
) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
    if isinstance(axs, plt.Axes):
        axs = [axs]
    else:
        axs = axs.flatten().tolist()
    for ax in axs:
        ax.grid(grid)
    return fig, axs


def time_plot(
        x: Union[np.ndarray, pd.Series],
        y: Union[np.ndarray, pd.Series],
        label: str = "",
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        prefix_title: bool = False,
        xlabel: str = "$t$",
        ylabel: str = "$x_t$",
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        show_trend: bool = False,
        with_acf: bool = False,
        with_pacf: bool = False,
        nlags: int = 40,
        figsize: tuple[int, int] = (12, 4),
        return_fig=False,
        **plot_kwargs,
) -> Optional[tuple[plt.Figure, plt.Axes]]:
    nplots = 1 + int(with_acf) + int(with_pacf)
    if ax is None:
        fig, axs = get_figure(ncols=nplots, figsize=figsize)
        ax = axs[0]
        if with_acf:
            ax_acf = axs[1]
        if with_pacf:
            ax_pacf = axs[1+int(with_acf)]
    else:
        fig = ax.get_figure()
        if with_acf or with_pacf:
            raise ValueError("Cannot plot ACF/PACF with single axis.")
    ax.plot(x, y, label=label, **plot_kwargs)
    if show_trend:
        trend_x = x
        if pd.api.types.is_datetime64_any_dtype(x):
            trend_x = np.array([d.toordinal() for d in x])
        coefs = np.polyfit(trend_x, y, 1)
        trend = np.polyval(coefs, trend_x)
        ax.plot(x, trend, label="Trend line", linestyle='--', color='red')
    if title is not None:
        prefixed_title = f"Time plot: {title}" if prefix_title else title
        ax.set_title(prefixed_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if label != "" or show_trend:
        ax.legend()
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if with_acf:
        acf_plot(y, nlags, ax_acf, title=title, prefix_title=prefix_title)
    if with_pacf:
        pacf_plot(y, nlags, ax_pacf, title=title, prefix_title=prefix_title)
    fig.tight_layout()
    if return_fig:
        return fig, ax


def acovf_plot(
        y: Union[np.ndarray, pd.Series], nlags: int, ax: plt.Axes, title: str = "", prefix_title: bool = True
) -> None:
    autocov = acovf(y, nlag=nlags, fft=False)
    ax.stem(np.arange(nlags+1), autocov, markerfmt="o", basefmt=" ")
    ax.set_title(f"ACOVF plot: {title}" if prefix_title else title)
    ax.set_xlabel("Lag $h$")
    ax.set_ylabel("$\\gamma (h)$")


def acf_plot(
        y: Union[np.ndarray, pd.Series], nlags: int, ax: plt.Axes, title: str = "", prefix_title: bool = True
) -> None:
    sm.graphics.tsa.plot_acf(y, lags=nlags, ax=ax)
    ax.set_title(f"ACF plot: {title}" if prefix_title else title)
    ax.set_xlabel("Lag $h$")
    ax.set_ylabel("$\\rho (h)$")


def pacf_plot(
        y: Union[np.ndarray, pd.Series], nlags: int, ax: plt.Axes, title: str = "", prefix_title: bool = True
) -> None:
    sm.graphics.tsa.plot_pacf(y, lags=nlags, ax=ax)
    ax.set_title(f"PACF plot: {title}" if prefix_title else title)
    ax.set_xlabel("Lag $h$")
    ax.set_ylabel("$\\pi (h)$")


def moving_average_smoothing(time_series: np.ndarray, window_size: int, mode: str = 'valid') -> tuple[str, np.ndarray]:
    low = int(np.ceil((window_size-1) / 2))
    window = range(-low, int(np.floor((window_size-1) / 2))+1)
    time_shifts = [f"{s}" if s < 0 else f"+{s}" for s in window]
    time_shifts = [s if s != "+0" else "" for s in time_shifts]
    formula_terms = [f'w_{{t{s}}}' for s in time_shifts]
    if len(formula_terms) > 5:
        formula_terms = formula_terms[:2] + ['\\ldots'] + formula_terms[-1:]
    formula = f"$y_t=({'+'.join(formula_terms)})/{window_size}$"
    ma = np.convolve(time_series, np.ones(window_size)/window_size, mode=mode)
    return formula, np.arange(low, low+len(ma)), ma


def stationarity_tests(time_series: np.ndarray) -> None:
    results = {}
    pvalues = {}
    adf_test = ADF(time_series)
    results['ADF'] = 'Stationary' if adf_test.pvalue < 0.05 else 'Non-Stationary'
    pvalues['ADF_pvalue'] = adf_test.pvalue
    pp_test = PhillipsPerron(time_series)
    results['PP'] = 'Stationary' if pp_test.pvalue < 0.05 else 'Non-Stationary'
    pvalues['PP_pvalue'] = pp_test.pvalue
    kpss_test = KPSS(time_series)
    results['KPSS'] = 'Stationary' if kpss_test.pvalue >= 0.05 else 'Non-Stationary'
    pvalues['KPSS_pvalue'] = kpss_test.pvalue
    za_test = ZivotAndrews(time_series)
    results['ZA'] = 'Stationary' if za_test.pvalue < 0.05 else 'Non-Stationary'
    pvalues['ZA_pvalue'] = za_test.pvalue

    print("Stationarity Test Results:")
    for test, result in results.items():
        print(f"\t- {test}: {result} (p-value: {pvalues[test + '_pvalue']:.4f})")


def arma_model_properties(model: ArmaProcess, formulae: Optional[str] = None) -> str:
    p = len(model.arcoefs)
    q = len(model.macoefs)
    if p != 0 and q != 0:
        model_name = f"ARMA({p},{q})"
    elif p == 0:
        model_name = f"MA({q})"
    else:
        model_name = f"AR({p})"
    if formulae is not None:
        print(f"{model_name} {formulae}")
    if p != 0:
        print(f'{model_name} AR roots: {model.arroots}')
        print(f'{model_name} is stationary: {model.isstationary}')
    if q != 0:
        print(f'{model_name} MA roots: {model.maroots}')
        print(f'{model_name} is invertible: {model.isinvertible}')
    return model_name


def plot_arma_process(ar_coeffs: list[float], ma_coeffs: list[float], formula: str = "", nsamples: int = 1000, acf_or_pacf: bool = True):
    model = ArmaProcess(ar=ar_coeffs, ma=ma_coeffs)
    model_name = arma_model_properties(model, formula)
    time_plot(
        x=np.arange(nsamples),
        y=model.generate_sample(nsample=nsamples),
        title=f"{model_name} {formula}",
        prefix_title=False,
        with_acf=acf_or_pacf,
        with_pacf= not acf_or_pacf,
        nlags=50,
        xlim=(0, 200),
    )


def arma_theoretical_spectral_density(frequencies: np.ndarray, sigma: float, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    phi_poly = 1 - sum(phi[i] * np.exp(-2j * np.pi * frequencies * (i + 1)) for i in range(len(phi)))
    theta_poly = 1 + sum(theta[i] * np.exp(-2j * np.pi * frequencies * (i + 1)) for i in range(len(theta)))
    return sigma**2 * (np.abs(theta_poly)**2) / (np.abs(phi_poly)**2)


def fit_and_forecast(
        model: BaseForecaster, time_series: pd.Series, train_size: int, horizon: Optional[int] = None
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if horizon is None:
        horizon = len(time_series) - train_size
    train_ts = time_series[:train_size]
    test_ts = time_series[train_size:train_size+horizon]
    model.fit(train_ts)
    if model.get_tag("capability:insample"):
        fitted_values = model.predict(train_ts.index)
    else:
        fitted_values = None
    pred = model.predict(fh=test_ts.index)
    if model.get_tag("capability:pred_int"):
        interval = model.predict_interval(fh=test_ts.index, coverage=[.75, .95])
    else:
        interval = None
    return fitted_values, pred, interval


def plot_fit_and_forecast(
        ax: plt.Axes, fitted_values: Optional[pd.Series] = None, pred: Optional[pd.Series] = None,
        interval: Optional[pd.Series] = None, model_name: str = ""
) -> None:
    color = ax._get_lines.get_next_color()
    model_name_shown = False
    if fitted_values is not None:
        ax.plot(fitted_values.index, fitted_values, label=model_name, color=color, linestyle='--')
        model_name_shown = True
    if pred is not None:
        ax.plot(pred.index, pred, label=None if model_name_shown else model_name, color=color, linestyle='--')
    if interval is not None:
        plot_interval(ax, interval)


def baseline_forecasting(
        time_series: pd.Series, train_size: int, horizon: Optional[int] = None, period: int = 1
) -> Dict[str, Tuple[pd.Series, pd.Series, pd.Series]]:
    baselines = {
        "Naïve": NaiveForecaster(strategy="last"),
        "Naïve with drift": NaiveForecaster(strategy="drift"),
        "Mean": NaiveForecaster(strategy="mean"),
        "Seasonal naïve": NaiveForecaster(strategy="last", sp=period),
    }
    return {
        name: fit_and_forecast(model, time_series, train_size, horizon)
        for name, model in baselines.items()
    }


def residual_analysis_plots(
        standardized_residuals: np.ndarray, return_fig=False, axs: Optional[List[plt.Axes]] = None
) -> Optional[tuple[plt.Figure, plt.Axes]]:
    if axs is None:
        fig, axs = get_figure(ncols=2, nrows=2, figsize=(12, 8))
        [ax_time, ax_hist, ax_qq, ax_acf] = axs
    else:
        [ax_time, ax_hist, ax_qq, ax_acf] = axs
        fig = ax_time.get_figure()
    for ax in axs:
        ax.grid(True)
    time_plot(
        x=np.arange(len(standardized_residuals)),
        y=standardized_residuals,
        ax=ax_time,
        title="Standardized residuals",
        xlabel="",
        ylabel="",
    )

    ax_hist.hist(standardized_residuals, density=True, label='Hist', edgecolor='#FFFFFF')
    kde = gaussian_kde(standardized_residuals)
    xlim = (-1.96 * 2, 1.96 * 2)
    x = np.linspace(xlim[0], xlim[1])
    ax_hist.plot(x, kde(x), label='KDE')
    ax_hist.plot(x, norm.pdf(x), label='N(0,1)')
    #ax_hist.set_xlim(xlim)
    ax_hist.legend()
    ax_hist.set_title('Histogram with estimated density')

    sm.qqplot(standardized_residuals, line='s', ax=ax_qq)
    ax_qq.set_title(f'Normal Q-Q')

    sm.graphics.tsa.plot_acf(standardized_residuals, ax=ax_acf)
    ax_acf.set_title(f'Correlogram')

    fig.tight_layout()
    if return_fig:
        return fig, axs


def split_time_series(
        time_series: pd.Series, valid_size: float = .2, test_size: float = .2
) -> Tuple[pd.Series, Tuple[pd.Series, pd.Series], pd.Series]:
    full_train, test = temporal_train_test_split(time_series, test_size=test_size)
    validation_split = int(len(full_train) * (1 - valid_size))
    train, eval = full_train[:validation_split], full_train[validation_split:]
    return full_train, (train, eval), test


def evaluate_forecaster(
        model: BaseForecaster, train: pd.Series, eval: pd.Series, fh: ForecastingHorizon,
        metrics: Optional[List] = None, is_test: bool = False
) -> pd.Series:
    if metrics is None:
        metrics = [MeanSquaredError(square_root=True), MeanAbsoluteError()]
    name = "test" if is_test else "validation"
    train_eval = pd.concat([train, eval])
    cv = ExpandingWindowSplitter(fh=fh, initial_window=len(train), step_length=len(fh))
    print(f"Evaluating {model.__class__.__name__}: {name} split divided into", cv.get_n_splits(train_eval), "cv folds")
    model_results = evaluate(forecaster=model, strategy="update" if is_test else "refit", y=train_eval, cv=cv, scoring=metrics)
    print_aggregated_scores(f"{model.__class__.__name__} {name}", model_results)
    model_preds = model.fit(train).predict(eval.index)
    return model_preds


def print_aggregated_scores(model_info: str, results: pd.DataFrame) -> None:
    for metric in [c for c in results.columns if c.startswith('test_')]:
        mean = results[metric].mean()
        std = results[metric].std()
        print(f"{model_info} {metric[5:]}: {mean:.3f} ± {std:.3f}")


def fit_sinusoids(
        time_series: pd.Series, frequencies: List[float]
) -> Tuple[RegressionResults, float, List[np.ndarray]]:
    n = len(time_series)
    t = np.arange(n)
    sinusoids = [fn(2 * np.pi * f * t) for f in frequencies for fn in [np.cos, np.sin]]
    X = np.column_stack((np.ones(n), *sinusoids))
    model = sm.OLS(time_series, X)
    results = model.fit()
    mu, *coeffs = results.params
    sinusoids = [(coeffs[j]*sinusoids[j], coeffs[j+1]*sinusoids[j+1]) for j in range(0, len(sinusoids), 2)]
    return results, mu, sinusoids


def plot_fft(
        ax: plt.Axes, time_series: pd.Series, sample_spacing: float = 1., sample_spacing_name: str = "unit time",
        show_negative_freqs: bool = False, return_fft: bool = False, periodogram: bool = False, log_scale: bool = False,
        show_estimated_density: bool = False, smoothing_iteration: int = 0, smoothing_window_percent: float = 0.05
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n = len(time_series)
    xf = np.fft.fftfreq(n, sample_spacing)
    yf = np.fft.fft(time_series) / n                            # numpy FFT output is not normalized.
    spectrum = np.abs(yf)                                       # magnitude spectrum
    if periodogram:
        spectrum = spectrum**2
    if not show_negative_freqs:
        xf = xf[:n//2]
        spectrum = spectrum[:n//2] * 2                          # multiply by 2 to account for the negative frequencies
    ax.stem(xf, spectrum, basefmt=" ")
    if show_estimated_density:
        smoothed_spectrum = spectrum
        smoothing_window = max(1, int(n*smoothing_window_percent))
        for _ in range(smoothing_iteration):
            smoothed_spectrum = moving_average_smoothing(smoothed_spectrum, smoothing_window, mode='same')[-1]
        label = "Estimated density"
        if smoothing_iteration > 0:
            label = f"{label} ({smoothing_iteration}x{smoothing_window}-MA)"
        ax.plot(xf, smoothed_spectrum, label=label, color='black')
    ax.set_title('Periodogram' if periodogram else 'DFT (Magnitude spectrum)')
    ax.set_xlabel(f'Frequency [cycles per {sample_spacing_name}]')
    ax.set_ylabel('Value' if periodogram else 'Magnitude')
    if log_scale:
        ax.set_yscale('log')
    ax.get_figure().tight_layout()
    if return_fft:
        return xf, yf, spectrum
