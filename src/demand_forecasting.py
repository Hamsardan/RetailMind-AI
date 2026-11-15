
"""Day 10: Store-1 multi-step sales forecasting with Keras attention."""

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

LOOKBACK = 30
HORIZON = 14


def load_store_sales(csv_path, store=1):
    df = pd.read_csv(
        csv_path,
        usecols=['Store', 'Date', 'Sales'],
        parse_dates=['Date']
    )

    sales = (
        df.loc[df['Store'].eq(store), ['Date', 'Sales']]
        .sort_values('Date')
        .set_index('Date')['Sales']
    )

    if sales.empty:
        raise ValueError(f'No sales data for store {store}')

    return sales.asfreq('D')


def prepare_data(sales, lookback=LOOKBACK, horizon=HORIZON):
    """Chronological 80/20 split with no overlapping train/test targets."""

    split = int(len(sales) * 0.8)
    scale = float(sales.iloc[:split].max())

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Training sales maximum must be positive')

    values = (sales / scale).to_numpy(dtype='float32')

    xs, ys, starts = [], [], []

    for start in range(lookback, len(values) - horizon + 1):
        x = values[start - lookback:start]
        y = values[start:start + horizon]

        if np.isfinite(x).all() and np.isfinite(y).all():
            xs.append(x[:, None])
            ys.append(y[:, None])
            starts.append(start)

    xs = np.asarray(xs, dtype='float32').reshape(-1, lookback, 1)
    ys = np.asarray(ys, dtype='float32').reshape(-1, horizon, 1)
    starts = np.asarray(starts)

    train = starts + horizon <= split
    test = starts >= split

    if not train.any() or not test.any():
        raise ValueError('Not enough complete windows')

    return (
        (xs[train], ys[train]),
        (xs[test], ys[test]),
        starts[test],
        scale,
        split
    )


def build_attention_model(lookback=LOOKBACK, horizon=HORIZON, units=32):
    encoder_input = keras.Input(
        shape=(lookback, 1),
        name='history'
    )

    encoder_outputs, state_h, state_c = keras.layers.LSTM(
        units,
        return_sequences=True,
        return_state=True,
        name='encoder'
    )(encoder_input)

    decoder_seed = keras.layers.RepeatVector(
        horizon,
        name='repeat_context'
    )(state_h)

    decoder_outputs = keras.layers.LSTM(
        units,
        return_sequences=True,
        name='decoder'
    )(
        decoder_seed,
        initial_state=[state_h, state_c]
    )

    context, attention_scores = keras.layers.AdditiveAttention(
        name='temporal_attention'
    )(
        [decoder_outputs, encoder_outputs],
        return_attention_scores=True
    )

    merged = keras.layers.Concatenate(
        name='combine'
    )([decoder_outputs, context])

    predictions = keras.layers.TimeDistributed(
        keras.layers.Dense(1),
        name='sales'
    )(merged)

    model = keras.Model(
        encoder_input,
        predictions,
        name='demand_seq2seq_attention'
    )

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    attention_model = keras.Model(
        encoder_input,
        attention_scores,
        name='attention_weights'
    )

    return model, attention_model


def build_day9_baseline(lookback=LOOKBACK):
    model = keras.Sequential([
        keras.Input(shape=(lookback, 1)),
        keras.layers.LSTM(32),
        keras.layers.Dense(1)
    ], name='day9_lstm_baseline')

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model


def recursive_baseline(model, windows, horizon=HORIZON):
    """Generate multi-step forecasts from a single-step LSTM."""

    window = np.array(windows, dtype='float32', copy=True)
    forecasts = []

    for _ in range(horizon):
        next_step = model.predict(
            window,
            verbose=0
        ).reshape(-1, 1, 1)

        forecasts.append(next_step)

        window = np.concatenate(
            [window[:, 1:, :], next_step],
            axis=1
        )

    return np.concatenate(forecasts, axis=1)


def metrics_by_horizon(actual, predicted, scale):
    a = actual[:, :, 0] * scale
    p = predicted[:, :, 0] * scale

    rows = []

    for h in (7, 14):
        diff = a[:, :h] - p[:, :h]

        rows.append({
            'horizon_days': h,
            'MAE': float(np.mean(np.abs(diff))),
            'RMSE': float(np.sqrt(np.mean(diff ** 2)))
        })

    return pd.DataFrame(rows)


def forecast_next_14_days(
    model,
    sales,
    scale,
    lookback=LOOKBACK,
    horizon=HORIZON
):
    recent = sales.iloc[-lookback:]

    if len(recent) != lookback or recent.isna().any():
        raise ValueError(
            'Latest history must contain 30 complete consecutive calendar days'
        )

    x = (
        recent.to_numpy(dtype='float32') / scale
    ).reshape(1, lookback, 1)

    prediction = np.maximum(
        model.predict(x, verbose=0)[0, :, 0] * scale,
        0
    )

    future_dates = pd.date_range(
        recent.index[-1] + pd.Timedelta(days=1),
        periods=horizon
    )

    return pd.DataFrame(
        {'Predicted Sales': prediction},
        index=future_dates
    ).rename_axis('Date')
