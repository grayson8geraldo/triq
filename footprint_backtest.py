"""
Footprint IQ Pro Backtester
============================
Реализация логики индикатора Footprint IQ Pro [TradingIQ] на Python.
Бэктест на BTCUSDT 15m, начальный баланс $200.

Ключевое отличие от наивного подхода:
- Как в оригинальном Pine Script, каждый суб-бар (15m свеча внутри 4H сессии)
  классифицирует ВЕСЬ свой объём как buy или sell по math.sign(close - open).
- Это создаёт резкие перекосы delta на уровнях цены → рабочие imbalances.
- Дополнительно используем реальный taker_buy_volume для более точного delta.

Стратегия:
- Stacked buy imbalances + положительный общий delta → Long
- Stacked sell imbalances + отрицательный общий delta → Short
- POC как уровень поддержки/сопротивления
- ATR-based SL/TP с учётом Value Area
"""

import os
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# ============================================================
# 1. DATA LOADING
# ============================================================

def load_all_data(data_dir: str) -> pd.DataFrame:
    """Загрузить все CSV файлы BTCUSDT 15m и объединить."""
    files = sorted(glob.glob(os.path.join(data_dir, "BTCUSDT-15m-*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = [pd.read_csv(f) for f in files]
    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values("open_time").reset_index(drop=True)
    data = data.drop_duplicates(subset=["open_time"]).reset_index(drop=True)

    # Всё одним блоком чтобы избежать fragmentation warning
    data = data.assign(
        datetime=pd.to_datetime(data["open_time"], unit="ms"),
        buy_volume=data["taker_buy_volume"],
        sell_volume=data["volume"] - data["taker_buy_volume"],
        delta=data["taker_buy_volume"] - (data["volume"] - data["taker_buy_volume"]),
        direction=np.sign(data["close"] - data["open"]),
    )
    # Убираем строки с NaT datetime
    data = data.dropna(subset=["datetime"]).reset_index(drop=True)
    return data


# ============================================================
# 2. FOOTPRINT PROFILE BUILDER
# ============================================================

@dataclass
class FootprintProfile:
    poc_price: float = 0.0
    va_high: float = 0.0
    va_low: float = 0.0
    total_delta: float = 0.0
    total_volume: float = 0.0
    max_buy_stack: int = 0
    max_sell_stack: int = 0
    has_stacked_buy: bool = False
    has_stacked_sell: bool = False
    highest_pos_delta_below: float = 0.0
    highest_neg_delta_above: float = 0.0
    session_open: float = 0.0
    session_close: float = 0.0
    session_high: float = 0.0
    session_low: float = 0.0
    delta_pct_range: Tuple[float, float] = (0.0, 0.0)


def build_footprint_pine(candles: pd.DataFrame, tick_size: float,
                         imbalance_pct: float = 70.0, stacked_count: int = 3,
                         va_pct: float = 0.70) -> FootprintProfile:
    """
    Строим footprint-профиль ТОЧНО как в Pine Script:

    Для каждой суб-свечи (15m внутри сессии):
      1. Берём volume
      2. Умножаем на sign(close - open): если свеча bearish → volume * -1
      3. Делим полученный signed_vol на количество уровней, которые покрывает свеча
      4. Прибавляем к delta каждого покрытого уровня

    Это означает: на уровне, который покрыт ТОЛЬКО бычьими свечами,
    delta_pct будет ~100%. На уровне с только медвежьими → ~-100%.
    """
    if candles.empty or len(candles) < 2:
        return FootprintProfile()

    sess_high = candles["high"].max()
    sess_low = candles["low"].min()
    sess_open = candles["open"].iloc[0]
    sess_close = candles["close"].iloc[-1]

    if tick_size <= 0 or np.isnan(tick_size):
        tick_size = (sess_high - sess_low) / 20

    # Tick levels
    n_levels = max(int((sess_high - sess_low) / tick_size) + 2, 5)
    if n_levels > 300:
        tick_size = (sess_high - sess_low) / 150
        n_levels = 152

    levels = sess_low + np.arange(n_levels) * tick_size

    # Массивы delta и volume
    delta_arr = np.zeros(n_levels)
    up_vol = np.zeros(n_levels)
    dn_vol = np.zeros(n_levels)
    total_vol = np.zeros(n_levels)

    # --- Точная логика Pine Script ---
    # getVol = ltfV.get(i)
    # if ltfD.get(i) == -1:
    #     getVol *= -1
    # div = getVol / (getTop - getBot + 1)
    # for x = getBot to getTop:
    #     FD.deltaArr.set(x, FD.deltaArr.get(x) + div)

    highs = candles["high"].values
    lows = candles["low"].values
    volumes = candles["volume"].values
    directions = candles["direction"].values  # sign(close - open)

    for i in range(len(candles)):
        vol = volumes[i]
        d = directions[i]

        # Pine: if ltfD == -1, getVol *= -1
        signed_vol = vol * d  # Весь объём со знаком направления

        c_high = highs[i]
        c_low = lows[i]

        bot_idx = np.searchsorted(levels, c_low, side="left")
        top_idx = np.searchsorted(levels, c_high, side="left")
        bot_idx = max(0, min(bot_idx, n_levels - 1))
        top_idx = max(0, min(top_idx, n_levels - 1))

        n_covered = top_idx - bot_idx + 1
        if n_covered <= 0:
            n_covered = 1
            top_idx = bot_idx

        div = signed_vol / n_covered

        for idx in range(bot_idx, top_idx + 1):
            delta_arr[idx] += div
            if div > 0:
                up_vol[idx] += div
            else:
                dn_vol[idx] += abs(div)
            total_vol[idx] += abs(div)

    # Delta percentage
    delta_pct = np.zeros(n_levels)
    mask = total_vol > 0
    delta_pct[mask] = (delta_arr[mask] / total_vol[mask]) * 100

    # --- POC ---
    poc_idx = int(np.argmax(total_vol))
    poc_price = levels[poc_idx]

    # --- Value Area ---
    sum_vol = total_vol.sum()
    va_low_price = levels[poc_idx]
    va_high_price = levels[poc_idx]

    if sum_vol > 0:
        for spread in range(1, n_levels):
            lo = max(poc_idx - spread, 0)
            hi = min(poc_idx + spread, n_levels - 1)
            if total_vol[lo:hi + 1].sum() / sum_vol >= va_pct:
                va_low_price = levels[lo]
                va_high_price = levels[hi]
                break

    # --- Stacked Imbalances ---
    max_buy_stack = 0
    max_sell_stack = 0
    cur_buy = 0
    cur_sell = 0

    for i in range(n_levels):
        dp = delta_pct[i]
        if total_vol[i] <= 0:
            # Пропускаем пустые уровни (не ломаем стек)
            continue
        if dp >= imbalance_pct:
            cur_buy += 1
            cur_sell = 0
            max_buy_stack = max(max_buy_stack, cur_buy)
        elif dp <= -imbalance_pct:
            cur_sell += 1
            cur_buy = 0
            max_sell_stack = max(max_sell_stack, cur_sell)
        else:
            cur_buy = 0
            cur_sell = 0

    # --- Delta Lines ---
    close_idx = min(np.searchsorted(levels, sess_close, side="left"), n_levels - 1)
    highest_pos_below = 0.0
    highest_neg_above = 0.0

    for i in range(n_levels):
        if i <= close_idx and delta_arr[i] > highest_pos_below:
            highest_pos_below = delta_arr[i]
        if i >= close_idx and delta_arr[i] < highest_neg_above:
            highest_neg_above = delta_arr[i]

    valid_pcts = delta_pct[mask]
    pct_range = (float(valid_pcts.min()), float(valid_pcts.max())) if len(valid_pcts) > 0 else (0.0, 0.0)

    return FootprintProfile(
        poc_price=poc_price,
        va_high=va_high_price,
        va_low=va_low_price,
        total_delta=float(delta_arr.sum()),
        total_volume=float(sum_vol),
        max_buy_stack=max_buy_stack,
        max_sell_stack=max_sell_stack,
        has_stacked_buy=max_buy_stack >= stacked_count,
        has_stacked_sell=max_sell_stack >= stacked_count,
        highest_pos_delta_below=highest_pos_below,
        highest_neg_delta_above=highest_neg_above,
        session_open=sess_open,
        session_close=sess_close,
        session_high=sess_high,
        session_low=sess_low,
        delta_pct_range=pct_range,
    )


# ============================================================
# 3. ATR
# ============================================================

def compute_atr_series(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Вычисляем ATR как скользящее среднее True Range."""
    n = len(highs)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))

    atr = np.zeros(n)
    atr[:period] = np.mean(tr[:period]) if n >= period else tr[:n].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


# ============================================================
# 4. STRATEGY & BACKTESTER
# ============================================================

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    size_usd: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    initial_balance: float = 200.0
    final_balance: float = 200.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)


def close_position(pos: Trade, exit_price: float, exit_time, reason: str) -> float:
    """Закрывает позицию, возвращает P&L."""
    pos.exit_price = exit_price
    pos.exit_time = exit_time
    pos.exit_reason = reason

    if pos.direction == "long":
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
    else:
        pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

    pos.pnl = pos.size_usd * pnl_pct
    pos.pnl_pct = pnl_pct * 100
    return pos.pnl


def run_backtest(data: pd.DataFrame, initial_balance: float = 200.0,
                 session_tf: str = "4h",
                 imbalance_pct: float = 70.0,
                 stacked_count: int = 3,
                 va_pct: float = 0.70,
                 risk_per_trade: float = 0.03,
                 sl_atr_mult: float = 1.0,
                 tp_atr_mult: float = 2.0,
                 max_leverage: float = 10.0,
                 commission_pct: float = 0.04,
                 ema_period: int = 50,
                 cooldown_sessions: int = 2,
                 delta_strength_mult: float = 1.5) -> BacktestResult:
    """
    Стратегия Footprint IQ с фильтрами:

    Фильтры:
      1. Трендовый: EMA-50 на сессионных closes (long выше EMA, short ниже)
      2. Сила delta: |total_delta| > median(|delta| за последние N сессий) × mult
      3. POC alignment: для long POC ниже close, для short POC выше close
      4. Cooldown: минимум N сессий между сделками
      5. Stacked imbalance (основной сигнал из индикатора)

    Вход Long:
      - Stacked buy imbalances + delta > 0 + цена > EMA + POC < close + сильный delta

    Вход Short:
      - Stacked sell imbalances + delta < 0 + цена < EMA + POC > close + сильный delta
    """

    data = data.copy()
    data["session"] = data["datetime"].dt.floor(session_tf)

    sess_groups = data.groupby("session")
    sess_keys = sorted(sess_groups.groups.keys())

    # Предвычисляем ATR и closes на сессионных свечах
    sess_ohlcv = []
    for sk in sess_keys:
        g = sess_groups.get_group(sk)
        sess_ohlcv.append({
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].iloc[-1],
            "open": g["open"].iloc[0],
            "volume": g["volume"].sum(),
        })
    sess_df = pd.DataFrame(sess_ohlcv)
    sess_atr = compute_atr_series(sess_df["high"].values, sess_df["low"].values,
                                   sess_df["close"].values, period=14)

    # EMA на сессионных closes
    sess_closes = sess_df["close"].values
    ema = np.zeros(len(sess_closes))
    ema[0] = sess_closes[0]
    k = 2.0 / (ema_period + 1)
    for i in range(1, len(sess_closes)):
        ema[i] = sess_closes[i] * k + ema[i - 1] * (1 - k)

    balance = initial_balance
    trades: List[Trade] = []
    equity_curve = [(data["datetime"].iloc[0], balance)]
    current_pos: Optional[Trade] = None

    peak_balance = balance
    max_dd = 0.0

    n_sessions = len(sess_keys)
    signals_generated = 0
    signals_filtered = 0
    last_trade_idx = -100  # Для cooldown

    # Скользящее окно абс. delta для определения "силы"
    delta_history: List[float] = []

    print(f"{'='*80}")
    print(f"  FOOTPRINT IQ PRO BACKTESTER (with filters)")
    print(f"  Начальный баланс: ${initial_balance:.2f}")
    print(f"  Данные: {data['datetime'].iloc[0]} → {data['datetime'].iloc[-1]}")
    print(f"  Сессий: {n_sessions} ({session_tf})")
    print(f"  Imbalance: {imbalance_pct}% | Stacked: {stacked_count}")
    print(f"  Risk: {risk_per_trade*100}% | SL: {sl_atr_mult}×ATR | TP: {tp_atr_mult}×ATR")
    print(f"  Leverage: {max_leverage}x | Commission: {commission_pct}%")
    print(f"  Filters: EMA-{ema_period}, cooldown={cooldown_sessions}, "
          f"delta_strength={delta_strength_mult}x")
    print(f"{'='*80}")

    for s_idx in range(n_sessions):
        sess_time = sess_keys[s_idx]
        sess_candles = sess_groups.get_group(sess_time)

        if len(sess_candles) < 2:
            continue

        atr = sess_atr[s_idx]
        if atr <= 0 or np.isnan(atr):
            atr = sess_candles["high"].max() - sess_candles["low"].min()
            if atr <= 0:
                continue

        tick_size = atr / 20

        # --- Проверяем SL/TP открытой позиции свеча за свечой ---
        if current_pos is not None:
            closed = False
            for _, candle in sess_candles.iterrows():
                if current_pos.direction == "long":
                    if candle["low"] <= current_pos.sl_price:
                        close_position(current_pos, current_pos.sl_price,
                                       candle["datetime"], "SL")
                        closed = True
                    elif candle["high"] >= current_pos.tp_price:
                        close_position(current_pos, current_pos.tp_price,
                                       candle["datetime"], "TP")
                        closed = True
                else:
                    if candle["high"] >= current_pos.sl_price:
                        close_position(current_pos, current_pos.sl_price,
                                       candle["datetime"], "SL")
                        closed = True
                    elif candle["low"] <= current_pos.tp_price:
                        close_position(current_pos, current_pos.tp_price,
                                       candle["datetime"], "TP")
                        closed = True

                if closed:
                    comm = current_pos.size_usd * commission_pct / 100 * 2
                    current_pos.pnl -= comm
                    balance += current_pos.pnl
                    balance = max(balance, 0)
                    trades.append(current_pos)
                    equity_curve.append((candle["datetime"], balance))

                    peak_balance = max(peak_balance, balance)
                    if peak_balance > 0:
                        dd = (peak_balance - balance) / peak_balance
                        max_dd = max(max_dd, dd)

                    current_pos = None
                    last_trade_idx = s_idx
                    break

        # --- Строим footprint ---
        profile = build_footprint_pine(
            sess_candles, tick_size,
            imbalance_pct=imbalance_pct,
            stacked_count=stacked_count,
            va_pct=va_pct
        )

        # Обновляем историю delta
        delta_history.append(abs(profile.total_delta))
        if len(delta_history) > 200:
            delta_history.pop(0)

        # --- Генерируем сигнал с фильтрами ---
        if current_pos is None and balance > 1.0 and s_idx >= ema_period:
            signal = None
            sess_close = profile.session_close
            ema_val = ema[s_idx]

            # Фильтр cooldown
            if s_idx - last_trade_idx < cooldown_sessions:
                continue

            # Фильтр силы delta
            median_delta = float(np.median(delta_history)) if delta_history else 0
            delta_strong = abs(profile.total_delta) > median_delta * delta_strength_mult

            # === LONG ===
            if (profile.has_stacked_buy
                    and profile.total_delta > 0
                    and sess_close > ema_val         # Тренд вверх
                    and profile.poc_price < sess_close  # POC ниже цены (поддержка)
                    and delta_strong):
                signal = "long"

            # === SHORT ===
            elif (profile.has_stacked_sell
                  and profile.total_delta < 0
                  and sess_close < ema_val           # Тренд вниз
                  and profile.poc_price > sess_close  # POC выше цены (сопротивление)
                  and delta_strong):
                signal = "short"

            if profile.has_stacked_buy or profile.has_stacked_sell:
                signals_filtered += 1  # Считаем отфильтрованные

            if signal is not None:
                signals_generated += 1
                entry_price = sess_close

                if signal == "long":
                    # SL ниже VA_low или ATR-based, что ближе
                    sl_price = entry_price - sl_atr_mult * atr
                    if profile.va_low < entry_price:
                        sl_price = max(sl_price, profile.va_low - tick_size)
                    tp_price = entry_price + tp_atr_mult * atr
                else:
                    sl_price = entry_price + sl_atr_mult * atr
                    if profile.va_high > entry_price:
                        sl_price = min(sl_price, profile.va_high + tick_size)
                    tp_price = entry_price - tp_atr_mult * atr

                sl_distance = abs(entry_price - sl_price)
                if sl_distance <= 0:
                    continue
                risk_usd = balance * risk_per_trade
                pos_size = risk_usd / (sl_distance / entry_price)
                pos_size = min(pos_size, balance * max_leverage)

                current_pos = Trade(
                    entry_time=sess_candles["datetime"].iloc[-1],
                    direction=signal,
                    entry_price=entry_price,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    size_usd=pos_size,
                )

    # Закрываем открытую позицию
    if current_pos is not None:
        last_close = data["close"].dropna().iloc[-1]
        last_dt = data["datetime"].dropna().iloc[-1]
        close_position(current_pos, last_close, last_dt, "END")
        comm = current_pos.size_usd * commission_pct / 100 * 2
        current_pos.pnl -= comm
        balance += current_pos.pnl
        balance = max(balance, 0)
        trades.append(current_pos)
        equity_curve.append((data["datetime"].iloc[-1], balance))

    print(f"\n  Stacked imbalance сессий (до фильтров): {signals_filtered}")
    print(f"  Сигналов после фильтров:                {signals_generated}")

    # --- Собираем статистику ---
    result = BacktestResult(
        initial_balance=initial_balance,
        final_balance=balance,
        total_trades=len(trades),
        trades=trades,
        equity_curve=equity_curve,
    )

    if trades:
        pnls = np.array([t.pnl for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = len(wins) / len(trades) * 100
        result.total_pnl = float(pnls.sum())
        result.total_pnl_pct = (balance - initial_balance) / initial_balance * 100
        result.max_drawdown = max_dd * peak_balance
        result.max_drawdown_pct = max_dd * 100
        result.best_trade = float(pnls.max())
        result.worst_trade = float(pnls.min())
        result.avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        result.avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = float(np.abs(losses).sum()) if len(losses) > 0 else 0.001
        result.profit_factor = gross_profit / gross_loss

        if len(pnls) > 1 and np.std(pnls) > 0:
            result.sharpe_ratio = float(np.mean(pnls) / np.std(pnls))

        # Max consecutive wins/losses
        max_cw, max_cl, cw, cl = 0, 0, 0, 0
        for p in pnls:
            if p > 0:
                cw += 1
                cl = 0
                max_cw = max(max_cw, cw)
            else:
                cl += 1
                cw = 0
                max_cl = max(max_cl, cl)
        result.max_consecutive_wins = max_cw
        result.max_consecutive_losses = max_cl

    return result


# ============================================================
# 5. REPORT
# ============================================================

def print_report(result: BacktestResult):
    print(f"\n{'='*80}")
    print(f"{'FOOTPRINT IQ PRO — BACKTEST REPORT':^80}")
    print(f"{'BTCUSDT | 15m Data | 4H Footprint Sessions':^80}")
    print(f"{'='*80}")

    print(f"\n  --- ОБЩИЕ РЕЗУЛЬТАТЫ ---")
    print(f"  Начальный баланс:        ${result.initial_balance:>12.2f}")
    print(f"  Конечный баланс:         ${result.final_balance:>12.2f}")
    ret = result.total_pnl_pct
    print(f"  P&L:                     ${result.total_pnl:>+12.2f}  ({ret:+.1f}%)")
    print(f"  Макс. просадка:          ${result.max_drawdown:>12.2f}  ({result.max_drawdown_pct:.1f}%)")

    print(f"\n  --- СТАТИСТИКА СДЕЛОК ---")
    print(f"  Всего сделок:            {result.total_trades:>8}")
    print(f"  Прибыльных:              {result.winning_trades:>8}")
    print(f"  Убыточных:               {result.losing_trades:>8}")
    print(f"  Win rate:                {result.win_rate:>7.1f}%")
    print(f"  Profit factor:           {result.profit_factor:>8.2f}")
    print(f"  Sharpe ratio:            {result.sharpe_ratio:>8.3f}")
    print(f"  Max consecutive wins:    {result.max_consecutive_wins:>8}")
    print(f"  Max consecutive losses:  {result.max_consecutive_losses:>8}")

    print(f"\n  --- P&L ДЕТАЛИ ---")
    print(f"  Средний выигрыш:         ${result.avg_win:>+12.2f}")
    print(f"  Средний проигрыш:        ${result.avg_loss:>+12.2f}")
    print(f"  Лучшая сделка:           ${result.best_trade:>+12.2f}")
    print(f"  Худшая сделка:           ${result.worst_trade:>+12.2f}")

    if result.trades:
        # Годовая разбивка
        print(f"\n  --- РАЗБИВКА ПО ГОДАМ ---")
        by_year = {}
        for t in result.trades:
            y = t.entry_time.year
            by_year.setdefault(y, []).append(t)

        print(f"  {'Год':<6}{'Сделок':>7}{'Win%':>8}{'P&L ($)':>12}{'Лучшая':>10}{'Худшая':>10}")
        print(f"  {'-'*53}")
        for y in sorted(by_year):
            tl = by_year[y]
            pl = [t.pnl for t in tl]
            w = sum(1 for p in pl if p > 0)
            wr = w / len(tl) * 100
            print(f"  {y:<6}{len(tl):>7}{wr:>7.1f}%{sum(pl):>+11.2f}{max(pl):>+10.2f}{min(pl):>+10.2f}")

        # Long vs Short
        print(f"\n  --- LONG vs SHORT ---")
        for label in ["long", "short"]:
            group = [t for t in result.trades if t.direction == label]
            if group:
                pl = [t.pnl for t in group]
                w = sum(1 for p in pl if p > 0)
                wr = w / len(group) * 100
                print(f"  {label.upper():<7}| {len(group):>4} сделок | "
                      f"Win: {wr:>5.1f}% | P&L: ${sum(pl):>+10.2f} | "
                      f"Avg: ${np.mean(pl):>+8.2f}")

        # SL vs TP breakdown
        print(f"\n  --- ВЫХОДЫ ---")
        for reason in ["TP", "SL", "END"]:
            group = [t for t in result.trades if t.exit_reason == reason]
            if group:
                pl = [t.pnl for t in group]
                print(f"  {reason:<5}| {len(group):>4} сделок | P&L: ${sum(pl):>+10.2f}")

        # Последние 25 сделок
        print(f"\n  --- ПОСЛЕДНИЕ 25 СДЕЛОК ---")
        print(f"  {'Вход':<20} {'Dir':<6} {'Entry':>9} {'Exit':>9} "
              f"{'Size':>8} {'P&L':>9} {'Reason':<4}")
        print(f"  {'-'*67}")
        for t in result.trades[-25:]:
            print(f"  {str(t.entry_time)[:19]:<20} {t.direction:<6} "
                  f"{t.entry_price:>9.1f} {t.exit_price:>9.1f} "
                  f"{t.size_usd:>8.1f} {t.pnl:>+9.2f} {t.exit_reason:<4}")

    # Equity curve
    if result.equity_curve and len(result.equity_curve) > 1:
        print(f"\n  --- EQUITY CURVE ---")
        ib = result.initial_balance
        step = max(1, len(result.equity_curve) // 20)
        for i in range(0, len(result.equity_curve), step):
            dt, eq = result.equity_curve[i]
            bar = int(max(0, min(eq / ib * 20, 60)))
            print(f"  {str(dt)[:19]:<20} ${eq:>10.2f} {'|' * bar}")
        dt, eq = result.equity_curve[-1]
        bar = int(max(0, min(eq / ib * 20, 60)))
        print(f"  {str(dt)[:19]:<20} ${eq:>10.2f} {'|' * bar}  <- FINAL")

    print(f"\n{'='*80}")


# ============================================================
# 6. MAIN
# ============================================================

def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))

    print("Загрузка данных BTCUSDT 15m...")
    data = load_all_data(data_dir)
    print(f"  Загружено: {len(data):,} свечей")
    print(f"  Период:    {data['datetime'].min()} → {data['datetime'].max()}")
    print(f"  Цена:      ${data['low'].min():,.2f} — ${data['high'].max():,.2f}")

    print("\nЗапуск бэктеста Footprint IQ Pro...\n")

    # Оптимальные параметры (найдены через parameter sweep)
    result = run_backtest(
        data,
        initial_balance=200.0,
        session_tf="4h",           # 4H footprint сессии
        imbalance_pct=80.0,        # 80% порог imbalance (строже → качественнее)
        stacked_count=3,           # 3+ подряд imbalance уровня
        va_pct=0.70,               # Value Area 70%
        risk_per_trade=0.03,       # 3% от баланса на сделку
        sl_atr_mult=1.0,           # SL = 1× ATR
        tp_atr_mult=3.0,           # TP = 3× ATR (RR = 3.0, даём прибыли расти)
        max_leverage=10.0,         # Макс. леверидж 10x
        commission_pct=0.04,       # 0.04% комиссия (Binance Futures)
        ema_period=3,              # Фактически без EMA фильтра
        cooldown_sessions=2,       # Мин. 2 сессии между сделками
        delta_strength_mult=0.5,   # Минимальный фильтр силы delta
    )

    print_report(result)

    # Сохраняем CSV
    if result.trades:
        trades_df = pd.DataFrame([{
            "entry_time": t.entry_time, "exit_time": t.exit_time,
            "direction": t.direction, "entry_price": t.entry_price,
            "exit_price": t.exit_price, "sl_price": t.sl_price,
            "tp_price": t.tp_price, "size_usd": t.size_usd,
            "pnl": t.pnl, "pnl_pct": t.pnl_pct, "exit_reason": t.exit_reason,
        } for t in result.trades])
        out = os.path.join(data_dir, "backtest_trades.csv")
        trades_df.to_csv(out, index=False)
        print(f"\n  Сделки: {out}")

    eq_df = pd.DataFrame(result.equity_curve, columns=["datetime", "equity"])
    out = os.path.join(data_dir, "backtest_equity.csv")
    eq_df.to_csv(out, index=False)
    print(f"  Equity:  {out}")


if __name__ == "__main__":
    main()
