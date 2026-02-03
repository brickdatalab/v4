# Poly Supabase Project Analysis
**Project ID:** `cxvntzszdkyggjjenefn`
**Region:** us-east-2
**Analysis Date:** February 3, 2026 17:52 UTC
**Status:** ACTIVE_HEALTHY

---

## Executive Summary

**Poly** is a cryptocurrency prediction system for Polymarket betting markets. It predicts BTC, ETH, and SOL price direction (UP/DOWN) at 15-minute and 1-hour intervals using XGBoost models.

| Metric | Value |
|--------|-------|
| **Backtest Accuracy** | 83-85% (HIGH confidence tier) |
| **Live Accuracy** | Recovering (CVD fix applied Feb 3) |
| **Execution Mode** | Manual (Vincent reads signals, places bets) |
| **Active Models** | 6 XGBoost (3 symbols × 2 timeframes) |
| **Data Volume** | ~25M+ rows across all schemas |
| **Cron Jobs** | 11 active, 99.98% success rate |

### Current Blockers
1. **Polymarket API**: 102/102 automated orders failed (API_ERROR_HTML)
2. **Model Accuracy**: Live performance recovering after CVD stale period fix

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION LAYER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Coinbase WebSocket → raw_trades_2026_XX (~19.8M rows)                      │
│                    → market_context (~17.9M rows)                            │
│                    → order_book_snapshots (~58K rows)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (Every minute via cron)
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGGREGATION LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  aggregate_ohlcv_1m() → ohlcv_1m (~76K rows)                                │
│  capture_trade_flow_snapshots() → trade_flow_snapshots (~81K rows)          │
│  Cascaded indicators → indicators.ohlcv_* (10 timeframes)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (Computed in real-time)
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INDICATORS LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  156 indicator configs across 6 categories:                                  │
│  • Moving Averages (17)  • Oscillators (16)  • Trend (24)                   │
│  • Volatility (18)       • Volume (17)       • Complex (13)                 │
│  Weekly partitioned: indicator_values_w2026_XX (~878K rows)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (Computed on-demand)
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FEATURES LAYER (alpha schema)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  alpha.features_15m (~634K rows, 60 columns)                                │
│  alpha.features_1h (~158K rows, 60 columns)                                 │
│  Features: percentile ranks, slopes, binary regimes, cyclical time          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (At :01, :16, :31, :46 via Cloud Run)
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Cloud Run: signal-pipeline-277919876041.us-central1.run.app                │
│  • Loads XGBoost models from alpha.models (6 models)                        │
│  • Computes magnitude (1-10) and confidence tier (HIGH/MED/LOW)             │
│  • Writes to alpha.signals (~378 rows)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (Manual execution)
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXECUTION LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Vincent reads: SELECT * FROM alpha.v_latest_signals                        │
│  Places bets manually on Polymarket (automation blocked by API issues)      │
│  Markets cached in polymarket.markets_live (1,560 markets)                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Schema Analysis

### 1. PUBLIC Schema (Core Data)

**Purpose:** Real-time market data ingestion and aggregation

| Table | Rows | Description |
|-------|------|-------------|
| `raw_trades_2026_XX` | ~19.8M | Partitioned trade data from Coinbase WebSocket |
| `market_context` | ~17.9M | High-frequency ticker data (best bid/ask) |
| `ohlcv_1m` | ~76K | 1-minute OHLCV candles |
| `indicators` | ~75K | Legacy EMA/RSI/MACD calculations |
| `trade_flow_snapshots` | ~81K | Minute-bucketed trade flow metrics |
| `order_book_snapshots` | ~58K | Level 2 order book snapshots |
| `all_readme` | 24 | Master documentation for AI agents |

**Key Functions:**
- `aggregate_ohlcv_1m()` - Aggregates raw trades to 1m candles
- `capture_trade_flow_snapshots()` - Captures buy/sell volume, CVD, imbalance
- `fn_maintain_partitions()` - Manages table partitioning

---

### 2. ALPHA Schema (Production ML System)

**Purpose:** Live XGBoost prediction system for Polymarket trading

| Table | Rows | Description |
|-------|------|-------------|
| `features_15m` | 633,915 | 60-column feature matrix for 15m predictions |
| `features_1h` | 158,437 | 60-column feature matrix for 1h predictions |
| `models` | 6 | XGBoost model registry with validation metrics |
| `signals` | 378 | Live trading signals (direction, magnitude, confidence) |
| `predictions_15m` | 1,832 | Historical 15m predictions |
| `predictions_1h` | 461 | Historical 1h predictions |
| `model_performance` | 0 | Aggregated accuracy metrics (not yet populated) |
| `readme` | 7 | Schema documentation |

**Model Performance (Validation):**

| Model | Accuracy | AUC | HIGH Tier Accuracy |
|-------|----------|-----|-------------------|
| BTC_15m | 75.0% | 0.833 | ~80-84% |
| BTC_1h | 78.6% | 0.873 | ~80-84% |
| ETH_15m | 76.1% | 0.846 | ~80-84% |
| ETH_1h | 79.3% | 0.880 | ~80-84% |
| SOL_15m | 76.9% | 0.856 | ~80-84% |
| SOL_1h | 80.5% | 0.888 | ~80-84% |

**Feature Categories (60 total):**
- **Momentum:** RSI percentiles (20/50/100 periods), RSI slopes (3/5/10)
- **Stochastic:** stoch_k_pct_20, stoch_k_pct_50
- **Trend:** ADX, MACD histogram slopes, supertrend_bullish
- **Volatility:** ATR, Bollinger Band width, price vs BB mid
- **Volume:** OBV, CVD (50-period), MFI, CMF
- **Price Position:** price_vs_sma_20/50, price_vs_ema_21
- **Binary Regimes:** is_oversold, is_overbought, is_bullish_macd, is_bearish_macd
- **Cross-Timeframe:** rsi_15m_vs_1h, trend_alignment_1h, volatility_ratio_1h
- **Temporal:** hour_sin, hour_cos, dow_sin, dow_cos, is_weekend

**Confidence Tier Distribution:**
- **HIGH** (prob > 0.65): ~73-75% of predictions, 80-84% accuracy
- **MED** (prob > 0.58): ~12-13% of predictions, 57-63% accuracy
- **LOW** (rest): ~13-15% of predictions, 49-57% accuracy

**Magnitude Formula:**
```
magnitude = round((abs(prob - 0.5) * 2)^1.3 * 15), clamped [1, 10]
```

---

### 3. INDICATORS Schema (Technical Analysis Engine)

**Purpose:** 156 configurable technical indicators across 10 timeframes

| Table | Rows | Description |
|-------|------|-------------|
| `ohlcv_1m` - `ohlcv_12h` | ~100K | 10 timeframe OHLCV tables |
| `indicator_configs` | 156 | Indicator definitions with JSONB params |
| `indicator_values_w2026_XX` | ~878K | Weekly partitioned computed values |
| `order_book_indicators` | ~59K | Depth ratios, imbalance, slippage |
| `computation_log` | ~9K | Execution audit trail |
| `job_queue` | ~52K | Async calculation workflow |
| `readme` | 37 | Schema documentation |

**Indicator Categories:**
| Category | Count | Examples |
|----------|-------|----------|
| Moving Averages | 17 | SMA, EMA, WMA, HMA, VWMA |
| Oscillators | 16 | RSI, Stochastic, MFI, Williams %R |
| Trend | 24 | ADX, MACD, Aroon, Supertrend, Ichimoku |
| Volatility | 18 | ATR, Bollinger, Keltner, Donchian |
| Volume | 17 | OBV, CVD, CMF, ADL, Force Index |
| Complex | 13 | Pivot Points, Linear Regression, Ultimate |

**OHLCV Timeframes:** 1m, 5m, 10m, 15m, 30m, 45m, 1h, 2h, 6h, 12h

**Key Design Patterns:**
- Directional volume tracking (buy_volume, sell_volume)
- Weekly partitioning for efficient archival
- Config-driven architecture with JSONB parameters
- Async job queue for distributed processing

---

### 4. TRAINING Schema (Historical Data Warehouse)

**Purpose:** 5+ years of historical data for model training

| Table | Rows | Description |
|-------|------|-------------|
| `spot_1m` | 4,557,646 | 1-minute candles (Aug 2017 → Mar 2025) |
| `spot_15m` | 633,984 | 15-minute candles (Sep 2019 → Feb 2026) |
| `spot_1h` | 158,506 | 1-hour candles |
| `spot_15m_indicators` | 633,987 | 140+ pre-calculated indicators |
| `spot_1h_indicators` | 158,506 | 140+ pre-calculated indicators |
| `unified_15m` | 633,981 | ML-ready dataset with labels |
| `unified_1h` | 158,503 | ML-ready dataset with labels |
| `synthetic_features` | 153,730 | Composite engineered signals |
| `readme` | 24 | Schema documentation |

**Data Sources:**
- WinkingFace CryptoLM (1m data from Aug 2017)
- zongowo v2-crypto-ohlcv-data (15m/1h from Sep 2019)

**Model Training Specification:**
```python
XGBoost Hyperparameters:
  max_depth = 4
  min_child_weight = 50
  subsample = 0.8
  learning_rate = 0.05
  early_stopping = 50 rounds

Walk-Forward CV:
  5 folds, 3-month test periods
  1-day gap between train/test
```

**Top 5 Features by Importance:**
1. `rsi_slope_3` - Short-term RSI momentum
2. `is_bearish_macd` - MACD regime flag
3. `stoch_k_pct_20` - Stochastic percentile
4. `is_bullish_macd` - MACD regime flag
5. `rsi_14_pct_20` - RSI percentile

**Label Distribution:**
- 15m candles: 51-57% DOWN bias
- 1h candles: ~50/50 balanced

---

### 5. CRON Schema (Job Scheduling)

**Purpose:** 11 scheduled jobs powering the real-time pipeline

| Job | Schedule | Duration | Success Rate |
|-----|----------|----------|--------------|
| `aggregate-ohlcv-1m` | `* * * * *` | 0.38s | 99.96% |
| `trade_flow_snapshots_every_minute` | `* * * * *` | 22.67s | 99.99% |
| `ensure_15m_indicator_jobs` | `* * * * *` | 0.01s | 100% |
| `ensure_1h_indicator_jobs` | `*/5 * * * *` | 0.01s | 100% |
| `populate-polymarket-tokens` | `*/5 * * * *` | ~0s | 100% |
| `partition_maintenance` | `0 3 * * *` | 33.44s | 100% |
| `polymarket-populate-markets-daily` | `5 8 * * *` | 0.01s | 100% |
| `signal-pipeline-01` | `1 * * * *` | ~0s | 100% |
| `signal-pipeline-16` | `16 * * * *` | ~0s | 100% |
| `signal-pipeline-31` | `31 * * * *` | ~0s | 100% |
| `signal-pipeline-46` | `46 * * * *` | ~0s | 100% |

**Total Historical Runs:** ~89,941
**Overall Success Rate:** 99.98%
**Failed Runs:** 12 (only in high-frequency jobs)

**Signal Pipeline Execution:**
- Cloud Run triggered at :01, :16, :31, :46 each hour
- Distributed to prevent thundering herd
- External service: `signal-pipeline-277919876041.us-central1.run.app`

---

## Known Issues

### 1. CRITICAL: Confidence Inversion (RESOLVED)

**Problem:** During Feb 2 CVD stale period, confidence was inverted:
- HIGH magnitude (8-10): 20-28% accuracy (WORSE than coin flip)
- LOW magnitude (1-4): 75-100% accuracy (EXCELLENT)

**Root Cause:** CVD feature froze at -332.0972 from 04:37-12:20 UTC Feb 2

**Resolution:** Binance volume backfill applied. CVD now dynamic (verified).

**Expected Outcome:** HIGH tier accuracy should return to 83-85%

### 2. Polymarket API Failures (UNRESOLVED)

**Problem:** 102/102 automated orders failed with `no_fill_after_3_attempts: API_ERROR_HTML`

**Impact:** Automated execution blocked; Vincent must place bets manually

**Status:** Under investigation

### 3. Dead Code

The following are deprecated but still running in cron:
- `gpt52-predictions` (deprecated model)
- `sfm-predictions` (deprecated model)
- `smash` and `smash_sfm` tables

---

## Database Extensions

**35 active extensions** including:
- `pg_cron` (1.6.4) - Job scheduling
- `vector` (0.8.0) - ML embeddings
- `http` (1.6) - WebSocket/API integration
- `pg_net` (0.19.5) - Async HTTP requests
- `pg_stat_statements` (1.11) - Query performance
- `postgis` (3.3.7) - Geospatial (if needed)

---

## Quick Reference Queries

**Latest Signals:**
```sql
SELECT * FROM alpha.v_latest_signals ORDER BY created_at DESC;
```

**Recent Accuracy:**
```sql
SELECT confidence_tier,
       COUNT(*) as total,
       SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct,
       ROUND(AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) * 100, 1) as accuracy
FROM alpha.signals
WHERE actual_result IS NOT NULL
GROUP BY confidence_tier;
```

**Accuracy by Magnitude:**
```sql
SELECT magnitude,
       COUNT(*) as total,
       ROUND(AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) * 100, 1) as accuracy
FROM alpha.signals
WHERE actual_result IS NOT NULL
GROUP BY magnitude ORDER BY magnitude;
```

**Data Freshness:**
```sql
SELECT MAX(created_at) as latest FROM public.ohlcv_1m;
SELECT MAX(created_at) as latest FROM alpha.signals;
```

**CVD Health Check:**
```sql
SELECT symbol, open_time, cvd_50
FROM indicators.ohlcv_15m
WHERE open_time > NOW() - INTERVAL '1 hour'
ORDER BY open_time DESC LIMIT 10;
```

**Cron Health:**
```sql
SELECT jobname,
       COUNT(*) as runs,
       SUM(CASE WHEN status = 'succeeded' THEN 1 ELSE 0 END) as succeeded,
       MAX(start_time) as last_run
FROM cron.job_run_details
WHERE start_time > NOW() - INTERVAL '24 hours'
GROUP BY jobname;
```

---

## Summary

Poly is a production-grade crypto prediction system with:
- **Strong backtest performance** (83-85% HIGH tier accuracy)
- **Robust data pipeline** (99.98% cron success rate)
- **Comprehensive feature engineering** (60 features, 6 models)
- **Real-time inference** (signals every 15 minutes)

**Current Priority:** Monitor live accuracy post-CVD fix to validate model performance recovery.
