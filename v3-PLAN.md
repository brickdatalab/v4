# ALPHA-v4 IMPLEMENTATION INSTRUCTIONS

---

## PHASE 1: SCHEMA SETUP

### Create Schema Structure

**CREATE schema `alpha_v4`**

**CREATE table `alpha_v4.models`**
- Columns: `id` (serial PRIMARY KEY), `model_name` (text), `symbol` (text), `timeframe` (text), `model_type` (text), `version` (text), `train_start` (timestamptz), `train_end` (timestamptz), `train_samples` (integer), `val_accuracy` (numeric), `val_precision` (numeric), `val_recall` (numeric), `val_auc` (numeric), `live_accuracy` (numeric), `live_samples` (integer), `feature_importance` (jsonb), `hyperparameters` (jsonb), `model_binary` (bytea), `is_active` (boolean), `created_at` (timestamptz), `activated_at` (timestamptz)

**CREATE table `alpha_v4.raw_features_15m`**
- Columns: `symbol` (text), `bucket_time` (timestamptz), `open` (numeric), `high` (numeric), `low` (numeric), `close` (numeric), `volume` (numeric), `plus 150 technical indicator columns` (all numeric)
- Primary Key: `symbol, bucket_time`

**CREATE table `alpha_v4.raw_features_1h`**
- Same structure as 15m table
- Primary Key: `symbol, bucket_time`

**CREATE table `alpha_v4.features_15m`**
- Columns: `symbol` (text), `bucket_time` (timestamptz), `rsi_14_pct_20` through `rsi_14_pct_100` (numeric), `stoch_k_pct_20` through `stoch_k_pct_100` (numeric), `cci_20_pct_50` through `cci_20_pct_100` (numeric), `mfi_14_pct_50` through `mfi_14_pct_100` (numeric), `williams_r_pct_50` through `williams_r_pct_100` (numeric), `adx_14_pct_50` through `adx_14_pct_100` (numeric), `atr_14_pct_50` through `atr_14_pct_100` (numeric), `bb_width_pct_50` through `bb_width_pct_100` (numeric), `obv_pct_50` through `obv_pct_100` (numeric), `cvd_50_pct_50` through `cvd_50_pct_100` (numeric), `cmf_20_pct_50` through `cmf_20_pct_100` (numeric), `price_vs_sma_20` (numeric), `price_vs_sma_50` (numeric), `price_vs_ema_21` (numeric), `price_vs_bb_mid` (numeric), `price_bb_position` (numeric), `rsi_slope_3` through `rsi_slope_20` (numeric), `macd_hist_slope_3` through `macd_hist_slope_10` (numeric), `adx_slope_5` through `adx_slope_20` (numeric), `atr_slope_5` through `atr_slope_20` (numeric), `price_momentum_3` through `price_momentum_20` (numeric), `obv_slope_5` through `obv_slope_20` (numeric), `cvd_slope_5` through `cvd_slope_20` (numeric), `is_oversold` (integer), `is_overbought` (integer), `is_extreme_oversold` (integer), `is_extreme_overbought` (integer), `is_low_volatility` (integer), `is_high_volatility` (integer), `is_strong_trend` (integer), `is_weak_trend` (integer), `is_bullish_macd` (integer), `is_bearish_macd` (integer), `is_above_sma_20` (integer), `is_above_sma_50` (integer), `is_above_ema_21` (integer), `supertrend_bullish` (integer), `hour_sin` (numeric), `hour_cos` (numeric), `dow_sin` (numeric), `dow_cos` (numeric), `is_weekend` (integer), `trend_1h_alignment` (integer), `volatility_ratio_1h` (numeric), `rsi_15m_vs_1h` (numeric), `next_label` (text), `next_pct_change` (numeric), `next_label_binary` (integer)
- Primary Key: `symbol, bucket_time`

**CREATE table `alpha_v4.features_1h`**
- Same structure as features_15m
- Primary Key: `symbol, bucket_time`

**CREATE table `alpha_v4.synthetic_indicators`**
- Columns: `symbol` (text), `bucket_time` (timestamptz), `syn_trend_strength` (numeric), `syn_volatility_regime` (numeric), `syn_market_structure` (text), `syn_momentum_divergence` (integer), `syn_volume_anomaly` (integer), `syn_support_proximity` (numeric), `syn_resistance_proximity` (numeric), `syn_btc_correlation_20` (numeric), `syn_eth_correlation_20` (numeric), `syn_macro_sentiment` (numeric), `syn_liquidation_zone` (integer), `syn_order_imbalance` (numeric), `syn_vwap_deviation` (numeric), `syn_session_bias` (text)
- Primary Key: `symbol, bucket_time`

**CREATE table `alpha_v4.labels`**
- Columns: `symbol` (text), `bucket_time` (timestamptz), `timeframe` (text), `label` (text), `label_binary` (integer), `pct_change` (numeric), `magnitude` (integer), `is_significant` (integer), `created_at` (timestamptz)
- Primary Key: `symbol, bucket_time, timeframe`

**CREATE table `alpha_v4.predictions`**
- Columns: `id` (serial PRIMARY KEY), `symbol` (text), `timeframe` (text), `event_start` (timestamptz), `event_end` (timestamptz), `prediction` (text), `direction_binary` (integer), `ensemble_probability` (numeric), `xgb_probability` (numeric), `lgb_probability` (numeric), `nn_probability` (numeric), `model_agreement` (integer), `confidence_score` (numeric), `confidence_tier` (text), `position_size` (integer), `expected_value` (numeric), `kelly_fraction` (numeric), `actual_result` (text), `is_correct` (boolean), `result_display` (text), `start_price` (numeric), `end_price` (numeric), `pct_change` (numeric), `created_at` (timestamptz), `resolved_at` (timestamptz)

**CREATE table `alpha_v4.performance_log`**
- Columns: `id` (serial PRIMARY KEY), `symbol` (text), `timeframe` (text), `window_start` (timestamptz), `window_end` (timestamptz), `predictions_count` (integer), `correct_count` (integer), `accuracy` (numeric), `avg_confidence` (numeric), `calibration_error` (numeric), `sharpe_ratio` (numeric), `max_drawdown` (numeric), `logged_at` (timestamptz)

---

## PHASE 2: LABEL ENGINEERING

### Create Multi-Horizon Labels

**For 15m timeframe, create 4 labels:**
- `label_15m`: Direction of next 15m candle (UP if close > open, DOWN if close < open)
- `label_30m`: Direction of next 30m aggregate (UP if 30m close > current open, DOWN otherwise)
- `label_60m`: Direction of next 60m aggregate
- `label_magnitude`: Integer 1-10 based on absolute percentage move (0-0.1%=1, 0.1-0.3%=2, 0.3-0.5%=3, 0.5-0.8%=4, 0.8-1.2%=5, 1.2-1.8%=6, 1.8-2.5%=7, 2.5-3.5%=8, 3.5-5%=9, >5%=10)

**For 1h timeframe, create 3 labels:**
- `label_1h`: Direction of next 1h candle
- `label_4h`: Direction of next 4h aggregate
- `label_magnitude`: Same 1-10 scale adjusted for hourly volatility

**Create significance filter:**
- `is_significant` = 1 if absolute pct_change > 0.05% for 15m, >0.15% for 1h
- `is_significant` = 0 otherwise

---

## PHASE 3: FEATURE ENGINEERING

### Base Feature Requirements

**Compute 100 percentile-ranked features per indicator:**
- For each raw indicator (RSI, Stoch, CCI, MFI, Williams %R, ADX, ATR, BB Width, OBV, CVD, CMF), calculate percentile over 20, 50, 100 lookback periods
- Store as `indicator_pct_20`, `indicator_pct_50`, `indicator_pct_100`

**Compute 50 slope features:**
- Calculate 3, 5, 10, 15, 20-period slopes for RSI, MACD hist, ADX, ATR, price, OBV, CVD using linear regression
- Store as `indicator_slope_period`

**Compute 30 momentum features:**
- Calculate 3, 5, 10, 15, 20-period price momentum (pct change)
- Calculate 3, 5, 10-period volume momentum

**Compute 20 regime features:**
- `is_oversold`: RSI < 30
- `is_overbought`: RSI > 70
- `is_extreme_oversold`: RSI < 20 AND price < BB lower
- `is_extreme_overbought`: RSI > 80 AND price > BB upper
- `is_low_volatility`: ATR percentile < 20
- `is_high_volatility`: ATR percentile > 80
- `is_strong_trend`: ADX > 25
- `is_weak_trend`: ADX < 15
- `is_bullish_macd`: MACD line > signal AND hist increasing
- `is_bearish_macd`: MACD line < signal AND hist decreasing
- `is_above_sma_20`: close > SMA 20
- `is_above_sma_50`: close > SMA 50
- `is_above_ema_21`: close > EMA 21
- `supertrend_bullish`: close > Supertrend line

**Compute 8 temporal features:**
- `hour_sin`: sin(2 * pi * hour / 24)
- `hour_cos`: cos(2 * pi * hour / 24)
- `dow_sin`: sin(2 * pi * day_of_week / 7)
- `dow_cos`: cos(2 * pi * day_of_week / 7)
- `is_weekend`: 1 if Saturday or Sunday, 0 otherwise
- `is_asian_session`: 1 if hour between 0-8 UTC
- `is_london_session`: 1 if hour between 8-16 UTC
- `is_ny_session`: 1 if hour between 14-22 UTC

### Cross-Timeframe Features

**For 15m model, add 1h context:**
- `trend_1h_alignment`: 1 if 15m direction matches 1h direction, 0 otherwise
- `volatility_ratio_1h`: 15m ATR / 1h ATR
- `rsi_15m_vs_1h`: RSI percentile difference (15m - 1h)
- `price_vs_1h_vwap`: (close - 1h VWAP) / 1h ATR
- `is_1h_bullish`: 1 if 1h close > 1h EMA 21, 0 otherwise

**For 1h model, add 4h and daily context:**
- `trend_4h_alignment`: 1 if 1h direction matches 4h direction
- `trend_daily_alignment`: 1 if 1h direction matches daily direction
- `volatility_ratio_4h`: 1h ATR / 4h ATR
- `rsi_1h_vs_4h`: RSI percentile difference (1h - 4h)
- `price_vs_daily_sma_20`: (close - daily SMA 20) / daily ATR

### Synthetic Indicators

**Create `syn_trend_strength`:**
- Composite score 0-100 based on ADX, price vs MAs, MACD alignment, slope consistency
- Formula: (ADX_normalized * 0.3) + (MA_alignment * 0.3) + (MACD_strength * 0.2) + (slope_consistency * 0.2)

**Create `syn_volatility_regime`:**
- 0 = low vol (ATR percentile < 30), 1 = normal, 2 = high vol (ATR percentile > 70)

**Create `syn_market_structure`:**
- "bull" if price > SMA 50 AND SMA 20 > SMA 50 AND ADX > 20
- "bear" if price < SMA 50 AND SMA 20 < SMA 50 AND ADX > 20
- "range" otherwise

**Create `syn_momentum_divergence`:**
- 1 if price making higher high but RSI making lower high (bearish divergence)
- -1 if price making lower low but RSI making higher low (bullish divergence)
- 0 otherwise

**Create `syn_volume_anomaly`:**
- 1 if current volume > 2 * average volume of last 20 periods
- 0 otherwise

**Create `syn_support_proximity`:**
- Distance to nearest support level (pivot low, BB lower, recent low) normalized by ATR
- Lower values = closer to support

**Create `syn_resistance_proximity`:**
- Distance to nearest resistance level normalized by ATR
- Lower values = closer to resistance

**Create `syn_btc_correlation_20`:**
- Rolling 20-period correlation between asset returns and BTC returns

**Create `syn_eth_correlation_20`:**
- Rolling 20-period correlation between asset returns and ETH returns

**Create `syn_macro_sentiment`:**
- Composite of DXY trend, SPY trend, VIX level normalized to -1 to 1 scale

**Create `syn_liquidation_zone`:**
- 1 if price is within 1 ATR of recent significant high/low (potential liquidation cascade)
- 0 otherwise

**Create `syn_order_imbalance`:**
- Estimate based on candle characteristics (wick ratio, close location, volume profile)
- Positive = buying pressure, negative = selling pressure

**Create `syn_vwap_deviation`:**
- (close - VWAP) / standard deviation of price around VWAP

**Create `syn_session_bias`:**
- "bull" if Asian session low held and London/NY breaking highs
- "bear" if Asian session high held and London/NY breaking lows
- "neutral" otherwise

---

## PHASE 4: MODEL ENSEMBLE ARCHITECTURE

### Model 1: XGBoost

**Hyperparameters:**
- `n_estimators`: 1000
- `max_depth`: 3
- `learning_rate`: 0.01
- `subsample`: 0.7
- `colsample_bytree`: 0.7
- `colsample_bylevel`: 0.7
- `min_child_weight`: 100
- `reg_alpha`: 0.5
- `reg_lambda`: 2.0
- `gamma`: 0.1
- `objective`: binary:logistic
- `eval_metric`: logloss
- `early_stopping_rounds`: 100

**Training:**
- Use 5-fold purged cross-validation with 48-hour embargo between train/test
- Train on data from 2021-01-01 to 2025-06-01
- Validate on 2025-06-01 to 2025-09-01
- Test on 2025-09-01 to 2026-01-01

### Model 2: LightGBM

**Hyperparameters:**
- `n_estimators`: 1000
- `max_depth`: 5
- `learning_rate`: 0.01
- `num_leaves`: 31
- `subsample`: 0.7
- `colsample_bytree`: 0.7
- `min_child_samples`: 100
- `reg_alpha`: 0.5
- `reg_lambda`: 2.0
- `objective`: binary
- `metric`: binary_logloss
- `early_stopping_rounds`: 100

**Training:**
- Same CV strategy as XGBoost
- Use different random seed (42 for XGB, 123 for LGB)

### Model 3: Neural Network (TabNet or MLP)

**Architecture (TabNet preferred):**
- `n_d`: 32
- `n_a`: 32
- `n_steps`: 5
- `gamma`: 1.5
- `lambda_sparse`: 0.001
- `optimizer`: Adam
- `learning_rate`: 0.001
- `batch_size`: 1024
- `max_epochs`: 200
- `patience`: 20

**If using MLP instead:**
- Input layer: feature_count neurons
- Hidden layer 1: 128 neurons, ReLU, Dropout 0.3
- Hidden layer 2: 64 neurons, ReLU, Dropout 0.3
- Hidden layer 3: 32 neurons, ReLU, Dropout 0.2
- Output layer: 1 neuron, Sigmoid
- Optimizer: Adam with learning rate 0.001
- Batch size: 512
- Early stopping patience: 15

**Training:**
- Same train/validation/test split
- Standardize features before training
- Use class weights if label imbalance > 60/40

### Ensemble Strategy

**Probability Averaging:**
- `ensemble_probability` = (xgb_prob + lgb_prob + nn_prob) / 3

**Model Agreement Scoring:**
- `model_agreement` = number of models predicting same direction (0, 1, 2, or 3)
- If all 3 agree, confidence boost
- If 2 agree, standard confidence
- If split or unanimous opposite, reduce confidence

**Weighted Ensemble (optional enhancement):**
- Weight models by recent 30-day accuracy
- Recalculate weights weekly
- `ensemble_probability` = weighted_average(probs, weights)

---

## PHASE 5: CONFIDENCE CALIBRATION

### Probability Calibration

**Apply Platt Scaling:**
- Train logistic regression on validation set predictions
- Map raw ensemble probabilities to calibrated probabilities
- Store calibration parameters per model

**Apply Isotonic Regression (if sufficient data):**
- Use 500+ validation samples
- Fit isotonic regression to map probabilities

### Confidence Tier Logic

**Tier Thresholds (calibrated probabilities):**
- `HIGH`: probability >= 0.65 OR probability <= 0.35 (direction confidence)
- `MED`: (probability >= 0.58 AND probability < 0.65) OR (probability > 0.35 AND probability <= 0.42)
- `LOW`: probability between 0.42 and 0.58

**Minimum Accuracy Targets:**
- HIGH tier must achieve >= 68% accuracy over rolling 100 predictions
- MED tier must achieve >= 58% accuracy
- LOW tier must achieve >= 52% accuracy
- If thresholds not met, recalibrate

### Position Sizing

**Kelly Criterion Calculation:**
- `win_rate` = model's accuracy on similar features (last 100 predictions)
- `avg_win` = average positive return when correct
- `avg_loss` = average negative return when wrong
- `edge` = win_rate * avg_win - (1 - win_rate) * avg_loss
- `kelly_fraction` = edge / avg_win (capped at 0.25 for safety)

**Position Size Mapping:**
- `position_size` = round(kelly_fraction * 40) capped at 1-10
- If confidence is HIGH, multiply by 1.2
- If confidence is LOW, multiply by 0.5
- If model_agreement < 2, set to minimum (1)

**Expected Value Calculation:**
- `expected_value` = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
- Only trade if expected_value > 0

---

## PHASE 6: PREDICTION PIPELINE

### Real-Time Feature Computation

**Create function `alpha_v4.compute_features_15m(symbol, bucket_time)`:**
- Pull last 100 candles from training.spot_15m
- Calculate all percentile features
- Calculate all slope features
- Calculate all regime features
- Pull 1h context and calculate cross-timeframe features
- Calculate all synthetic indicators
- Return single row feature vector

**Create function `alpha_v4.compute_features_1h(symbol, bucket_time)`:**
- Same as 15m but pull 4h and daily context
- Return single row feature vector

### Prediction Generation

**Create function `alpha_v4.generate_prediction(symbol, timeframe, event_start)`:**
1. Compute features for symbol/timeframe/bucket_time
2. Load active models for symbol/timeframe from alpha_v4.models
3. Get predictions from XGB, LGB, NN
4. Calculate ensemble probability
5. Determine direction (UP if prob > 0.5, DOWN if prob < 0.5)
6. Calculate confidence tier
7. Calculate position size
8. Calculate expected value
9. Insert into alpha_v4.predictions
10. Return prediction record

### Event Triggering

**Create cron job or scheduled function:**
- Run every 15 minutes at :00, :15, :30, :45 for 15m predictions
- Run every hour at :00 for 1h predictions
- For each symbol (BTC, ETH, SOL):
  - Call `generate_prediction(symbol, '15m', now())`
  - Call `generate_prediction(symbol, '1h', now())` if on the hour

**No-Pass Rule:**
- Every event must generate UP or DOWN prediction
- Never return "no trade" or "pass"
- If model confidence is extremely low (0.45-0.55), still predict but set position_size to 1

---

## PHASE 7: RESULT TRACKING

### Resolution Logic

**Create function `alpha_v4.resolve_prediction(prediction_id)`:**
1. Look up prediction by ID
2. Get event_end time from prediction
3. Query actual price at event_start (start_price)
4. Query actual price at event_end (end_price)
5. Calculate pct_change = (end_price - start_price) / start_price * 100
6. Determine actual_result: UP if pct_change > 0, DOWN if pct_change < 0
7. Determine is_correct: 1 if prediction matches actual_result, 0 otherwise
8. Update prediction record with actual_result, is_correct, start_price, end_price, pct_change, resolved_at
9. Update result_display: "✓ CORRECT" if correct, "✗ WRONG" if wrong

**Create scheduled job:**
- Run every 15 minutes
- Resolve all predictions where event_end <= now() AND resolved_at IS NULL

### Performance Monitoring

**Create function `alpha_v4.update_performance_log()`:**
- Run every hour
- Calculate rolling 7-day, 30-day, 90-day accuracy per symbol/timeframe
- Calculate calibration error (expected vs actual accuracy per tier)
- Calculate Sharpe ratio of prediction returns
- Calculate max drawdown
- Insert into alpha_v4.performance_log

---

## PHASE 8: VIEW CREATION

### Create Prediction Results View

**CREATE VIEW `alpha_v4.v_predictions_results` AS**
```sql
SELECT 
    p.symbol,
    p.timeframe,
    p.event_start,
    p.event_end,
    CASE 
        WHEN p.direction_binary = 1 THEN 'UP ' || p.symbol
        ELSE 'DOWN ' || p.symbol
    END as bet,
    p.position_size as magnitude,
    CASE 
        WHEN p.ensemble_probability > 0.5 THEN p.ensemble_probability
        ELSE 1 - p.ensemble_probability
    END as probability,
    p.confidence_tier,
    p.actual_result,
    p.is_correct,
    CASE 
        WHEN p.actual_result IS NULL THEN 'PENDING'
        WHEN p.is_correct = true THEN '✓ CORRECT'
        ELSE '✗ WRONG'
    END as result_display,
    p.start_price,
    p.end_price,
    p.pct_change,
    p.created_at
FROM alpha_v4.predictions p
ORDER BY p.created_at DESC;
```

### Export Function

**Create function `alpha_v4.export_predictions_to_csv()`:**
- Export v_predictions_results to CSV format
- Include headers: symbol, timeframe, event_start, event_end, bet, magnitude, probability, confidence_tier, actual_result, is_correct, result_display, start_price, end_price, pct_change, created_at
- Save to designated export location

---

## PHASE 9: RETRAINING PROTOCOL

### Automated Retraining

**Trigger conditions:**
- Monthly retraining regardless of performance
- Immediate retraining if rolling 7-day accuracy drops below 60%
- Immediate retraining if calibration error exceeds 10%

**Retraining process:**
1. Pull last 30 days of predictions
2. Calculate actual accuracy vs expected
3. If gap > 15%, investigate for data leakage
4. Pull fresh training data (extend by 1 month)
5. Retrain all 3 models with same hyperparameters
6. Validate on most recent 3 months
7. If validation accuracy > 75%, deploy new models
8. If not, keep current models and flag for review

### Feature Importance Monitoring

**Weekly analysis:**
- Calculate feature importance drift
- If top 5 features change significantly, investigate
- If new synthetic indicators show low importance, iterate on formula

---

## PHASE 10: DEPLOYMENT CHECKLIST

**Before going live:**
- [ ] All 6 models trained (BTC/ETH/SOL × 15m/1h)
- [ ] All models achieve >75% validation accuracy
- [ ] All models achieve >0.85 AUC
- [ ] Ensemble achieves >78% validation accuracy
- [ ] HIGH confidence tier achieves >68% on 100+ test predictions
- [ ] MED confidence tier achieves >58%
- [ ] LOW confidence tier achieves >52%
- [ ] No data leakage detected in audit
- [ ] Prediction pipeline tested end-to-end
- [ ] Resolution pipeline tested end-to-end
- [ ] Performance logging verified
- [ ] Rollback plan documented

**Live monitoring:**
- Dashboard showing accuracy by symbol/timeframe/confidence tier
- Alert if rolling 7-day accuracy drops below 60%
- Alert if calibration error exceeds 10%
- Daily export of v_predictions_results

---

## TARGET METRICS

**Validation targets:**
- Per-model accuracy: >75%
- Ensemble accuracy: >78%
- AUC: >0.85

**Live performance targets:**
- Overall accuracy: >65%
- HIGH confidence accuracy: >70%
- MED confidence accuracy: >60%
- LOW confidence accuracy: >55%
- Calibration error: <8%