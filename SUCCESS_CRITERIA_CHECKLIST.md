# ALPHA v4 - SUCCESS CRITERIA CHECKLIST

---

## PHASE 1: SCHEMA SETUP VERIFICATION

### Database Infrastructure
- [ ] **Schema Created**: `alpha_v4` schema exists and is accessible
  - *Verify*: `SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'alpha_v4';` returns 1 row
  
- [ ] **Tables Created**: All 8 core tables exist
  - *Verify*: `SELECT table_name FROM information_schema.tables WHERE table_schema = 'alpha_v4';` returns: models, raw_features_15m, raw_features_1h, features_15m, features_1h, synthetic_indicators, labels, predictions, performance_log

- [ ] **Primary Keys Set**: All tables have primary keys defined
  - *Verify*: `SELECT constraint_name FROM information_schema.table_constraints WHERE constraint_type = 'PRIMARY KEY' AND table_schema = 'alpha_v4';` returns 8+ constraints

- [ ] **Indexes Created**: Performance indexes exist on predictions and features tables
  - *Verify*: `SELECT indexname FROM pg_indexes WHERE schemaname = 'alpha_v4';` shows indexes on (symbol, timeframe, event_start), (created_at), (symbol, bucket_time)

- [ ] **Storage Allocated**: Under 40GB used for schema
  - *Verify*: `SELECT pg_size_pretty(pg_total_relation_size('alpha_v4.predictions'));` and sum of all tables shows < 40GB

---

## PHASE 2: DATA PIPELINE VERIFICATION

### Raw Data Availability
- [ ] **OHLCV Data Current**: indicators.ohlcv_15m has data within last 15 minutes
  - *Verify*: `SELECT MAX(bucket_time) FROM indicators.ohlcv_15m;` returns timestamp within 15 minutes of now()

- [ ] **Historical Data Complete**: 5 years of historical data exists (2021-01-01 to present)
  - *Verify*: `SELECT MIN(bucket_time), MAX(bucket_time), COUNT(*) FROM indicators.ohlcv_15m WHERE symbol = 'BTC';` shows MIN of 2021-01-01 or earlier, MAX of current time, COUNT > 150,000 rows

- [ ] **No Data Gaps**: No missing 15-minute intervals in last 90 days
  - *Verify*: Query checking for gaps returns 0 results: `SELECT bucket_time FROM generate_series(NOW() - INTERVAL '90 days', NOW(), INTERVAL '15 minutes') AS bucket_time WHERE NOT EXISTS (SELECT 1 FROM indicators.ohlcv_15m WHERE bucket_time = bucket_time);` returns empty

### Label Generation
- [ ] **Labels Created**: alpha_v4.labels has entries for all symbol/timeframe combinations
  - *Verify*: `SELECT symbol, timeframe, COUNT(*) FROM alpha_v4.labels GROUP BY symbol, timeframe;` shows 6 rows (BTC/ETH/SOL × 15m/1h) with counts matching feature counts

- [ ] **Label Distribution Balanced**: UP/DOWN ratio between 45/55 and 55/45
  - *Verify*: `SELECT label, COUNT(*) FROM alpha_v4.labels GROUP BY label;` shows neither UP nor DOWN exceeds 60% of total

- [ ] **Magnitude Labels Present**: All magnitude values 1-10 exist in data
  - *Verify*: `SELECT DISTINCT magnitude FROM alpha_v4.labels ORDER BY magnitude;` returns 1,2,3,4,5,6,7,8,9,10

- [ ] **Next Label Binary Correct**: next_label_binary = 1 for UP, 0 for DOWN
  - *Verify*: `SELECT next_label, next_label_binary FROM alpha_v4.labels LIMIT 10;` shows UP=1, DOWN=0 consistently

---

## PHASE 3: FEATURE ENGINEERING VERIFICATION

### Feature Count & Coverage
- [ ] **Feature Count Met**: features_15m has 180+ columns (excluding metadata)
  - *Verify*: `SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = 'alpha_v4' AND table_name = 'features_15m';` returns > 180

- [ ] **Percentile Features Present**: All indicators have pct_20, pct_50, pct_100 variants
  - *Verify*: `SELECT column_name FROM information_schema.columns WHERE table_name = 'features_15m' AND column_name LIKE '%pct_%';` returns 300+ rows (10 indicators × 3 percentiles × 10 symbols tracked)

- [ ] **Slope Features Present**: All slope columns exist (rsi_slope_3 through rsi_slope_20, etc.)
  - *Verify*: `SELECT column_name FROM information_schema.columns WHERE table_name = 'features_15m' AND column_name LIKE '%slope%';` returns 50+ rows

- [ ] **Regime Features Present**: All boolean regime indicators exist
  - *Verify*: `SELECT column_name FROM information_schema.columns WHERE table_name = 'features_15m' AND column_name LIKE 'is_%';` returns 15+ rows

- [ ] **Cross-Timeframe Features Present**: 1h alignment features exist in 15m table
  - *Verify*: `SELECT column_name FROM information_schema.columns WHERE table_name = 'features_15m' AND (column_name LIKE '%1h%' OR column_name LIKE 'trend_1h%');` returns 5+ rows

### Data Quality
- [ ] **No NULL Features**: Core features have < 1% NULL values
  - *Verify*: `SELECT COUNT(*) - COUNT(rsi_14_pct_20) AS null_count FROM alpha_v4.features_15m;` divided by total count shows < 0.01

- [ ] **Percentile Range Valid**: All pct_ values between 0 and 1
  - *Verify*: `SELECT MAX(rsi_14_pct_20), MIN(rsi_14_pct_20) FROM alpha_v4.features_15m;` shows MAX ≤ 1.0, MIN ≥ 0.0

- [ ] **Slope Values Reasonable**: Slope values not exceeding ±100
  - *Verify*: `SELECT MAX(ABS(rsi_slope_3)) FROM alpha_v4.features_15m;` returns < 100

- [ ] **Temporal Features Valid**: hour_sin and hour_cos between -1 and 1
  - *Verify*: `SELECT MAX(hour_sin), MIN(hour_sin), MAX(hour_cos), MIN(hour_cos) FROM alpha_v4.features_15m;` all between -1 and 1

### Synthetic Indicators
- [ ] **Synthetic Table Populated**: synthetic_indicators has rows for all symbols
  - *Verify*: `SELECT symbol, COUNT(*) FROM alpha_v4.synthetic_indicators GROUP BY symbol;` shows BTC, ETH, SOL with counts matching features count

- [ ] **Trend Strength Valid**: syn_trend_strength between 0 and 100
  - *Verify*: `SELECT MAX(syn_trend_strength), MIN(syn_trend_strength) FROM alpha_v4.synthetic_indicators;` shows 0-100 range

- [ ] **Volatility Regime Valid**: syn_volatility_regime only contains 0, 1, or 2
  - *Verify*: `SELECT DISTINCT syn_volatility_regime FROM alpha_v4.synthetic_indicators;` returns only 0, 1, 2

- [ ] **Market Structure Valid**: syn_market_structure only contains 'bull', 'bear', 'range'
  - *Verify*: `SELECT DISTINCT syn_market_structure FROM alpha_v4.synthetic_indicators;` returns only those three values

- [ ] **Correlation Features Valid**: syn_btc_eth_correlation_20 between -1 and 1
  - *Verify*: `SELECT MAX(syn_btc_eth_correlation_20), MIN(syn_btc_eth_correlation_20) FROM alpha_v4.synthetic_indicators;` between -1 and 1

---

## PHASE 4: MODEL TRAINING VERIFICATION

### Training Data
- [ ] **Training Set Size**: > 200,000 samples for 15m models, > 50,000 for 1h models
  - *Verify*: Check train_samples column in alpha_v4.models shows appropriate counts

- [ ] **No Data Leakage**: Training end date is at least 48 hours before validation start
  - *Verify*: `SELECT train_end, val_start FROM alpha_v4.models;` shows gap ≥ 48 hours

- [ ] **Class Balance**: Training data has 40-60% split between classes
  - *Verify*: Training logs show class distribution, or query labels table for training period

### Model Performance (Validation)
- [ ] **XGBoost Accuracy**: ≥ 75% validation accuracy
  - *Verify*: `SELECT val_accuracy FROM alpha_v4.models WHERE model_type = 'xgboost';` ≥ 0.75

- [ ] **LightGBM Accuracy**: ≥ 75% validation accuracy
  - *Verify*: `SELECT val_accuracy FROM alpha_v4.models WHERE model_type = 'lightgbm';` ≥ 0.75

- [ ] **MLP Accuracy**: ≥ 73% validation accuracy (slightly lower acceptable for NN)
  - *Verify*: `SELECT val_accuracy FROM alpha_v4.models WHERE model_type = 'mlp';` ≥ 0.73

- [ ] **AUC Scores**: All models have AUC ≥ 0.85
  - *Verify*: `SELECT val_auc FROM alpha_v4.models;` all values ≥ 0.85

- [ ] **Precision/Recall Balance**: Precision ≥ 0.70, Recall ≥ 0.70
  - *Verify*: `SELECT val_precision, val_recall FROM alpha_v4.models;` both ≥ 0.70

- [ ] **Feature Importance Logged**: Top 20 features stored in feature_importance column
  - *Verify*: `SELECT feature_importance FROM alpha_v4.models LIMIT 1;` returns JSON with 20+ features

### Model Persistence
- [ ] **Model Binary Stored**: model_binary column not NULL for all active models
  - *Verify*: `SELECT COUNT(*) FROM alpha_v4.models WHERE model_binary IS NULL AND is_active = true;` returns 0

- [ ] **Model Size Reasonable**: Model binaries < 500MB each
  - *Verify*: `SELECT pg_column_size(model_binary) FROM alpha_v4.models;` all < 524288000 bytes

- [ ] **Backup Created**: Models backed up to Storage bucket
  - *Verify*: Check Supabase Storage `model-backups` bucket has 6 files (one per model)

---

## PHASE 5: PREDICTION PIPELINE VERIFICATION

### Real-Time Feature Computation
- [ ] **Features Current**: alpha_v4.features_15m has data within last 15 minutes
  - *Verify*: `SELECT MAX(bucket_time) FROM alpha_v4.features_15m;` within 15 min of now

- [ ] **Materialized Views Refreshing**: mv_features_15m updates every 15 minutes
  - *Verify*: Check last refresh time: `SELECT last_refresh FROM pg_matviews WHERE matviewname = 'mv_features_15m';` within 15 min

- [ ] **No Calculation Errors**: No ERROR logs in Edge Function or database function logs
  - *Verify*: Check Supabase logs for feature computation function - no errors in last hour

### Prediction Generation
- [ ] **Predictions Inserted**: New predictions appear every 15 minutes
  - *Verify*: `SELECT MAX(created_at) FROM alpha_v4.predictions WHERE timeframe = '15m';` within 15 min of now

- [ ] **All Symbols Covered**: BTC, ETH, SOL all have recent predictions
  - *Verify*: `SELECT symbol, MAX(created_at) FROM alpha_v4.predictions GROUP BY symbol;` all show recent timestamps

- [ ] **All Timeframes Covered**: Both 15m and 1h have recent predictions
  - *Verify*: `SELECT timeframe, MAX(created_at) FROM alpha_v4.predictions GROUP BY timeframe;` both show recent timestamps

- [ ] **No Missing Events**: Every 15-minute slot has exactly 3 predictions (BTC, ETH, SOL)
  - *Verify*: For any given 15-min window, `SELECT COUNT(DISTINCT symbol) FROM alpha_v4.predictions WHERE event_start = 'specific_time';` returns 3

### Prediction Data Quality
- [ ] **Probabilities Valid**: All probabilities between 0 and 1
  - *Verify*: `SELECT MAX(ensemble_probability), MIN(ensemble_probability) FROM alpha_v4.predictions;` between 0 and 1

- [ ] **Direction Consistent**: direction_binary = 1 when prediction = 'UP', 0 when 'DOWN'
  - *Verify*: `SELECT prediction, direction_binary FROM alpha_v4.predictions LIMIT 20;` shows consistency

- [ ] **Confidence Tiers Assigned**: All predictions have HIGH, MED, or LOW tier
  - *Verify*: `SELECT DISTINCT confidence_tier FROM alpha_v4.predictions;` returns only HIGH, MED, LOW

- [ ] **Position Sizes Valid**: All position_size values between 1 and 10
  - *Verify*: `SELECT MAX(position_size), MIN(position_size) FROM alpha_v4.predictions;` MIN=1, MAX=10

- [ ] **Model Agreement Tracked**: model_agreement column populated (0-3)
  - *Verify*: `SELECT DISTINCT model_agreement FROM alpha_v4.predictions;` returns 0,1,2,3

### Resolution Pipeline
- [ ] **Predictions Resolving**: Resolved predictions appear within 1 minute of event_end
  - *Verify*: `SELECT COUNT(*) FROM alpha_v4.predictions WHERE event_end < NOW() - INTERVAL '5 minutes' AND resolved_at IS NULL;` returns 0 (or very few pending)

- [ ] **Actual Results Populated**: actual_result column filled for resolved predictions
  - *Verify*: `SELECT COUNT(*) FROM alpha_v4.predictions WHERE resolved_at IS NOT NULL AND actual_result IS NULL;` returns 0

- [ ] **Price Data Captured**: start_price and end_price populated for resolved predictions
  - *Verify*: `SELECT COUNT(*) FROM alpha_v4.predictions WHERE resolved_at IS NOT NULL AND (start_price IS NULL OR end_price IS NULL);` returns 0

- [ ] **Is Correct Calculated**: is_correct = true when prediction matches actual_result
  - *Verify*: Sample 10 rows where actual_result = 'UP' and prediction = 'UP', confirm is_correct = true

---

## PHASE 6: PERFORMANCE VERIFICATION

### Accuracy Metrics
- [ ] **Overall Accuracy**: Rolling 7-day accuracy ≥ 65%
  - *Verify*: `SELECT AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) FROM alpha_v4.predictions WHERE created_at > NOW() - INTERVAL '7 days';` ≥ 0.65

- [ ] **HIGH Confidence Accuracy**: ≥ 70% accuracy for HIGH tier predictions
  - *Verify*: `SELECT AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) FROM alpha_v4.predictions WHERE confidence_tier = 'HIGH' AND created_at > NOW() - INTERVAL '7 days';` ≥ 0.70

- [ ] **MED Confidence Accuracy**: ≥ 60% accuracy for MED tier
  - *Verify*: Same query with MED tier, result ≥ 0.60

- [ ] **LOW Confidence Accuracy**: ≥ 52% accuracy for LOW tier
  - *Verify*: Same query with LOW tier, result ≥ 0.52

- [ ] **Per-Symbol Accuracy**: Each symbol (BTC, ETH, SOL) has ≥ 60% accuracy
  - *Verify*: Group by symbol, all three ≥ 0.60

- [ ] **Per-Timeframe Accuracy**: Both 15m and 1h have ≥ 60% accuracy
  - *Verify*: Group by timeframe, both ≥ 0.60

### Calibration Metrics
- [ ] **Calibration Error**: Expected vs actual accuracy gap < 8%
  - *Verify*: For HIGH tier with avg probability 0.75, actual accuracy should be 0.75 ± 0.08

- [ ] **Probability Correlation**: Higher probabilities correlate with higher accuracy
  - *Verify*: Bin predictions by probability deciles, accuracy should increase with probability

- [ ] **No Inverted Calibration**: HIGH tier accuracy not lower than LOW tier
  - *Verify*: `SELECT confidence_tier, AVG(CASE WHEN is_correct THEN 1 ELSE 0 END) as acc FROM alpha_v4.predictions GROUP BY confidence_tier ORDER BY acc DESC;` shows HIGH > MED > LOW

### Consistency Metrics
- [ ] **No Streaks of Failure**: No 10-prediction losing streak in last 7 days
  - *Verify*: Query for consecutive losses, max streak < 10

- [ ] **Directional Balance**: Prediction distribution 45-55% UP vs DOWN
  - *Verify*: `SELECT prediction, COUNT(*) FROM alpha_v4.predictions GROUP BY prediction;` ratio between 0.45 and 0.55

- [ ] **Model Agreement Distribution**: Model_agreement = 3 (all agree) for > 30% of predictions
  - *Verify*: `SELECT model_agreement, COUNT(*) FROM alpha_v4.predictions GROUP BY model_agreement;` shows agreement=3 > 30% of total

---

## PHASE 7: INFRASTRUCTURE VERIFICATION

### Resource Utilization
- [ ] **CPU Usage**: Average CPU < 80% during peak (prediction time)
  - *Verify*: Supabase dashboard shows CPU < 80%

- [ ] **RAM Usage**: Database using 20-28GB of 32GB available
  - *Verify*: Supabase dashboard shows memory in 20-28GB range

- [ ] **Storage Usage**: Under 90GB of 100GB used
  - *Verify*: Supabase dashboard shows storage < 90GB

- [ ] **Connection Pool**: Never hitting max connections (100)
  - *Verify*: Supabase dashboard shows active connections < 80

### Edge Functions
- [ ] **Functions Deployed**: All 6 prediction functions deployed
  - *Verify*: Supabase Edge Functions dashboard shows: predict-btc-15m, predict-btc-1h, predict-eth-15m, predict-eth-1h, predict-sol-15m, predict-sol-1h

- [ ] **No Timeouts**: Edge Functions completing in < 10 seconds
  - *Verify*: Logs show execution time < 10000ms

- [ ] **No Errors**: Zero ERROR level logs in Edge Functions
  - *Verify*: Log explorer shows no errors in last 24 hours

### Cron Jobs
- [ ] **Cron Scheduled**: Jobs running every 15 minutes
  - *Verify*: Supabase Cron dashboard shows schedule `*/15 * * * *`

- [ ] **Cron Executing**: Last execution within 15 minutes
  - *Verify*: Cron logs show recent execution

- [ ] **No Cron Failures**: Last 10 executions all succeeded
  - *Verify*: Cron history shows 10/10 success

---

## PHASE 8: VIEW & EXPORT VERIFICATION

### View Creation
- [ ] **View Exists**: alpha_v4.v_predictions_results view created
  - *Verify*: `SELECT viewname FROM pg_views WHERE schemaname = 'alpha_v4';` includes v_predictions_results

- [ ] **View Columns Match**: Columns match v_latest_signals_predictions-results.csv format
  - *Verify*: `SELECT column_name FROM information_schema.columns WHERE table_name = 'v_predictions_results';` shows: symbol, timeframe, event_start, event_end, bet, magnitude, probability, confidence_tier, actual_result, is_correct, result_display, start_price, end_price, pct_change, created_at

- [ ] **View Data Current**: View returns rows within last 15 minutes
  - *Verify*: `SELECT MAX(created_at) FROM alpha_v4.v_predictions_results;` within 15 min

- [ ] **Result Display Format**: result_display shows "✓ CORRECT", "✗ WRONG", or "PENDING"
  - *Verify*: `SELECT DISTINCT result_display FROM alpha_v4.v_predictions_results;` shows exactly those three values

### Export Functionality
- [ ] **Export Working**: CSV export function executes without error
  - *Verify*: Run export function, check for successful completion

- [ ] **CSV Format Valid**: Exported CSV has correct headers and data types
  - *Verify*: Open CSV, confirm headers match expected, data is comma-separated

- [ ] **Export Timely**: Export completes in < 30 seconds for 10,000 rows
  - *Verify*: Time the export execution

---

## PHASE 9: A/B TESTING & COMPARISON (If Running Parallel)

### Alpha Comparison
- [ ] **Both Schemas Active**: alpha and alpha_v4 both generating predictions
  - *Verify*: Both schemas have predictions with timestamps within last hour

- [ ] **Accuracy Comparison**: alpha_v4 accuracy > alpha accuracy
  - *Verify*: Calculate 7-day accuracy for both, alpha_v4 higher by > 5%

- [ ] **Prediction Count Match**: Same number of predictions in both schemas
  - *Verify*: Count predictions in both for same time period, difference < 1%

---

## PHASE 10: FINAL GO-LIVE CHECKLIST

Before switching from alpha to alpha_v4 exclusively:

- [ ] **30-Day Accuracy**: Rolling 30-day accuracy ≥ 65%
- [ ] **7-Day Accuracy**: Rolling 7-day accuracy ≥ 65%
- [ ] **HIGH Tier Accuracy**: ≥ 70% over last 100 HIGH predictions
- [ ] **No Major Bugs**: Zero critical bugs in last 7 days
- [ ] **Performance Stable**: Query performance < 2 seconds for dashboard
- [ ] **Backup Tested**: Successfully restored model from backup
- [ ] **Rollback Plan**: Documented procedure to switch back to alpha
- [ ] **Monitoring Active**: Alerts configured for accuracy < 60%
- [ ] **Team Trained**: Team knows how to interpret v4 vs alpha differences

---

## CONTINUOUS MONITORING CHECKLIST (Daily)

Run these checks daily:

- [ ] **Accuracy Check**: Yesterday's accuracy ≥ 60%
- [ ] **Calibration Check**: HIGH tier accuracy ≥ 65%
- [ ] **Data Freshness**: Latest prediction within 15 minutes
- [ ] **Resolution Lag**: No unresolved predictions older than 2 hours
- [ ] **Storage Check**: Storage usage < 95GB
- [ ] **Error Log Check**: Zero ERROR logs in last 24 hours
- [ ] **Feature Freshness**: Latest features computed within 15 minutes
- [ ] **Model Health**: All 6 models marked is_active = true

---

## SUCCESS CRITERIA SUMMARY

**Minimum Viable Product (Stage 1-2 Complete):**
- 6 models trained with ≥ 75% validation accuracy
- Predictions generating every 15 minutes for all symbols/timeframes
- Live accuracy ≥ 60% for first 7 days

**Production Ready (All Stages Complete):**
- Live accuracy ≥ 65% sustained for 30 days
- HIGH confidence tier ≥ 70% accuracy
- Calibration error < 8%
- Zero data leakage detected
- All infrastructure monitoring green

**DO NOT PROCEED TO NEXT PHASE UNTIL ALL CHECKBOXES IN CURRENT PHASE ARE VERIFIED.**