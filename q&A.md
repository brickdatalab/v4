# ALPHA v4 - FINAL IMPLEMENTATION CLARIFICATIONS (2XL INFRASTRUCTURE)

---

## DIRECT ANSWERS TO ALL QUESTIONS (REVISED FOR 2XL)

### 1. DATA SOURCE FOR LIVE PREDICTIONS
**Pull from `indicators.ohlcv_15m` and `indicators.ohlcv_1h` for live predictions.** These are your real-time tables. `training.spot_15m` is historical only. For feature computation during live inference, always use the indicators schema tables.

**Additional instruction for 2XL:** Create materialized views `alpha_v4.mv_features_15m` and `alpha_v4.mv_features_1h` that refresh every 15 minutes using your 8-core ARM CPU. The 32GB RAM allows you to keep 90 days of pre-computed features in memory-cached materialized views.

---

### 2. RAW FEATURES (150 INDICATORS)
**Derive the list from `training.spot_15m_indicators`.** Use all 104 columns from that table as your base. Add 46 more by creating variations:
- Fast/slow versions of existing indicators (RSI 7, RSI 21 in addition to RSI 14)
- Multiple timeframe variants (9-period, 21-period, 50-period)
- Raw OHLCV + derived metrics (body size, wick ratios, true range)

**Additional instruction for 2XL:** Compute 200 indicators, not 150. Your 8-core ARM can handle the extra computation. Add:
- Ichimoku cloud components (tenkan, kijun, senkou spans)
- Fibonacci retracement levels
- Pivot points (standard, camarilla, woodie)
- Volume profile metrics
- Heikin-Ashi candles

---

### 3. MODEL TRAINING ENVIRONMENT
**Train in Supabase using pgvector + Python via plpython3u extension.**

**Why 2XL changes this:** Your 32GB RAM and 8-core ARM CPU can train models directly in Supabase.

**Implementation:**
1. Enable `plpython3u` extension in your Supabase project
2. Install XGBoost, LightGBM, PyTorch in the database environment
3. Train models using SQL functions with Python code blocks
4. Store serialized models in `alpha_v4.models.model_binary` as BYTEA
5. Use 16GB of your 32GB RAM for training (leave 16GB for database operations)

**Alternative if plpython3u is unavailable:** Train locally in Cowork session, upload to Supabase Storage bucket `model-binaries`.

**Loading for inference:** Query model_binary from database, deserialize in Edge Function or plpython3u function.

---

### 4. TABNET vs MLP
**Implement MLP only.** Use PyTorch with the architecture specified in Phase 4.

**Additional instruction for 2XL:** Train MLP with 256-128-64-32 architecture instead of 128-64-32. Your 32GB RAM can handle larger networks. Use batch size 2048 for faster training.

---

### 5. MACRO DATA
**Skip `syn_macro_sentiment` entirely.** Remove from synthetic indicators table. Focus on crypto-native features only.

---

### 6. CLOUD RUN vs SUPABASE
**Use Supabase Edge Functions + cron for inference. Do not use Cloud Run.**

**Additional instruction for 2XL:** Leverage your 8-core ARM by running parallel feature computation:
- Core 1-2: BTC features
- Core 3-4: ETH features  
- Core 5-6: SOL features
- Core 7-8: Model inference

Use PostgreSQL's `pg_background` or parallel query execution for concurrent processing.

---

### 7. EXISTING ALPHA SCHEMA
**Keep `alpha` schema running in parallel until v4 achieves >65% live accuracy for 30 consecutive days.**

**Additional instruction for 2XL:** Your 100GB storage can easily handle both schemas. Keep full history in both. Do not delete alpha until v4 proves superiority.

---

### 8. BTC CORRELATION EDGE CASE
**For BTC, use correlations with ETH and SOL only.**

**Implementation:**
- `syn_btc_eth_correlation_20`: BTC-ETH correlation
- `syn_btc_sol_correlation_20`: BTC-SOL correlation
- For ETH: `syn_eth_btc_correlation_20` and `syn_eth_sol_correlation_20`
- For SOL: `syn_sol_btc_correlation_20` and `syn_sol_eth_correlation_20`

Remove `syn_btc_correlation_20` as standalone feature.

---

### 9. EXECUTION STRATEGY
**Implement incrementally in 3 stages:**

**Stage 1 (Test before proceeding):**
- Phase 1: Schema setup
- Phase 2: Labels
- Phase 3: Features (200 indicators, base + synthetic)

**Stage 2 (Test before proceeding):**
- Phase 4: XGBoost + LightGBM ensemble (skip NN initially)
- Phase 5: Confidence calibration
- Phase 6: Prediction pipeline

**Stage 3 (After Stage 2 hits >60% accuracy):**
- Add MLP to ensemble
- Add advanced position sizing

**Additional instruction for 2XL:** Run Stage 1 and 2 in parallel across your 8 cores. Use parallel workers for feature computation while training models on separate cores.

---

### 10. DEPLOYMENT TARGET
**Supabase Edge Functions + cron.**

**Additional instruction for 2XL:** 
- Create 3 Edge Functions: `predict-btc`, `predict-eth`, `predict-sol`
- Trigger all 3 simultaneously from cron job
- Each function runs on separate logical core
- Use 8-second timeout per function (well within limits)

---

## ADDITIONAL INSTRUCTIONS FOR 2XL INFRASTRUCTURE

### Parallel Feature Computation
**Use all 8 cores for feature engineering.** Create PostgreSQL function that spawns parallel workers:
- Worker 1: Compute RSI variants for all symbols
- Worker 2: Compute MACD variants for all symbols
- Worker 3: Compute Bollinger Bands for all symbols
- Worker 4: Compute volume indicators for all symbols
- Worker 5: Compute trend indicators for all symbols
- Worker 6: Compute volatility indicators for all symbols
- Worker 7: Compute synthetic indicators
- Worker 8: Aggregate and normalize

**Use `max_parallel_workers_per_gather = 8` in PostgreSQL config.**

### In-Database Training
**Train models using plpython3u with this resource allocation:**
- 16GB RAM for training data and model fitting
- 4 cores for XGBoost (n_jobs=4)
- 2 cores for LightGBM
- 2 cores for MLP

**Training SQL pattern:**
```sql
SELECT alpha_v4.train_xgboost('BTC', '15m', '2021-01-01', '2025-06-01');
```

### Materialized View Refresh Strategy
**Refresh materialized views concurrently using your 8 cores:**
- `alpha_v4.mv_features_15m`: Refresh every 15 minutes
- `alpha_v4.mv_features_1h`: Refresh every hour
- Use `CONCURRENTLY` option to avoid locks
- Store 90 days of features (approximately 25,920 rows per symbol for 15m)

### Storage Allocation
**Allocate your 100GB as follows:**
- 40GB: Raw data and features
- 30GB: Model binaries and backups
- 20GB: Prediction history (90 days)
- 10GB: Logs and performance metrics

### Memory-Cached Features
**Enable PostgreSQL's shared_buffers to use 16GB of your 32GB RAM for caching hot feature data.** Set `shared_buffers = '16GB'` in database configuration.

### Real-Time Feature Streaming
**Use your 8-core ARM to compute features in real-time as new OHLCV data arrives.** Create trigger on `indicators.ohlcv_15m` that computes and inserts features immediately upon new candle close.

### Model Ensemble Parallelism
**Run all 3 models (XGB, LGB, MLP) in parallel during inference:**
- Load all 3 models into memory (32GB RAM easily handles this)
- Spawn 3 threads/processes
- Each model predicts simultaneously
- Aggregate results

### Batch Prediction Capability
**With 32GB RAM, you can batch predict next 24 hours (96 15m candles) in single operation.** Pre-compute predictions for entire day, store in `alpha_v4.predictions` with future timestamps.

### High-Frequency Monitoring
**Log performance metrics every 15 minutes instead of daily.** Your 8-core ARM can handle the write load. Use `alpha_v4.performance_log` with 15-minute granularity.

### Feature Store Architecture
**Create dedicated feature store tables:**
- `alpha_v4.feature_store_15m`: Pre-computed features for last 90 days
- `alpha_v4.feature_store_1h`: Pre-computed features for last 90 days
- Partition by symbol and week for fast queries
- Index all feature columns for sub-second retrieval

### Model Versioning
**Store up to 10 model versions per symbol/timeframe.** Your 100GB storage allows keeping model history. Table structure:
- `alpha_v4.model_versions`: All historical models
- `alpha_v4.models`: Only active models

### Backup and Recovery
**Daily automated backups of:**
- All model binaries to `model-backups` bucket
- Feature store tables to CSV in `feature-backups` bucket
- Prediction history to `predictions-backup` bucket

Use 20GB of your 100GB for backups (retain 30 days).

### Connection Pooling
**Configure connection pool to use 100 connections.** Your 2XL plan supports this. Allocate:
- 20 connections for feature computation
- 20 connections for model training
- 30 connections for inference
- 30 connections for monitoring and logging

### Query Optimization
**Enable parallel sequential scans and parallel index scans.** Set:
- `max_parallel_workers = 8`
- `max_parallel_maintenance_workers = 4`
- `parallel_tuple_cost = 0.01`
- `parallel_setup_cost = 100`

### CPU Utilization Target
**Target 70-80% CPU utilization during peak operations.** Your 8-core ARM should handle:
- Feature computation: 40% CPU
- Model inference: 20% CPU
- Database operations: 20% CPU
- Headroom: 20% CPU

### RAM Utilization Target
**Target 24GB of 32GB RAM usage (75%).** Breakdown:
- Shared buffers: 16GB
- Feature cache: 4GB
- Model binaries in memory: 2GB
- Query operations: 2GB

### Edge Function Optimization
**Deploy 6 Edge Functions (one per symbol/timeframe combination):**
- `predict-btc-15m`
- `predict-btc-1h`
- `predict-eth-15m`
- `predict-eth-1h`
- `predict-sol-15m`
- `predict-sol-1h`

Each function loads only relevant model and features. Trigger all 6 simultaneously from cron.

### Database Function for Batch Inference
**Create PostgreSQL function `alpha_v4.batch_predict()` that:**
1. Loads all 6 models into memory
2. Fetches latest features for all symbols/timeframes
3. Runs predictions in parallel using 6 cores
4. Inserts all predictions in single transaction
5. Completes in under 5 seconds

### Monitoring Dashboard Queries
**Pre-compute dashboard metrics every 15 minutes using materialized views:**
- `alpha_v4.mv_accuracy_7d`
- `alpha_v4.mv_accuracy_30d`
- `alpha_v4.mv_calibration_error`
- `alpha_v4.mv_sharpe_ratio`

Refresh concurrently to avoid blocking reads.

### Data Retention with 2XL Power
**Keep 180 days of predictions instead of 90.** Your 100GB storage and 32GB RAM can handle this. Set up monthly archival of data older than 180 days to cold storage.

### Feature Importance Tracking
**Compute feature importance weekly using SHAP values.** Your 8-core ARM can handle SHAP computation for 200 features. Store results in `alpha_v4.feature_importance_history`.

### A/B Testing Infrastructure
**Run alpha and alpha_v4 in parallel with full resource allocation:**
- alpha: Uses 2 cores, 8GB RAM
- alpha_v4: Uses 6 cores, 24GB RAM

Monitor both simultaneously. Compare accuracy in real-time using `alpha_v4.ab_test_results` table.

### Rollback with Zero Downtime
**Maintain hot-swappable model infrastructure.** Keep previous model version loaded in memory alongside current version. If current version fails, switch to previous version in under 1 second.

### Stress Testing
**Before production deployment, stress test with:**
- 1000 concurrent prediction requests
- 10,000 feature computations per minute
- 100MB/s write throughput to predictions table

Your 2XL infrastructure should handle this comfortably.

### Final Resource Allocation
| Component | CPU Cores | RAM | Storage |
|-----------|-----------|-----|---------|
| Database Operations | 2 | 16GB | 40GB |
| Feature Computation | 2 | 8GB | - |
| Model Training | 2 | 4GB | 30GB |
| Model Inference | 1 | 2GB | - |
| Monitoring/Logging | 1 | 2GB | 20GB |
| **Total** | **8** | **32GB** | **90GB** |
| **Headroom** | **0** | **0GB** | **10GB** |

---

## FINAL INSTRUCTION

**Execute Stage 1 immediately. Leverage all 8 cores and 32GB RAM. Target 65%+ live accuracy within 60 days. Do not stop until achieved.**