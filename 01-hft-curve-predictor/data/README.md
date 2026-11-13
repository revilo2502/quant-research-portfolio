# Data

`synth_data.py` generates a synthetic order book so the whole pipeline runs
with zero API keys / paid data. Swap in real data later:

## Option A: Binance/Bybit L2 order book (crypto, free, easiest)
Use their WebSocket depth streams (`<symbol>@depth10@100ms` on Binance) to
capture top-10 book snapshots. Write a small Python script that appends each
snapshot to a CSV with the **same column schema** `synth_data.py` produces
(`timestamp, mid_price, spread, bid_px_0..9, bid_sz_0..9, ask_px_0..9,
ask_sz_0..9, trade_flow`) and the rest of the pipeline (`features.py`,
`train.py`, `backtest.py`, the C++ engine) works unchanged.

## Option B: LOBSTER (equities, academic standard)
https://lobsterdata.com/ offers free sample data (1 day, a few tickets) in
message/orderbook CSV pairs. Reconstruct top-10 snapshots from the message
stream and map to the same schema as above.

## Why synthetic for this repo
Real tick data either needs a paid subscription or a live WebSocket capture
session (can't be bundled in a zip). The synthetic generator produces a
price process with genuine short-horizon predictability (regime-switching
drift + trade-flow correlated with future returns) so the models have real
signal to find, not just noise -- good enough to validate the full pipeline
end-to-end before plugging in real data.
