#pragma once
#include <array>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace hft {

// Fixed-depth limit order book, reconstructed from top-of-book snapshots.
// Depth is a compile-time constant so the hot path has zero heap allocation.
constexpr int kDepth = 10;

struct BookLevel {
    double price = 0.0;
    double size = 0.0;
};

struct BookSnapshot {
    uint64_t timestamp = 0;
    std::array<BookLevel, kDepth> bids{};
    std::array<BookLevel, kDepth> asks{};

    double mid_price() const {
        return (bids[0].price + asks[0].price) / 2.0;
    }
    double spread() const {
        return asks[0].price - bids[0].price;
    }
    double micro_price() const {
        double bs = bids[0].size, as = asks[0].size;
        if (bs + as <= 0.0) return mid_price();
        return (bids[0].price * as + asks[0].price * bs) / (bs + as);
    }
    double order_book_imbalance(int depth = 1) const {
        double bid_vol = 0, ask_vol = 0;
        depth = std::min(depth, kDepth);
        for (int i = 0; i < depth; ++i) {
            bid_vol += bids[i].size;
            ask_vol += asks[i].size;
        }
        double denom = bid_vol + ask_vol;
        return denom > 0.0 ? (bid_vol - ask_vol) / denom : 0.0;
    }
};

// Rolling feature state the engine maintains incrementally -- O(1) per update
// instead of recomputing rolling windows from scratch (that's the whole point
// of doing this in C++: the Python/pandas .rolling() version is O(window) per
// step with allocation; this ring-buffer version is O(1) with none).
template <int WindowSize>
class RollingStats {
public:
    void push(double x) {
        if (count_ < WindowSize) {
            buf_[count_] = x;
            sum_ += x;
            sum_sq_ += x * x;
            count_++;
        } else {
            double old = buf_[idx_];
            sum_ += x - old;
            sum_sq_ += x * x - old * old;
            buf_[idx_] = x;
        }
        idx_ = (idx_ + 1) % WindowSize;
    }

    double mean() const { return count_ > 0 ? sum_ / count_ : 0.0; }

    double stddev() const {
        if (count_ < 2) return 0.0;
        double m = mean();
        double var = sum_sq_ / count_ - m * m;
        return var > 0.0 ? std::sqrt(var) : 0.0;
    }

    int count() const { return count_; }

private:
    std::array<double, WindowSize> buf_{};
    double sum_ = 0.0;
    double sum_sq_ = 0.0;
    int idx_ = 0;
    int count_ = 0;
};

// Feature vector matching python/features.py's column order for the
// direction classifier (h=1 target). Keep in sync manually -- see
// cpp/inference/feature_order.md.
struct FeatureVector {
    static constexpr int kNumFeatures = 15;
    std::array<float, kNumFeatures> values{};
};

class OrderBookEngine {
public:
    FeatureVector on_snapshot(const BookSnapshot& snap) {
        double mid = snap.mid_price();
        double ret1 = (prev_mid_ > 0.0) ? (mid - prev_mid_) / prev_mid_ : 0.0;
        prev_mid_ = mid;

        vol10_.push(ret1);
        vol50_.push(ret1);
        vol200_.push(ret1);

        mid10_.push(mid);
        mid50_.push(mid);
        mid200_.push(mid);

        double mom10 = mid10_.count() >= 2 ? (mid - mid10_first_or(mid)) / mid10_first_or(mid) : 0.0;

        FeatureVector fv;
        int i = 0;
        fv.values[i++] = static_cast<float>(mid);
        fv.values[i++] = static_cast<float>(snap.spread());
        fv.values[i++] = static_cast<float>(snap.micro_price());
        fv.values[i++] = static_cast<float>(snap.micro_price() - mid);
        fv.values[i++] = static_cast<float>(snap.order_book_imbalance(1));
        fv.values[i++] = static_cast<float>(snap.order_book_imbalance(3));
        fv.values[i++] = static_cast<float>(snap.order_book_imbalance(5));
        fv.values[i++] = static_cast<float>(snap.order_book_imbalance(10));
        fv.values[i++] = static_cast<float>(ret1);
        fv.values[i++] = static_cast<float>(vol10_.stddev());
        fv.values[i++] = static_cast<float>(vol50_.stddev());
        fv.values[i++] = static_cast<float>(vol200_.stddev());
        fv.values[i++] = static_cast<float>(mom10);
        fv.values[i++] = 0.0f;  // trade_flow placeholder (wire up from trade feed)
        fv.values[i++] = 0.0f;  // trade_flow_roll placeholder
        return fv;
    }

private:
    double prev_mid_ = 0.0;
    RollingStats<10> vol10_;
    RollingStats<50> vol50_;
    RollingStats<200> vol200_;
    RollingStats<10> mid10_;
    RollingStats<50> mid50_;
    RollingStats<200> mid200_;

    double mid10_first_or(double fallback) const {
        // approximation: use mean as a stand-in for "value W steps ago" in this
        // minimal ring buffer; swap in an indexable deque for exact momentum.
        return mid10_.count() > 0 ? mid10_.mean() : fallback;
    }
};

}  // namespace hft
