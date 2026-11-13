// Fast C++ Monte Carlo engine for the Avellaneda-Stoikov market maker.
// Mirrors python/simulate.py's logic exactly (same formulas from
// math/derivation.md Sections 4-5, same thinned-Poisson fill model) so the
// two can be cross-checked against each other -- and because a market
// maker's live quoting loop is exactly the kind of hot path that eventually
// needs to run outside Python.
//
// Usage: ./as_simulate [n_sims] [n_steps] [seed]

#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <string>
#include <numeric>
#include <chrono>
#include <algorithm>

struct Params {
    double S0 = 100.0;
    double sigma = 2.0;
    double T = 1.0;
    double A = 140.0;
    double k = 1.5;
    double gamma = 0.1;
};

struct SimResult {
    double final_pnl;
    int final_inventory;
    int max_abs_inventory;
    int n_fills;
};

inline double reservation_price(double s, int q, double gamma, double sigma, double T, double t) {
    return s - q * gamma * sigma * sigma * (T - t);
}

inline double optimal_spread(double gamma, double sigma, double T, double t, double k) {
    double inventory_risk = gamma * sigma * sigma * (T - t);
    double liquidity = (2.0 / gamma) * std::log(1.0 + gamma / k);
    return inventory_risk + liquidity;
}

// strategy: true = Avellaneda-Stoikov, false = naive fixed symmetric spread
SimResult run_one_sim(const Params& p, int n_steps, bool as_strategy, std::mt19937_64& rng) {
    std::normal_distribution<double> norm(0.0, 1.0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    double dt = p.T / n_steps;
    double sqrt_dt = std::sqrt(dt);

    double s = p.S0;
    int q = 0;
    double X = 0.0;
    int n_fills = 0;
    int max_abs_q = 0;
    double naive_half_spread = 1.0 / p.k;

    for (int i = 0; i < n_steps; ++i) {
        double t = i * dt;

        double bid, ask;
        if (as_strategy) {
            double r = reservation_price(s, q, p.gamma, p.sigma, p.T, t);
            double spread = optimal_spread(p.gamma, p.sigma, p.T, t, p.k);
            bid = r - spread / 2.0;
            ask = r + spread / 2.0;
        } else {
            bid = s - naive_half_spread;
            ask = s + naive_half_spread;
        }

        double delta_b = s - bid;
        double delta_a = ask - s;
        double p_fill_bid = p.A * std::exp(-p.k * delta_b) * dt;
        double p_fill_ask = p.A * std::exp(-p.k * delta_a) * dt;

        if (unif(rng) < p_fill_bid) { q += 1; X -= bid; n_fills++; }
        if (unif(rng) < p_fill_ask) { q -= 1; X += ask; n_fills++; }

        max_abs_q = std::max(max_abs_q, std::abs(q));

        // advance mid-price
        s += p.sigma * sqrt_dt * norm(rng);
    }

    double final_pnl = X + q * s;
    return {final_pnl, q, max_abs_q, n_fills};
}

int main(int argc, char** argv) {
    int n_sims = argc > 1 ? std::stoi(argv[1]) : 500;
    int n_steps = argc > 2 ? std::stoi(argv[2]) : 200;
    uint64_t seed = argc > 3 ? std::stoull(argv[3]) : 42;

    Params p;
    std::mt19937_64 rng(seed);

    std::vector<double> as_pnls, naive_pnls;
    std::vector<int> as_max_inv, naive_max_inv;
    as_pnls.reserve(n_sims);
    naive_pnls.reserve(n_sims);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int sim = 0; sim < n_sims; ++sim) {
        auto as_res = run_one_sim(p, n_steps, true, rng);
        auto naive_res = run_one_sim(p, n_steps, false, rng);
        as_pnls.push_back(as_res.final_pnl);
        naive_pnls.push_back(naive_res.final_pnl);
        as_max_inv.push_back(as_res.max_abs_inventory);
        naive_max_inv.push_back(naive_res.max_abs_inventory);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    auto mean = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    auto mean_i = [](const std::vector<int>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    auto stddev = [&](const std::vector<double>& v, double m) {
        double sq = 0.0;
        for (double x : v) sq += (x - m) * (x - m);
        return std::sqrt(sq / (v.size() - 1));
    };

    double as_mean = mean(as_pnls), naive_mean = mean(naive_pnls);
    double as_std = stddev(as_pnls, as_mean), naive_std = stddev(naive_pnls, naive_mean);

    std::cout << "n_sims=" << n_sims << " n_steps=" << n_steps << " elapsed_sec=" << elapsed << "\n";
    std::cout << "sims_per_sec=" << (2 * n_sims / elapsed) << "\n\n";
    std::cout << "Avellaneda-Stoikov: mean_pnl=" << as_mean << " std_pnl=" << as_std
              << " sharpe=" << (as_mean / as_std) << " mean_max_abs_inv=" << mean_i(as_max_inv) << "\n";
    std::cout << "Naive symmetric:    mean_pnl=" << naive_mean << " std_pnl=" << naive_std
              << " sharpe=" << (naive_mean / naive_std) << " mean_max_abs_inv=" << mean_i(naive_max_inv) << "\n";
    return 0;
}
