// Replays the synthetic LOB CSV through OrderBookEngine and times it --
// this is the "C++ order book reconstruction + feature engine" half of the
// latency benchmark (see benchmarks/latency_compare.py for the comparison
// against the pandas/Python feature pipeline).
//
// Usage: ./replay_engine ../../data/synthetic_lob.csv [max_rows]

#include "orderbook.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>

using namespace hft;

struct CsvRow {
    uint64_t timestamp;
    double mid_price, spread;
    std::array<double, kDepth> bid_px, bid_sz, ask_px, ask_sz;
    double trade_flow;
};

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(item);
    return out;
}

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "../../data/synthetic_lob.csv";
    size_t max_rows = argc > 2 ? std::stoul(argv[2]) : SIZE_MAX;

    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Could not open " << path << "\n"
                  << "Run: python3 python/synth_data.py  (from the project root) first.\n";
        return 1;
    }

    std::string header;
    std::getline(file, header);
    auto cols = split_csv_line(header);

    // find column indices (header format from synth_data.py)
    auto idx_of = [&](const std::string& name) -> int {
        for (size_t i = 0; i < cols.size(); ++i)
            if (cols[i] == name) return static_cast<int>(i);
        return -1;
    };
    int c_ts = idx_of("timestamp"), c_mid = idx_of("mid_price"), c_spread = idx_of("spread");
    int c_tf = idx_of("trade_flow");
    std::array<int, kDepth> c_bid_px{}, c_bid_sz{}, c_ask_px{}, c_ask_sz{};
    for (int l = 0; l < kDepth; ++l) {
        c_bid_px[l] = idx_of("bid_px_" + std::to_string(l));
        c_bid_sz[l] = idx_of("bid_sz_" + std::to_string(l));
        c_ask_px[l] = idx_of("ask_px_" + std::to_string(l));
        c_ask_sz[l] = idx_of("ask_sz_" + std::to_string(l));
    }

    OrderBookEngine engine;
    std::string line;
    size_t n = 0;
    double checksum = 0.0;  // prevent the optimizer from eliding the work

    auto t0 = std::chrono::high_resolution_clock::now();

    while (n < max_rows && std::getline(file, line)) {
        auto f = split_csv_line(line);
        if (f.size() < cols.size()) continue;

        BookSnapshot snap;
        snap.timestamp = std::stoull(f[c_ts]);
        for (int l = 0; l < kDepth; ++l) {
            snap.bids[l] = {std::stod(f[c_bid_px[l]]), std::stod(f[c_bid_sz[l]])};
            snap.asks[l] = {std::stod(f[c_ask_px[l]]), std::stod(f[c_ask_sz[l]])};
        }

        FeatureVector fv = engine.on_snapshot(snap);
        checksum += fv.values[0];
        n++;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "rows_processed=" << n << "\n";
    std::cout << "elapsed_sec=" << elapsed_sec << "\n";
    std::cout << "rows_per_sec=" << (n / elapsed_sec) << "\n";
    std::cout << "avg_latency_ns_per_row=" << (elapsed_sec * 1e9 / n) << "\n";
    std::cout << "checksum=" << checksum << "\n";  // ignore; anti-optimizer guard
    return 0;
}
