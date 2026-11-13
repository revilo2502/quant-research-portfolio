// Loads models/clf_h1.onnx (exported by python/train.py --export-onnx) and
// runs inference via the ONNX Runtime C++ API. This is the "train in
// Python, infer in C++" pattern that makes the low-latency path realistic:
// you get LightGBM/sklearn's training convenience without paying Python's
// interpreter overhead at prediction time.
//
// Build only happens if ONNX Runtime is found (see CMakeLists.txt) --
// everything else in this repo works without it.
//
// Usage: ./infer_onnx ../../models/clf_h1.onnx <feature values...>

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <onnxruntime_cxx_api.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> [feature_values...]\n";
        return 1;
    }
    std::string model_path = argv[1];

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "hft-infer");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, model_path.c_str(), opts);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    int64_t n_features = input_shape.back();
    if (n_features < 0) n_features = argc > 2 ? (argc - 2) : 15;

    std::vector<float> features(n_features, 0.0f);
    for (int i = 2; i < argc && (i - 2) < n_features; ++i) {
        features[i - 2] = std::stof(argv[i]);
    }

    std::array<int64_t, 2> shape{1, n_features};
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, features.data(), features.size(), shape.data(), shape.size());

    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    // warmup
    session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    constexpr int kIters = 10000;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIters; ++i) {
        auto out = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;

    auto final_out = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    std::cout << "avg_inference_latency_us=" << avg_us << "\n";
    std::cout << "iterations=" << kIters << "\n";
    return 0;
}
