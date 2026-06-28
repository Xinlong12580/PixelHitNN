#include <iostream>
#include <vector>
#include <string>
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include<chrono>
#include<iostream>
#include<fstream>
double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p * (v.size() - 1));
    return v[idx];
}

int main() {

    int batch_size = 200;
    int input_size_y = 21;
    int input_size_x = 13;
    std::string graph_path = "../PixelHitNN/checkpoints/L1F.keras.pb";

    // Load GraphDef
    tensorflow::GraphDef graph_def;
    tensorflow::Status status = tensorflow::ReadBinaryProto(
        tensorflow::Env::Default(),
        graph_path,
        &graph_def
    );

    if (!status.ok()) {
        std::cerr << "Failed to load graph: " << status.ToString() << std::endl;
        return 1;
    }

    // Create session options
    tensorflow::SessionOptions options;
    options.config.mutable_graph_options()
    ->mutable_optimizer_options()
    ->set_global_jit_level(tensorflow::OptimizerOptions::ON_1);
    auto* rewrite_options =options.config.mutable_graph_options()->mutable_rewrite_options();

    rewrite_options->set_constant_folding(tensorflow::RewriterConfig::ON);
    rewrite_options->set_layout_optimizer(tensorflow::RewriterConfig::ON);
    rewrite_options->set_function_optimization(tensorflow::RewriterConfig::ON);
    rewrite_options->set_dependency_optimization(tensorflow::RewriterConfig::ON);
    rewrite_options->set_arithmetic_optimization(tensorflow::RewriterConfig::ON);
    rewrite_options->set_loop_optimization(tensorflow::RewriterConfig::ON);
    rewrite_options->set_shape_optimization(tensorflow::RewriterConfig::ON);
    rewrite_options->set_remapping(tensorflow::RewriterConfig::ON);
    rewrite_options->set_memory_optimization(tensorflow::RewriterConfig::MANUAL);
    std::cout<<"using grappler"<<std::endl;
    // For CPU:
    // nothing special needed

    // For GPU, if your TensorFlow C++ build supports CUDA:
    options.config.set_allow_soft_placement(true);
    options.config.mutable_gpu_options()->set_allow_growth(true);
    options.config.mutable_gpu_options()->set_visible_device_list("0");
    //options.config.mutable_gpu_options()->set_visible_device_list("");
    // Optional debug:
    options.config.set_log_device_placement(false);

    // Create session
    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(options)
    );

    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cerr << "Failed to create session: " << status.ToString() << std::endl;
        return 1;
    }

    // Example input tensor
    // Change shape to match your real model input.
    tensorflow::Tensor input_tensor_x(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape({batch_size, input_size_x, 1})
    );

    auto input_x = input_tensor_x.tensor<float, 3>();
    for (int n = 0; n < batch_size; n++){
	    for (int i = 0; i < input_size_x; ++i) {
		input_x(n, i, 0) = 0.0;
	    }
    }

    // If your model has multiple inputs, define them too.
    tensorflow::Tensor angles_tensor_x(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape({batch_size, 2})
    );

    auto angles_x = angles_tensor_x.tensor<float, 2>();
	for (int n = 0; n < batch_size; n++){
	    angles_x(n, 0) = 0.0;
	    angles_x(n, 1) = 0.0;
	}
    tensorflow::Tensor charge_tensor_x(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape({batch_size, 1})
    );
    auto charge_x = charge_tensor_x.tensor<float, 2>();
	for (int n = 0; n < batch_size; n++){
    		charge_x(n, 0) = 1.0;
	}
    



    tensorflow::Tensor input_tensor_y(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape({batch_size, input_size_y, 1})
    );

    auto input_y = input_tensor_y.tensor<float, 3>();
    for (int n = 0; n < batch_size; n++){
	    for (int i = 0; i < input_size_y; ++i) {
		input_y(n, i, 0) = 0.0;
	    }
    }

    // If your model has multiple inputs, define them too.
    tensorflow::Tensor angles_tensor_y(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape({batch_size, 2})
    );

    auto angles_y = angles_tensor_y.tensor<float, 2>();
	for (int n = 0; n < batch_size; n++){
	    angles_y(n, 0) = 0.0;
	    angles_y(n, 1) = 0.0;
	}
    tensorflow::Tensor charge_tensor_y(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape({batch_size, 1})
    );
    auto charge_y = charge_tensor_y.tensor<float, 2>();
	for (int n = 0; n < batch_size; n++){
    		charge_y(n, 0) = 1.0;
    }

    // Run inference.
    // Replace these names with your real graph input/output tensor names.
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
        {"gx/pixel_projection_x:0", input_tensor_x},
        {"gx/angles:0", angles_tensor_x},
        {"gx/cluster_charge:0", charge_tensor_x},
        {"gy/pixel_projection_y:0", input_tensor_y},
        {"gy/angles:0", angles_tensor_y},
        {"gy/cluster_charge:0", charge_tensor_y}
    };

    std::vector<std::string> output_names = {
        "gx/Identity:0",
        "gy/Identity:0",
    };

    std::vector<tensorflow::Tensor> outputs;


    const int n_warmup = 50;
    const int n_runs   = 1000;

    // Warmup: triggers graph placement, CUDA context init, cuDNN/cuBLAS setup, memory allocation, etc.
    for (int i = 0; i < n_warmup; ++i) {
        auto status = session->Run(inputs, output_names, {}, &outputs);
        if (!status.ok()) {
            std::cerr << "Warmup failed: " << status.ToString() << std::endl;
            return 1;
        }
    }

    std::vector<double> times_ns;
    times_ns.reserve(n_runs);

    for (int i = 0; i < n_runs; ++i) {
        outputs.clear();
        auto t0 = std::chrono::steady_clock::now();

        auto status = session->Run(inputs, output_names, {}, &outputs);

        auto t1 = std::chrono::steady_clock::now();

        if (!status.ok()) {
            std::cerr << "Run failed: " << status.ToString() << std::endl;
            return 1;
        }

        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        times_ns.push_back(ns);
    }
    long long sum = std::accumulate(times_ns.begin(), times_ns.end(), 0LL);
    double mean_ns = static_cast<double>(sum) / times_ns.size();
    
    std::cout << "Batch size:      " << batch_size << "\n";
    std::cout << "Runs:      " << n_runs << "\n";
    std::cout << "Mean:      " << mean_ns << " ns\n";
    std::cout << "Median:    " << percentile(times_ns, 0.50) << " ns\n";
    std::cout << "p90:       " << percentile(times_ns, 0.90) << " ns\n";
    std::cout << "p99:       " << percentile(times_ns, 0.99) << " ns\n";

    std::ofstream file("output_merged.txt");

    if (!file.is_open()) {
        std::cerr << "Failed to open file\n";
        return 1;
    }

    for (double x : times_ns) {
        file << x << "\n";
    }

    file.close();
    
    std::cout << "Inference succeeded." << std::endl;
    std::cout << "Number of outputs: " << outputs.size() << std::endl;
    std::cout << "Output shape: " << outputs[0].shape().DebugString() << std::endl;

    
    session->Close();
    return 0;
}
