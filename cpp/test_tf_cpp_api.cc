#include <iostream>
#include <memory>
#include <vector>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/graph.pb.h"

int main() {
    std::cout << "Testing TensorFlow C++ API\n";
    tensorflow::SessionOptions options;
    options.config.set_allow_soft_placement(true);
    options.config.set_log_device_placement(true);
    options.config.mutable_gpu_options()->set_allow_growth(true);

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(options)
    );

    if (!session) {
        std::cerr << "Failed to create TensorFlow session\n";
        return 1;
    }

    std::cout << "Session created successfully\n";

    // Build a tiny graph: C = A * B
    tensorflow::GraphDef graph;

    auto* a = graph.add_node();
    a->set_name("A");
    a->set_op("Placeholder");
    (*a->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);

    auto* b = graph.add_node();
    b->set_name("B");
    b->set_op("Placeholder");
    (*b->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);

    auto* matmul = graph.add_node();
    matmul->set_name("MatMul");
    matmul->set_op("MatMul");
    matmul->add_input("A");
    matmul->add_input("B");
    (*matmul->mutable_attr())["T"].set_type(tensorflow::DT_FLOAT);
    (*matmul->mutable_attr())["transpose_a"].set_b(false);
    (*matmul->mutable_attr())["transpose_b"].set_b(false);

    // Force GPU placement.
    matmul->set_device("/device:GPU:0");

    auto status = session->Create(graph);
    if (!status.ok()) {
        std::cerr << "session->Create failed:\n"
                  << status.ToString() << "\n";
        return 1;
    }

    const int N = 4096;

    tensorflow::Tensor A(tensorflow::DT_FLOAT, tensorflow::TensorShape({N, N}));
    tensorflow::Tensor B(tensorflow::DT_FLOAT, tensorflow::TensorShape({N, N}));

    auto A_flat = A.flat<float>();
    auto B_flat = B.flat<float>();

    for (int i = 0; i < A_flat.size(); ++i) A_flat(i) = 1.0f;
    for (int i = 0; i < B_flat.size(); ++i) B_flat(i) = 1.0f;

    std::vector<tensorflow::Tensor> outputs;

    status = session->Run(
        {{"A:0", A}, {"B:0", B}},
        {"MatMul:0"},
        {},
        &outputs
    );

    if (!status.ok()) {
        std::cerr << "session->Run failed:\n"
                  << status.ToString() << "\n";
        return 1;
    }

    std::cout << "Run succeeded\n";
    std::cout << "Output shape: "
              << outputs[0].shape().DebugString() << "\n";
    std::cout << "First value: "
              << outputs[0].flat<float>()(0) << "\n";

    session->Close();
    return 0;
}
                             
