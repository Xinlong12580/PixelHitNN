#include <iostream>
#include <string>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"

std::string PrefixTensorName(const std::string& input_name,
                             const std::string& prefix) {
    // TensorFlow input strings can look like:
    //   node
    //   node:0
    //   ^control_node
    //
    // We want:
    //   prefix/node
    //   prefix/node:0
    //   ^prefix/control_node

    if (input_name.empty()) return input_name;

    bool is_control = false;
    std::string s = input_name;

    if (s[0] == '^') {
        is_control = true;
        s = s.substr(1);
    }

    // Split possible output suffix, e.g. "MatMul:0"
    std::string node_part = s;
    std::string suffix;

    size_t colon_pos = s.rfind(':');
    if (colon_pos != std::string::npos) {
        node_part = s.substr(0, colon_pos);
        suffix = s.substr(colon_pos);
    }

    std::string out = prefix + "/" + node_part + suffix;

    if (is_control) {
        out = "^" + out;
    }

    return out;
}

void PrefixColocationAttrs(tensorflow::NodeDef* node,
                           const std::string& prefix) {
    // Some TensorFlow graphs have node.attr["_class"].list.s entries like:
    //   "loc:@some_node"
    //
    // If we rename nodes, these should also be renamed:
    //   "loc:@gx/some_node"

    auto it = node->mutable_attr()->find("_class");
    if (it == node->mutable_attr()->end()) return;

    auto* list = it->second.mutable_list();

    for (int i = 0; i < list->s_size(); ++i) {
        std::string value = list->s(i);

        const std::string marker = "loc:@";
        if (value.rfind(marker, 0) == 0) {
            std::string old_node = value.substr(marker.size());
            std::string new_value = marker + prefix + "/" + old_node;
            list->set_s(i, new_value);
        }
    }
}

void AddPrefixedGraph(const tensorflow::GraphDef& src,
                      const std::string& prefix,
                      tensorflow::GraphDef* dst) {
    for (const auto& node : src.node()) {
        tensorflow::NodeDef* new_node = dst->add_node();
        *new_node = node;

        // Rename node itself
        new_node->set_name(prefix + "/" + node.name());

        // Rename all inputs
        for (int i = 0; i < new_node->input_size(); ++i) {
            new_node->set_input(
                i,
                PrefixTensorName(new_node->input(i), prefix)
            );
        }

        // Rename colocation constraints if present
        PrefixColocationAttrs(new_node, prefix);
    }
}

bool LoadGraphDef(const std::string& path, tensorflow::GraphDef* graph) {
    auto status = tensorflow::ReadBinaryProto(
        tensorflow::Env::Default(),
        path,
        graph
    );

    if (!status.ok()) {
        std::cerr << "Failed to read graph: " << path << "\n"
                  << status.ToString() << std::endl;
        return false;
    }

    return true;
}

bool SaveGraphDef(const std::string& path,
                  const tensorflow::GraphDef& graph) {
    auto status = tensorflow::WriteBinaryProto(
        tensorflow::Env::Default(),
        path,
        graph
    );

    if (!status.ok()) {
        std::cerr << "Failed to write graph: " << path << "\n"
                  << status.ToString() << std::endl;
        return false;
    }

    return true;
}

bool MergeTwoGraphs(const std::string& graph1_path,
                    const std::string& graph2_path,
                    const std::string& output_path,
                    const std::string& prefix1 = "gx",
                    const std::string& prefix2 = "gy") {
    tensorflow::GraphDef graph1;
    tensorflow::GraphDef graph2;
    tensorflow::GraphDef merged;

    if (!LoadGraphDef(graph1_path, &graph1)) return false;
    if (!LoadGraphDef(graph2_path, &graph2)) return false;

    AddPrefixedGraph(graph1, prefix1, &merged);
    AddPrefixedGraph(graph2, prefix2, &merged);

    if (!SaveGraphDef(output_path, merged)) return false;

    std::cout << "Merged graph written to: " << output_path << "\n";
    std::cout << "Graph1 nodes: " << graph1.node_size() << "\n";
    std::cout << "Graph2 nodes: " << graph2.node_size() << "\n";
    std::cout << "Merged nodes: " << merged.node_size() << "\n";

    return true;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0]
                  << " graph1.pb graph2.pb merged.pb\n";
        return 1;
    }

    bool ok = MergeTwoGraphs(
        argv[1],
        argv[2],
        argv[3],
        "gx",
        "gy"
    );

    return ok ? 0 : 1;
}
