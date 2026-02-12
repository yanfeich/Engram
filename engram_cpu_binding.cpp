#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "engram_cpu.hpp"

namespace py = pybind11;

/*
// 包装器类，处理不同数据类型
class EngramCPUWrapper {
private:
    std::unique_ptr<EngramCPU<float>> fp32_impl_;
    std::unique_ptr<EngramCPU<uint16_t>> bf16_impl_;
    bool use_bf16_;
    
public:
    EngramCPUWrapper(
        int64_t layer_id,
        const Config& config,
        const std::vector<int64_t>& lookup_table,
        const std::vector<int64_t>& multipliers,
        const std::vector<std::vector<int64_t>>& vocab_sizes_for_layer,
        const std::vector<int64_t>& offsets,
        const torch::Tensor& embedding_weights) {
        std::cout<<"***********  Initializing EngramCPUWrapper with dtype: " << config.dtype << std::endl;
        use_bf16_ = (config.dtype == "bf16");
        if (config.dtype == "bf16") {
            bf16_impl_ = std::make_unique<EngramCPU<uint16_t>>(
                layer_id, config, lookup_table, multipliers, vocab_sizes_for_layer, offsets, embedding_weights);
        } else {
            fp32_impl_ = std::make_unique<EngramCPU<float>>(
                layer_id, config, lookup_table, multipliers, vocab_sizes_for_layer, offsets, embedding_weights);
        }
    }
    
    void set_weights(const torch::Tensor& k_weight, const torch::Tensor& k_bias,
                     const torch::Tensor& k_norm_weight,
                     const torch::Tensor& v_weight, const torch::Tensor& v_bias) {
        if (use_bf16_) {
            bf16_impl_->set_weights(k_weight, k_bias, k_norm_weight, v_weight, v_bias);
        } else {
            fp32_impl_->set_weights(k_weight, k_bias, k_norm_weight, v_weight, v_bias);
        }
    }
    
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& input_ids) {
        if (use_bf16_) {
            return bf16_impl_->forward(input_ids);
        } else {
            return fp32_impl_->forward(input_ids);
        }
    }
};
*/
PYBIND11_MODULE(engram_cpu, m) {
    py::class_<EngramCPU::Config>(m, "EngramConfig")
        .def(py::init<>())
        .def_readwrite("engram_vocab_size", &EngramCPU::Config::engram_vocab_size)
        .def_readwrite("max_ngram_size", &EngramCPU::Config::max_ngram_size)
        .def_readwrite("n_embed_per_ngram", &EngramCPU::Config::n_embed_per_ngram)
        .def_readwrite("n_head_per_ngram", &EngramCPU::Config::n_head_per_ngram)
        .def_readwrite("layer_ids", &EngramCPU::Config::layer_ids)
        .def_readwrite("pad_id", &EngramCPU::Config::pad_id)
        .def_readwrite("seed", &EngramCPU::Config::seed)
        .def_readwrite("kernel_size", &EngramCPU::Config::kernel_size)
        .def_readwrite("hidden_size", &EngramCPU::Config::hidden_size)
        .def_readwrite("engram_hidden_size", &EngramCPU::Config::engram_hidden_size)
        .def_readwrite("hc_mult", &EngramCPU::Config::hc_mult)
        .def_readwrite("tokenizer_vocab_size", &EngramCPU::Config::tokenizer_vocab_size)
        .def_readwrite("dtype", &EngramCPU::Config::dtype);
    
    py::class_<EngramCPU>(m, "EngramCPU")
        .def(py::init<
            int64_t,
            const EngramCPU::Config&,
            const std::vector<int64_t>&,
            const std::vector<int64_t>&,
            const std::vector<std::vector<int64_t>>&,
            const std::vector<int64_t>&,
            const torch::Tensor&
        >(), py::arg("layer_id"), py::arg("config"),
            py::arg("lookup_table"), py::arg("multipliers"),
            py::arg("vocab_sizes_for_layer"), py::arg("offsets"),
            py::arg("embedding_weights"))
        .def("set_weights", &EngramCPU::set_weights,
             py::arg("k_weight"),
             py::arg("k_bias"),
             py::arg("k_norm_weight"),
             py::arg("v_weight"),
             py::arg("v_bias"))
        .def("forward", &EngramCPU::forward, py::arg("input_ids"));
}