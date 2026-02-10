#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "engram_cpu.hpp"

namespace py = pybind11;

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
        .def_readwrite("tokenizer_vocab_size", &EngramCPU::Config::tokenizer_vocab_size);
    
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