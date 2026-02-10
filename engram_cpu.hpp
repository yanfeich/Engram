#ifndef ENGRAM_CPU_H
#define ENGRAM_CPU_H

#include <vector>
#include <cstdint>
#include <memory>
#include <string>
#include <torch/extension.h>

class EngramCPU {
public:
    struct Config {
        std::vector<int64_t> engram_vocab_size;
        int64_t max_ngram_size;
        int64_t n_embed_per_ngram;
        int64_t n_head_per_ngram;
        std::vector<int64_t> layer_ids;
        int64_t pad_id;
        int64_t seed;
        int64_t kernel_size;
        int64_t hidden_size;
        int64_t engram_hidden_size;
        int64_t hc_mult;
        int64_t tokenizer_vocab_size;
    };

    EngramCPU(
        int64_t layer_id,
        const Config& config,
        const std::vector<int64_t>& lookup_table,
        const std::vector<int64_t>& multipliers,
        const std::vector<std::vector<int64_t>>& vocab_sizes_for_layer,
        const std::vector<int64_t>& offsets,
        const torch::Tensor& embedding_weights
    );

    void set_weights(const torch::Tensor& k_weight,
                     const torch::Tensor& k_bias,
                     const torch::Tensor& k_norm_weight,
                     const torch::Tensor& v_weight,
                     const torch::Tensor& v_bias);
    
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& input_ids);
    //torch::Tensor forward(const torch::Tensor& input_ids);

private:
    int64_t layer_id_;
    Config config_;
    
    // Lookup table
    std::vector<int64_t> lookup_table_;
    
    std::vector<int64_t> multipliers_;
    std::vector<std::vector<int64_t>> vocab_sizes_for_layer_;
    
    // Offsets for multi-head embedding
    std::vector<int64_t> offsets_;
    
    // weights
    torch::Tensor embedding_weights_;
    torch::Tensor v_weight_;
    torch::Tensor v_bias_;
    torch::Tensor k_weights_;
    torch::Tensor k_biases_;
    torch::Tensor k_norm_weights_;

    // Helper methods
    std::vector<int64_t> compressed_tokenizer(const std::vector<int64_t>& input_ids, int64_t B, int64_t T);
    std::vector<std::vector<int64_t>> get_ngram_hashes(
        const std::vector<int64_t>& compressed_ids,
        int64_t B,
        int64_t T
    );
    torch::Tensor multi_head_embedding(const std::vector<std::vector<int64_t>>& hash_ids, int64_t B, int64_t T);
    torch::Tensor rms_norm(const torch::Tensor& x, const torch::Tensor& weight);
};

#endif // ENGRAM_CPU_H