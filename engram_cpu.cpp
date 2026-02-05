#include "engram_cpu.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

// Helper function for RMSNorm
torch::Tensor rms_norm_impl(const torch::Tensor& x, const torch::Tensor& weight) {
    auto variance = x.pow(2).mean(-1, true);
    auto x_normed = x * torch::rsqrt(variance + 1e-6);
    return x_normed * weight;
}

EngramCPU::EngramCPU(
    int64_t layer_id,
    const Config& config,
    const std::vector<int64_t>& lookup_table,
    const std::vector<int64_t>& multipliers,
    const std::vector<std::vector<int64_t>>& vocab_sizes_for_layer,
    const std::vector<int64_t>& offsets,
    const torch::Tensor& embedding_weights
) : layer_id_(layer_id),
    config_(config),
    lookup_table_(lookup_table),
    multipliers_(multipliers),
    vocab_sizes_for_layer_(vocab_sizes_for_layer),
    offsets_(offsets),
    embedding_weights_(embedding_weights) {
    
    if (multipliers_.size() != static_cast<size_t>(config_.max_ngram_size)) {
        std::string msg = "Multipliers size mismatch. Expected " + 
                         std::to_string(config_.max_ngram_size) + 
                         ", got " + std::to_string(multipliers_.size());
        throw std::runtime_error(msg);
    }
}

std::vector<int64_t> EngramCPU::compressed_tokenizer(const std::vector<int64_t>& input_ids, int64_t B, int64_t T) {
    std::vector<int64_t> result(input_ids.size());
    int64_t vocab_size = static_cast<int64_t>(lookup_table_.size());
    
    #pragma omp parallel for collapse(2) if (B * T > 1000)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            int64_t idx = b * T + t;
            int64_t token_id = input_ids[idx];
            if (token_id >= 0 && token_id < vocab_size) {
                result[idx] = lookup_table_[token_id];
            } else {
                result[idx] = token_id; // Keep as is if out of bounds
            }
        }
    }
    
    return result;
}

std::vector<std::vector<int64_t>> EngramCPU::get_ngram_hashes(
    const std::vector<int64_t>& compressed_ids,
    int64_t B,
    int64_t T
) {
    std::vector<std::vector<int64_t>> shifts(config_.max_ngram_size);
    for (int64_t k = 0; k < config_.max_ngram_size; ++k) {
        shifts[k].resize(B * T);
        if (k == 0) {
            std::memcpy(shifts[k].data(), compressed_ids.data(), B * T * sizeof(int64_t));
        } else {
            #pragma omp parallel for collapse(2) if (B * T > 1000)
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t t = 0; t < T; ++t) {
                    int64_t idx = b * T + t;
                    if (t < k) {
                        shifts[k][idx] = config_.pad_id;
                    } else {
                        shifts[k][idx] = compressed_ids[b * T + (t - k)];
                    }
                }
            }
        }
    }
    
    // Calculate total number of heads
    int64_t total_heads = 0;
    for (int64_t n = 2; n <= config_.max_ngram_size; ++n) {
        total_heads += config_.n_head_per_ngram;
    }
    
    std::vector<std::vector<int64_t>> all_hashes(total_heads, std::vector<int64_t>(B * T));
    
    int64_t head_idx = 0;
    for (int64_t n = 2; n <= config_.max_ngram_size; ++n) {
        int64_t ngram_index = n - 2;
        const auto& head_vocab_sizes = vocab_sizes_for_layer_[ngram_index];
        
        #pragma omp parallel for collapse(2) if (B * T > 1000)
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t t = 0; t < T; ++t) {
                int64_t idx = b * T + t;
                
                // 计算mix
                int64_t mix = shifts[0][idx] * multipliers_[0];
                for (int64_t k = 1; k < n; ++k) {
                    mix = mix ^ (shifts[k][idx] * multipliers_[k]);
                }
                
                // 为每个头计算hash
                for (int64_t j = 0; j < config_.n_head_per_ngram; ++j) {
                    int64_t mod = head_vocab_sizes[j];
                    // 处理负数
                    int64_t positive_mix = mix;
                    if (mix < 0) {
                        positive_mix = mod - ((-mix) % mod);
                        if (positive_mix == mod) positive_mix = 0;
                    }
                    int64_t head_hash = positive_mix % mod;
                    all_hashes[head_idx + j][idx] = head_hash;
                }
            }
        }
        head_idx += config_.n_head_per_ngram;
    }
    
    return all_hashes;
}

torch::Tensor EngramCPU::multi_head_embedding(const std::vector<std::vector<int64_t>>& hash_ids) {
    int64_t B = hash_ids[0].size() / (hash_ids[0].size() / hash_ids.size()); // Approximate, will fix
    int64_t T = hash_ids[0].size() / B;
    int64_t num_heads = hash_ids.size();
    
    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(embedding_weights_.dtype())
        .device(torch::kCPU);
    
    int64_t D = config_.n_embed_per_ngram / config_.n_head_per_ngram;
    torch::Tensor output = torch::zeros({B, T, num_heads * D}, options);
    
    auto embedding_accessor = embedding_weights_.accessor<float, 2>();
    auto output_accessor = output.accessor<float, 3>();
    
    #pragma omp parallel for collapse(3) if (B * T * num_heads > 1000)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t h = 0; h < num_heads; ++h) {
                int64_t hash_val = hash_ids[h][b * T + t];
                int64_t offset_hash = hash_val + offsets_[h];
                
                for (int64_t d = 0; d < D; ++d) {
                    output_accessor[b][t][h * D + d] = embedding_accessor[offset_hash][d];
                }
            }
        }
    }
    
    return output;
}

torch::Tensor EngramCPU::rms_norm(const torch::Tensor& x, const torch::Tensor& weight) {
    return rms_norm_impl(x, weight);
}

torch::Tensor EngramCPU::forward(const torch::Tensor& input_ids) {
    // Ensure input is on CPU
    auto input_cpu = input_ids.to(torch::kCPU);
    
    // Get dimensions
    auto sizes = input_cpu.sizes();
    int64_t B = sizes[0];
    int64_t T = sizes[1];
    
    // Convert to vector for processing
    std::vector<int64_t> input_vec(B * T);
    auto input_accessor = input_cpu.accessor<int64_t, 2>();
    
    #pragma omp parallel for collapse(2) if (B * T > 1000)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            input_vec[b * T + t] = input_accessor[b][t];
        }
    }
    
    // Step 1: Compressed tokenizer
    auto compressed_ids = compressed_tokenizer(input_vec, B, T);
    
    // Step 2: Get n-gram hashes
    auto hash_ids = get_ngram_hashes(compressed_ids, B, T);
    
    // Step 3: Multi-head embedding
    auto embeddings = multi_head_embedding(hash_ids);
    
    // Reshape embeddings if needed
    embeddings = embeddings.reshape({B, T, -1});
    
    return embeddings;
}