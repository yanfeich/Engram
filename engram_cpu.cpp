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

void EngramCPU::set_weights(const torch::Tensor& k_weight,
                            const torch::Tensor& k_bias,
                            const torch::Tensor& k_norm_weight,
                            const torch::Tensor& v_weight,
                            const torch::Tensor& v_bias) {
    k_weights_ = k_weight.cpu().contiguous();
    k_biases_ = k_bias.cpu().contiguous();
    k_norm_weights_ = k_norm_weight.cpu().contiguous();
    v_weight_ = v_weight.cpu().contiguous();
    v_bias_ = v_bias.cpu().contiguous();
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

torch::Tensor EngramCPU::multi_head_embedding(const std::vector<std::vector<int64_t>>& hash_ids,
    int64_t B,
    int64_t T) {
    int64_t D = config_.n_embed_per_ngram / config_.n_head_per_ngram;
    int64_t num_heads = hash_ids.size();
    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(embedding_weights_.dtype())
        .device(torch::kCPU);
    
    torch::Tensor output = torch::zeros({B, T, num_heads * D}, options);
    
#define VER_2
#ifdef VER_1
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
#elif defined(VER_2)    
    auto embedding_ptr = embedding_weights_.data_ptr<float>();
    auto output_ptr = output.data_ptr<float>();
    int64_t embedding_stride = embedding_weights_.size(1);
    
    #pragma omp parallel for
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t h = 0; h < num_heads; ++h) {
                int64_t hash_val = hash_ids[h][b * T +t];
                int64_t offset_hash = hash_val + offsets_[h];

                if (offset_hash >= 0 && offset_hash < embedding_weights_.size(0)) {
                    float* src = embedding_ptr + offset_hash * embedding_stride;
                    float* dst = output_ptr + (b * T * num_heads * D) +(t * num_heads * D) + (h * D);
                    std::memcpy(dst, src, D * sizeof(float));
                }
            }
        }
    }
#endif
    
    return output;
}

torch::Tensor EngramCPU::rms_norm(const torch::Tensor& x, const torch::Tensor& weight) {
    return rms_norm_impl(x, weight);
}

#if 0
std::pair<torch::Tensor, torch::Tensor> EngramCPU::forward(const torch::Tensor& input_ids) {
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
    // std::cout << "Input IDs loaded into vector." << std::endl;
    
    // Step 1: Compressed tokenizer
    auto compressed_ids = compressed_tokenizer(input_vec, B, T);
    // std::cout << "Compressed tokenizer." << std::endl;
    
    // Step 2: Get n-gram hashes
    auto hash_ids = get_ngram_hashes(compressed_ids, B, T);
    // std::cout << "Get n-gram hashes." << std::endl;
    
    // Step 3: Multi-head embedding
    auto embeddings = multi_head_embedding(hash_ids, B, T);
    //auto emb_slice = embeddings.index({torch::indexing::Slice(0, 3), 0, torch::indexing::Slice(0, 8)});
    // std::cout << "embeddings[0:3, 0, 0:8]:\n" << emb_slice << std::endl;
    
    auto total_elements = B * T;
    // 1. 计算value投影
    torch::Tensor value;
    {
        // 重塑为2D: [B*T, engram_hidden_size]
        auto embeddings_2d = embeddings.view({total_elements, config_.engram_hidden_size});
        // std::cout << "embeddings_2d[0:3, 0:8]:\n" << embeddings_2d.index({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 8)}) << std::endl;
        
        // value = embeddings @ v_weight^T + v_bias
        value = torch::addmm(v_bias_, embeddings_2d, v_weight_.t());
        
        value = value.view({B, T, 1, config_.hidden_size});
    }
    // std::cout << "Value projection done." << std::endl;
    // std::cout << "value shape: " << value.sizes() << std::endl;
    
    // 2. 为normed_keys预分配内存: [hc_mult, B, T, hidden_size]
    auto normed_keys = torch::empty({config_.hc_mult, B, T, config_.hidden_size}, 
                                    torch::TensorOptions().dtype(embeddings.dtype()));
    
    // 3. 并行处理每个head
    at::parallel_for(0, config_.hc_mult, 1, [&](int64_t start_hc, int64_t end_hc) {
        for (int hc_idx = start_hc; hc_idx < end_hc; ++hc_idx) {
            // 重塑输入为2D
            auto embeddings_2d = embeddings.view({total_elements, config_.engram_hidden_size});
            
            // 预分配key的内存
            auto keys = torch::empty({total_elements, config_.hidden_size}, 
                                     torch::TensorOptions().dtype(embeddings.dtype()));
            
            // 并行计算key投影
            int64_t grain_size = 256;  // 每个线程处理的最小元素数
            at::parallel_for(0, total_elements, grain_size, [&](int64_t start, int64_t end) {
                auto k_weight_h = k_weights_[hc_idx];
                auto k_bias_h = k_biases_[hc_idx];
                
                auto batch_embeddings = embeddings_2d.slice(0, start, end);
                auto batch_keys = torch::addmm(k_bias_h, batch_embeddings, k_weight_h.t());
                keys.slice(0, start, end) = batch_keys;
            });
            // std::cout << "Key projection for head " << hc_idx << " done." << std::endl;
            
            // 重塑keys为3D: [B, T, hidden_size]
            auto keys_3d = keys.view({B, T, config_.hidden_size});
            
            // 并行计算RMSNorm
            auto norm_weight_h = k_norm_weights_[hc_idx];
            auto normed_keys_h = torch::empty({B, T, config_.hidden_size}, 
                                              torch::TensorOptions().dtype(embeddings.dtype()));
            
            at::parallel_for(0, B, 1, [&](int64_t start_b, int64_t end_b) {
                for (int64_t b = start_b; b < end_b; ++b) {
                    auto normed_keys_b = torch::empty({T, config_.hidden_size}, 
                                                     torch::TensorOptions().dtype(embeddings.dtype()));
                    
                    for (int64_t l = 0; l < T; ++l) {
                        auto key_slice = keys_3d[b][l];
                        auto variance = key_slice.mul(key_slice).mean(-1, true);
                        auto inv_rms = torch::rsqrt(variance + 1e-6);
                        normed_keys_b[l] = key_slice * inv_rms * norm_weight_h;
                    }
                    
                    normed_keys_h[b] = normed_keys_b;
                }
            });
            // std::cout << "RMSNorm for head " << hc_idx << " done." << std::endl;
            
            // 存储到结果中
            normed_keys[hc_idx] = normed_keys_h;
        }
    });
    
    // std::cout << "value shape: " << value.sizes() << std::endl;
    // std::cout << "normed_keys shape: " << normed_keys.sizes() << std::endl;
    return {value, normed_keys};
}
#else

static int64_t call_count = 0;
static double t_copy = 0.0, t_compress = 0.0, t_hash = 0.0, t_embed = 0.0, t_vproj = 0.0, t_kproj = 0.0, t_knorm = 0.0;
std::pair<torch::Tensor, torch::Tensor> EngramCPU::forward(const torch::Tensor& input_ids) {
    using Clock = std::chrono::high_resolution_clock;

    auto t_start = Clock::now();
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
    auto t_after_copy = Clock::now();
    t_copy += std::chrono::duration<double, std::milli>(t_after_copy - t_start).count();
    
    // Step 1: Compressed tokenizer
    auto t1 = Clock::now();
    auto compressed_ids = compressed_tokenizer(input_vec, B, T);
    auto t2 = Clock::now();
    t_compress += std::chrono::duration<double, std::milli>(t2 - t1).count();
    
    // Step 2: Get n-gram hashes
    auto t3 = Clock::now();
    auto hash_ids = get_ngram_hashes(compressed_ids, B, T);
    auto t4 = Clock::now();
    t_hash += std::chrono::duration<double, std::milli>(t4 - t3).count();
    
    // Step 3: Multi-head embedding
    auto t5 = Clock::now();
    auto embeddings = multi_head_embedding(hash_ids, B, T);
    auto t6 = Clock::now();
    t_embed += std::chrono::duration<double, std::milli>(t6 - t5).count();
    

    auto total_elements = B * T;
    // 1. 计算value投影
    torch::Tensor value;

    // 重塑为2D: [B*T, engram_hidden_size]
    auto embeddings_2d = embeddings.view({total_elements, config_.engram_hidden_size});
    // std::cout << "embeddings_2d[0:3, 0:8]:\n" << embeddings_2d.index({torch::indexing::Slice(0,3), torch::indexing::Slice(0, 8)}) << std::endl;
    
    // value = embeddings @ v_weight^T + v_bias
    auto t7 = Clock::now();
    value = torch::addmm(v_bias_, embeddings_2d, v_weight_.t());
    auto t8 = Clock::now();
    t_vproj += std::chrono::duration<double, std::milli>(t8 - t7).count();
    
    value = value.view({B, T, 1, config_.hidden_size});

    // std::cout << "Value projection done." << std::endl;
    // std::cout << "value shape: " << value.sizes() << std::endl;
    
    // 2. 为normed_keys预分配内存: [hc_mult, B, T, hidden_size]
    auto normed_keys = torch::empty({config_.hc_mult, B, T, config_.hidden_size}, 
                                    torch::TensorOptions().dtype(embeddings.dtype()));
    
    // 3. 并行处理每个head
    at::parallel_for(0, config_.hc_mult, 1, [&](int64_t start_hc, int64_t end_hc) {
        for (int hc_idx = start_hc; hc_idx < end_hc; ++hc_idx) {
            // 重塑输入为2D
            auto embeddings_2d = embeddings.view({total_elements, config_.engram_hidden_size});
            
            // 预分配key的内存
            auto keys = torch::empty({total_elements, config_.hidden_size}, 
                                     torch::TensorOptions().dtype(embeddings.dtype()));
            
            // 并行计算key投影
            auto t9 = Clock::now();
            int64_t grain_size = 256;  // 每个线程处理的最小元素数
            at::parallel_for(0, total_elements, grain_size, [&](int64_t start, int64_t end) {
                auto k_weight_h = k_weights_[hc_idx];
                auto k_bias_h = k_biases_[hc_idx];
                
                auto batch_embeddings = embeddings_2d.slice(0, start, end);
                auto batch_keys = torch::addmm(k_bias_h, batch_embeddings, k_weight_h.t());
                keys.slice(0, start, end) = batch_keys;
            });
            auto t10 = Clock::now();
            t_kproj += std::chrono::duration<double, std::milli>(t10 - t9).count();
            // std::cout << "Key projection for head " << hc_idx << " done." << std::endl;
            
            // 重塑keys为3D: [B, T, hidden_size]
            auto keys_3d = keys.view({B, T, config_.hidden_size});
            
            // 并行计算RMSNorm
            auto norm_weight_h = k_norm_weights_[hc_idx];
            auto normed_keys_h = torch::empty({B, T, config_.hidden_size}, 
                                              torch::TensorOptions().dtype(embeddings.dtype()));
            
            auto t11 = Clock::now();
            at::parallel_for(0, B, 1, [&](int64_t start_b, int64_t end_b) {
                for (int64_t b = start_b; b < end_b; ++b) {
                    auto normed_keys_b = torch::empty({T, config_.hidden_size}, 
                                                     torch::TensorOptions().dtype(embeddings.dtype()));
                    
                    for (int64_t l = 0; l < T; ++l) {
                        auto key_slice = keys_3d[b][l];
                        auto variance = key_slice.mul(key_slice).mean(-1, true);
                        auto inv_rms = torch::rsqrt(variance + 1e-6);
                        normed_keys_b[l] = key_slice * inv_rms * norm_weight_h;
                    }
                    
                    normed_keys_h[b] = normed_keys_b;
                }
            });
            auto t12 = Clock::now();
            t_knorm += std::chrono::duration<double, std::milli>(t12 - t11).count();
            // std::cout << "RMSNorm for head " << hc_idx << " done." << std::endl;
            
            // 存储到结果中
            normed_keys[hc_idx] = normed_keys_h;
        }
    });
    

    ++call_count;
    if (call_count % 500 == 0) {
        std::cout << "[forward_profile] avg(ms) copy=" << (t_copy / call_count)
                  << ", compress=" << (t_compress / call_count)
                  << ", hash=" << (t_hash / call_count)
                  << ", embed=" << (t_embed / call_count)
                  << ", vproj=" << (t_vproj / call_count)
                  << ", kproj=" << (t_kproj / call_count)
                  << ", knorm=" << (t_knorm / call_count)
                  << ", total=" << ((t_copy + t_compress + t_hash + t_embed + t_vproj + t_kproj + t_knorm) / call_count)
                  << std::endl;
    }
    
    // std::cout << "value shape: " << value.sizes() << std::endl;
    // std::cout << "normed_keys shape: " << normed_keys.sizes() << std::endl;
    return {value, normed_keys};
}
#endif