#include "engram_cpu.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>


// 优化的线性层
inline void linear_impl_simd(
    float* output,
    const float* input,
    const float* weight,  // [output_dim, input_dim]
    const float* bias,    // [output_dim]
    int64_t batch_size,
    int64_t output_dim,
    int64_t input_dim) {
    
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* batch_input = input + b * input_dim;
        float* batch_output = output + b * output_dim;
        
        #pragma omp simd
        for (int64_t i = 0; i < output_dim; ++i) {
            float sum = bias ? bias[i] : 0.0f;
            const float* weight_row = weight + i * input_dim;
            
            // 向量化点积
            #pragma omp simd reduction(+:sum)
            for (int64_t j = 0; j < input_dim; ++j) {
                sum += batch_input[j] * weight_row[j];
            }
            batch_output[i] = sum;
        }
    }
}

inline void linear_impl_avx512(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int64_t batch_size,
    int64_t output_dim,
    int64_t input_dim) {
    
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* batch_input = input + b * input_dim;
        float* batch_output = output + b * output_dim;
        
        for (int64_t i = 0; i < output_dim; ++i) {
            float sum = bias ? bias[i] : 0.0f;
            const float* weight_row = weight + i * input_dim;
            
            // 使用AVX512进行向量化点积
            int64_t j = 0;
            __m512 sum_vec = _mm512_setzero_ps();
            
            for (; j + 16 <= input_dim; j += 16) {
                __m512 input_vec = _mm512_load_ps(batch_input + j);
                __m512 weight_vec = _mm512_load_ps(weight_row + j);
                sum_vec = _mm512_fmadd_ps(input_vec, weight_vec, sum_vec);
            }
            
            // 水平求和
            sum += _mm512_reduce_add_ps(sum_vec);
            
            // 处理剩余元素
            for (; j < input_dim; ++j) {
                sum += batch_input[j] * weight_row[j];
            }
            
            batch_output[i] = sum;
        }
    }
}

//template <typename float_t>
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
    
    // 设置数据类型
    dtype_ = dnnl::memory::data_type::f32;
    /*
    if constexpr (std::is_same_v<float_t, float>) {
        dtype_ = memory::data_type::f32;
    } else if constexpr (std::is_same_v<float_t, dnnl::bfloat16>) {
        dtype_ = memory::data_type::bf16;
    } else {
        throw std::runtime_error("Unsupported data type");
    }
    */
    
    // 调整容器大小
    key_weights_.resize(config_.hc_mult);
    key_biases_.resize(config_.hc_mult);
    key_norm_weights_.resize(config_.hc_mult);
    key_matmul_pds_.resize(config_.hc_mult);
    key_matmul_primitives_.resize(config_.hc_mult);
    norm_pds_.resize(config_.hc_mult);
    norm_primitives_.resize(config_.hc_mult);
    key_matmul_args_.resize(config_.hc_mult);
    norm_args_.resize(config_.hc_mult);
    
    if (multipliers_.size() != static_cast<size_t>(config_.max_ngram_size)) {
        std::string msg = "Multipliers size mismatch. Expected " + 
                         std::to_string(config_.max_ngram_size) + 
                         ", got " + std::to_string(multipliers_.size());
        throw std::runtime_error(msg);
    }
}

//template <typename Type>
void EngramCPU::set_weights(const torch::Tensor& k_weight,
                            const torch::Tensor& k_bias,
                            const torch::Tensor& k_norm_weight,
                            const torch::Tensor& v_weight,
                            const torch::Tensor& v_bias) {
    // 获取Tensor数据指针
    auto k_weight_ptr = k_weight.data_ptr<float_t>();
    auto k_bias_ptr = k_bias.data_ptr<float_t>();
    auto k_norm_weight_ptr = k_norm_weight.data_ptr<float_t>();
    auto v_weight_ptr = v_weight.data_ptr<float_t>();
    auto v_bias_ptr = v_bias.data_ptr<float_t>();
    
    // 获取维度信息
    auto k_weight_sizes = k_weight.sizes();
    int engram_hidden_size = k_weight_sizes[1];
    int hidden_size = k_weight_sizes[0];

    std::cout << "Setting weights for layer " << layer_id_ << " with hidden_size=" << hidden_size 
              << ", engram_hidden_size=" << engram_hidden_size << std::endl;
        
    // 设置value权重 - 使用行优先布局
    memory::dims v_weight_dims = {config_.hidden_size, config_.engram_hidden_size};
    value_weight_.desc = memory::desc(v_weight_dims, dtype_, memory::format_tag::ba);
    value_weight_.memory = memory(value_weight_.desc, engine_, const_cast<float_t*>(v_weight_ptr));
    
    // 设置value偏置
    memory::dims v_bias_dims = {1, config_.hidden_size};
    value_bias_.desc = memory::desc(v_bias_dims, dtype_, memory::format_tag::ab);
    value_bias_.memory = memory(value_bias_.desc, engine_, const_cast<float_t*>(v_bias_ptr));
    
    // 为每个head设置key权重
    for (int hc_idx = 0; hc_idx < config_.hc_mult; ++hc_idx) {
        // key权重 - 行优先布局
        memory::dims k_weight_dims = {config_.hidden_size, config_.engram_hidden_size};
        key_weights_[hc_idx].desc = memory::desc(k_weight_dims, dtype_, memory::format_tag::ba);
        key_weights_[hc_idx].memory = memory(key_weights_[hc_idx].desc, engine_, 
                                            const_cast<float_t*>(k_weight_ptr) + hc_idx * config_.hidden_size * config_.engram_hidden_size);
        if (0) {
            const float_t* w_ptr = static_cast<const float_t*>(key_weights_[hc_idx].memory.get_data_handle());
            std::cout << "key_weights_[" << hc_idx << "]: ";
            for (int64_t i = 0; i < 8; ++i) {
            std::cout << w_ptr[i] << " ";
            }
            std::cout << std::endl;
        }
        // key偏置
        memory::dims k_bias_dims = {1, config_.hidden_size};
        key_biases_[hc_idx].desc = memory::desc(k_bias_dims, dtype_, memory::format_tag::ab);
        key_biases_[hc_idx].memory = memory(key_biases_[hc_idx].desc, engine_, 
                                           const_cast<float_t*>(k_bias_ptr) + hc_idx * config_.hidden_size);
        
        // norm权重
        memory::dims k_norm_dims = {config_.hidden_size};
        key_norm_weights_[hc_idx].desc = memory::desc(k_norm_dims, dtype_, memory::format_tag::a);
        key_norm_weights_[hc_idx].memory = memory(key_norm_weights_[hc_idx].desc, engine_, 
                                                 const_cast<float_t*>(k_norm_weight_ptr) + hc_idx * config_.hidden_size);
    }
        if (0) {
            std::cout << std::endl;
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
    // TODO: 把 for n 拿到最内层
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


#if 1
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
    // 重塑为2D: [B*T, engram_hidden_size]
    auto embeddings_2d = embeddings.view({total_elements, config_.engram_hidden_size});
    // std::cout << "embeddings_2d[0:3, 0:8]:\n" << embeddings_2d.index({torch::indexing::Slice(0, 3),torch::indexing::Slice(0, 8)}) << std::endl;
    
    /////////////////////////////////////////////////////////////////////////////
    // value = embeddings @ v_weight^T + v_bias
    //torch::Tensor value = torch::addmm(v_bias_, embeddings_2d, v_weight_.t());
    /////////////////////////////////////////////////////////////////////////////
    //auto value = torch::zeros({B * T, config_.hidden_size}, torch::TensorOptions().dtype(embeddings.dtype()));
    //linear_impl(
    //    value.data_ptr<float>(),
    //    embeddings_2d.data_ptr<float>(),
    //    v_weight_.transpose(0, 1).contiguous().data_ptr<float>(), // [input_dim, output_dim]
    //    v_bias_.data_ptr<float>(),
    //    total_elements,
    //    config_.hidden_size,
    //    config_.engram_hidden_size
    //);
    //value = value.view({B, T, 1, config_.hidden_size});
    //// std::cout << "Value projection done." << std::endl;
    //// std::cout << "value shape: " << value.sizes() << std::endl;
    //// 2. 为normed_keys预分配内存: [hc_mult, B, T, hidden_size]
    //auto normed_keys = torch::empty({config_.hc_mult, B, T, config_.hidden_size}, 
    //                                torch::TensorOptions().dtype(embeddings.dtype()));
    //
    //// 3. 并行处理每个head
    //at::parallel_for(0, config_.hc_mult, 1, [&](int64_t start_hc, int64_t end_hc) {
    //    for (int hc_idx = start_hc; hc_idx < end_hc; ++hc_idx) {
    //        // 重塑输入为2D
    //        auto embeddings_2d = embeddings.view({total_elements, config_.engram_hidden_size});
    //        
    //        // 预分配key的内存
    //        auto keys = torch::empty({total_elements, config_.hidden_size}, 
    //                                 torch::TensorOptions().dtype(embeddings.dtype()));
    //        
    //        // 并行计算key投影
    //        int64_t grain_size = 256;  // 每个线程处理的最小元素数
    //        at::parallel_for(0, total_elements, grain_size, [&](int64_t start, int64_t end) {
    //            auto k_weight_h = k_weights_[hc_idx];
    //            auto k_bias_h = k_biases_[hc_idx];
    //            
    //            auto batch_embeddings = embeddings_2d.slice(0, start, end);
    //            auto batch_keys = torch::addmm(k_bias_h, batch_embeddings, k_weight_h.t());
    //            keys.slice(0, start, end) = batch_keys;
    //        });
    //        // std::cout << "Key projection for head " << hc_idx << " done." << std::endl;
    //        
    //        // 重塑keys为3D: [B, T, hidden_size]
    //        auto keys_3d = keys.view({B, T, config_.hidden_size});
    //        
    //        // 并行计算RMSNorm
    //        auto norm_weight_h = k_norm_weights_[hc_idx];
    //        auto normed_keys_h = torch::empty({B, T, config_.hidden_size}, 
    //                                          torch::TensorOptions().dtype(embeddings.dtype()));
    //        
    //        at::parallel_for(0, B, 1, [&](int64_t start_b, int64_t end_b) {
    //            for (int64_t b = start_b; b < end_b; ++b) {
    //                auto normed_keys_b = torch::empty({T, config_.hidden_size}, 
    //                                                 torch::TensorOptions().dtype(embeddings.dtype()));
    //                
    //                for (int64_t l = 0; l < T; ++l) {
    //                    auto key_slice = keys_3d[b][l];
    //                    auto variance = key_slice.mul(key_slice).mean(-1, true);
    //                    auto inv_rms = torch::rsqrt(variance + 1e-6);
    //                    normed_keys_b[l] = key_slice * inv_rms * norm_weight_h;
    //                }
    //                
    //                normed_keys_h[b] = normed_keys_b;
    //            }
    //        });
    //        // std::cout << "RMSNorm for head " << hc_idx << " done." << std::endl;
    //        
    //        // 存储到结果中
    //        normed_keys[hc_idx] = normed_keys_h;
    //    }
    //});
    /////////////////////////////////////////////////////////////////////////////

    // 准备输出tensor
    auto value = torch::empty({B, T, 1, config_.hidden_size}, 
                              torch::TensorOptions().dtype(embeddings.dtype()).device(torch::kCPU));

    // 创建输入内存（使用行优先布局）
    memory::dims embeddings_dims = {B*T, config_.engram_hidden_size};
    memory::desc embeddings_md = memory::desc(embeddings_dims, dtype_, memory::format_tag::ab);
    memory embeddings_mem(embeddings_md, engine_, const_cast<float_t*>(embeddings.data_ptr<float_t>()));
    
    // 处理value
    {
        // 创建输出内存描述符
        memory::dims value_output_dims = {B*T, config_.hidden_size};
        memory::desc value_output_md = memory::desc(value_output_dims, dtype_, memory::format_tag::ab);
        memory value_output_mem(value_output_md, engine_, value.data_ptr<float_t>());
        
        // 创建matmul primitive_desc - 最新API
        value_matmul_pd_ = matmul::primitive_desc(
            engine_,
            embeddings_md,                    // src_desc
            value_weight_.desc,               // weights_desc
            value_bias_.desc,                 // bias_desc
            value_output_md                   // dst_desc
        );
        
        value_matmul_primitive_ = matmul(value_matmul_pd_);
        
        // 执行
        value_matmul_args_[DNNL_ARG_SRC] = embeddings_mem;
        value_matmul_args_[DNNL_ARG_WEIGHTS] = value_weight_.memory;
        value_matmul_args_[DNNL_ARG_BIAS] = value_bias_.memory;
        value_matmul_args_[DNNL_ARG_DST] = value_output_mem;
        
        value_matmul_primitive_.execute(stream_, value_matmul_args_);
    }
    

    auto normed_keys = torch::empty({config_.hc_mult, B, T, config_.hidden_size}, 
                              torch::TensorOptions().dtype(embeddings.dtype()).device(torch::kCPU));

    // 处理每个head的key
    for (int hc_idx = 0; hc_idx < config_.hc_mult; ++hc_idx) {
        // 准备key输出内存
        memory::dims key_output_dims = {B*T, config_.hidden_size};
        memory::desc key_output_md = memory::desc(key_output_dims, dtype_, memory::format_tag::ab);
        float_t* key_output_ptr = normed_keys.data_ptr<float_t>() + hc_idx * B * T * config_.hidden_size;
        memory key_output_mem(key_output_md, engine_, key_output_ptr);
        
        // 1. key matmul
        {
            // 创建matmul primitive_desc - 最新API
            key_matmul_pds_[hc_idx] = matmul::primitive_desc(
                engine_,
                embeddings_md,                    // src_desc
                key_weights_[hc_idx].desc,        // weights_desc
                key_biases_[hc_idx].desc,         // bias_desc
                key_output_md                     // dst_desc
            );
            
            key_matmul_primitives_[hc_idx] = matmul(key_matmul_pds_[hc_idx]);
            
            key_matmul_args_[hc_idx][DNNL_ARG_SRC] = embeddings_mem;
            key_matmul_args_[hc_idx][DNNL_ARG_WEIGHTS] = key_weights_[hc_idx].memory;
            key_matmul_args_[hc_idx][DNNL_ARG_BIAS] = key_biases_[hc_idx].memory;
            key_matmul_args_[hc_idx][DNNL_ARG_DST] = key_output_mem;
            
            key_matmul_primitives_[hc_idx].execute(stream_, key_matmul_args_[hc_idx]);
        }
        if (0) {
                stream_.wait();
                const auto* emb_ptr = embeddings.data_ptr<float_t>();
                const auto* w_ptr = static_cast<const float_t*>(key_weights_[hc_idx].memory.get_data_handle());
                const auto* b_ptr = static_cast<const float_t*>(key_biases_[hc_idx].memory.get_data_handle());
                const auto* out_ptr = static_cast<const float_t*>(key_output_mem.get_data_handle());

                int64_t emb_count = std::min<int64_t>(8, B * T * config_.engram_hidden_size);
                int64_t w_count = std::min<int64_t>(8, config_.hidden_size * config_.engram_hidden_size);
                int64_t b_count = std::min<int64_t>(8, config_.hidden_size);
                int64_t out_count = std::min<int64_t>(8, B * T * config_.hidden_size);

                std::cout << "\nembeddings_mem: " << "[" << hc_idx << "] - ";
                for (int64_t i = 0; i < emb_count; ++i) std::cout << emb_ptr[i] << " ";
                std::cout << "\nkey_weights_: ";
                for (int64_t i = 0; i < w_count; ++i) std::cout << w_ptr[i] << " ";
                std::cout << "\nkey_biases_: ";
                for (int64_t i = 0; i < b_count; ++i) std::cout << b_ptr[i] << " ";
                std::cout << "\nkey_output_mem: ";
                for (int64_t i = 0; i < out_count; ++i) std::cout << out_ptr[i] << " ";
                std::cout << std::endl;
        }
        // 2. norm (使用LayerNorm近似RMSNorm)
        {
            float epsilon = 1e-6f;
            
            // 创建layer normalization primitive_desc - 最新API
            norm_pds_[hc_idx] = layer_normalization_forward::primitive_desc(
                engine_,
                prop_kind::forward_inference,     // prop_kind
                key_output_md,                    // data_desc
                key_output_md,                    // dst_desc (原地操作)
                epsilon,                          // epsilon
                normalization_flags::use_scale | normalization_flags::rms_norm
            );
            
            norm_primitives_[hc_idx] = layer_normalization_forward(norm_pds_[hc_idx]);
            
            // 创建统计量内存（虽然不使用，但API需要）
            memory::dims stats_dims = {B, T};
            memory::desc stats_md = memory::desc(stats_dims, memory::data_type::f32, memory::format_tag::ab);
            memory mean_mem(stats_md, engine_);
            memory variance_mem(stats_md, engine_);
            
            norm_args_[hc_idx][DNNL_ARG_SRC] = key_output_mem;
            norm_args_[hc_idx][DNNL_ARG_SCALE] = key_norm_weights_[hc_idx].memory;
            norm_args_[hc_idx][DNNL_ARG_MEAN] = mean_mem;
            norm_args_[hc_idx][DNNL_ARG_VARIANCE] = variance_mem;
            norm_args_[hc_idx][DNNL_ARG_DST] = key_output_mem;  // 原地操作
            
            norm_primitives_[hc_idx].execute(stream_, norm_args_[hc_idx]);
        }
    }
    stream_.wait();
    
    // std::cout << "value shape: " << value.sizes() << std::endl;
    // std::cout << "normed_keys shape: " << normed_keys.sizes() << std::endl;
    return {value, normed_keys};
}
// 显式实例化
//template class EngramCPU<float>;
//template class EngramCPU<uint16_t>;

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