#include "engram_cpu.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <immintrin.h>


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

    embedding_weights_ = embedding_weights_.to(torch::kCPU).contiguous();
    
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

    total_heads_ = 0;
    for (int64_t n = 2; n <= config_.max_ngram_size; ++n) {
        total_heads_ += config_.n_head_per_ngram;
    }
    embed_dim_per_head_ = config_.n_embed_per_ngram / config_.n_head_per_ngram;
}

//template <typename Type>
void EngramCPU::set_weights(const torch::Tensor& k_weight,
                            const torch::Tensor& k_bias,
                            const torch::Tensor& k_norm_weight,
                            const torch::Tensor& v_weight,
                            const torch::Tensor& v_bias) {
    k_weights_ = k_weight.to(torch::kCPU).contiguous();
    k_biases_ = k_bias.to(torch::kCPU).contiguous();
    k_norm_weights_ = k_norm_weight.to(torch::kCPU).contiguous();
    v_weight_ = v_weight.to(torch::kCPU).contiguous();
    v_bias_ = v_bias.to(torch::kCPU).contiguous();

    // 获取Tensor数据指针
    auto k_weight_ptr = k_weights_.data_ptr<float_t>();
    auto k_bias_ptr = k_biases_.data_ptr<float_t>();
    auto k_norm_weight_ptr = k_norm_weights_.data_ptr<float_t>();
    auto v_weight_ptr = v_weight_.data_ptr<float_t>();
    auto v_bias_ptr = v_bias_.data_ptr<float_t>();
    
    // 获取维度信息
    auto k_weight_sizes = k_weight.sizes();
    int engram_hidden_size = k_weight_sizes[1];
    int hidden_size = k_weight_sizes[0];

    std::cout << "Setting weights for layer " << layer_id_ << " with hidden_size=" << hidden_size 
              << ", engram_hidden_size=" << engram_hidden_size << std::endl;
        
    // 设置value权重 - 逻辑形状为 [engram_hidden_size, hidden_size]
    // 使用 ba 以便直接视作转置后的权重
    memory::dims v_weight_dims = {config_.engram_hidden_size, config_.hidden_size};
    value_weight_.desc = memory::desc(v_weight_dims, dtype_, memory::format_tag::ba);
    value_weight_.memory = memory(value_weight_.desc, engine_, const_cast<float_t*>(v_weight_ptr));
    
    // 设置value偏置 - 1xN 以匹配 matmul 的 bias 语义
    memory::dims v_bias_dims = {1, config_.hidden_size};
    value_bias_.desc = memory::desc(v_bias_dims, dtype_, memory::format_tag::ab);
    value_bias_.memory = memory(value_bias_.desc, engine_, const_cast<float_t*>(v_bias_ptr));
    
    // 为每个head设置key权重
    for (int hc_idx = 0; hc_idx < config_.hc_mult; ++hc_idx) {
        // key权重 - 逻辑形状为 [engram_hidden_size, hidden_size]
        memory::dims k_weight_dims = {config_.engram_hidden_size, config_.hidden_size};
        key_weights_[hc_idx].desc = memory::desc(k_weight_dims, dtype_, memory::format_tag::ba);
        key_weights_[hc_idx].memory = memory(key_weights_[hc_idx].desc, engine_, 
                                            const_cast<float_t*>(k_weight_ptr) + hc_idx * config_.hidden_size * config_.engram_hidden_size);
        // key偏置 - 1xN 以匹配 matmul 的 bias 语义
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

    primitive_cache_.clear();
}

EngramCPU::PrimitiveCache& EngramCPU::get_or_create_cache(int64_t m) {
    auto it = primitive_cache_.find(m);
    if (it != primitive_cache_.end()) {
        return it->second;
    }

    PrimitiveCache cache;
    cache.m = m;
    cache.src_md = memory::desc({m, config_.engram_hidden_size}, dtype_, memory::format_tag::ab);
    cache.dst_md = memory::desc({m, config_.hidden_size}, dtype_, memory::format_tag::ab);
    cache.bias_md = memory::desc({1, config_.hidden_size}, dtype_, memory::format_tag::ab);

    cache.value_pd = matmul::primitive_desc(
        engine_,
        cache.src_md,
        value_weight_.desc,
        cache.bias_md,
        cache.dst_md
    );
    cache.value_prim = matmul(cache.value_pd);

    cache.key_pds.resize(config_.hc_mult);
    cache.key_prims.resize(config_.hc_mult);
    cache.norm_pds.resize(config_.hc_mult);
    cache.norm_prims.resize(config_.hc_mult);
    cache.mean_mems.resize(config_.hc_mult);
    cache.var_mems.resize(config_.hc_mult);

    const float epsilon = 1e-6f;
    memory::desc stats_md = memory::desc({m}, memory::data_type::f32, memory::format_tag::a);

    for (int hc_idx = 0; hc_idx < config_.hc_mult; ++hc_idx) {
        cache.key_pds[hc_idx] = matmul::primitive_desc(
            engine_,
            cache.src_md,
            key_weights_[hc_idx].desc,
            cache.bias_md,
            cache.dst_md
        );
        cache.key_prims[hc_idx] = matmul(cache.key_pds[hc_idx]);

        cache.norm_pds[hc_idx] = layer_normalization_forward::primitive_desc(
            engine_,
            prop_kind::forward_inference,
            cache.dst_md,
            cache.dst_md,
            epsilon,
            normalization_flags::use_scale | normalization_flags::rms_norm
        );
        cache.norm_prims[hc_idx] = layer_normalization_forward(cache.norm_pds[hc_idx]);

        cache.mean_mems[hc_idx] = memory(stats_md, engine_);
        cache.var_mems[hc_idx] = memory(stats_md, engine_);
    }

    auto inserted = primitive_cache_.emplace(m, std::move(cache));
    return inserted.first->second;
}

void EngramCPU::compressed_tokenizer(const int64_t* input_ids, int64_t B, int64_t T, std::vector<int64_t>& out) {
    out.resize(B * T);
    int64_t vocab_size = static_cast<int64_t>(lookup_table_.size());
    
    #pragma omp parallel for collapse(2) if (B * T > 1000)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            int64_t idx = b * T + t;
            int64_t token_id = input_ids[idx];
            if (token_id >= 0 && token_id < vocab_size) {
                out[idx] = lookup_table_[token_id];
            } else {
                out[idx] = token_id; // Keep as is if out of bounds
            }
        }
    }
}

void EngramCPU::get_ngram_hashes(
    const std::vector<int64_t>& compressed_ids,
    int64_t B,
    int64_t T,
    std::vector<int64_t>& out
) {
    out.resize(total_heads_ * B * T);
    
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
                int64_t mix = compressed_ids[idx] * multipliers_[0];
                for (int64_t k = 1; k < n; ++k) {
                    int64_t shifted = (t < k) ? config_.pad_id : compressed_ids[b * T + (t - k)];
                    mix = mix ^ (shifted * multipliers_[k]);
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
                    out[(head_idx + j) * (B * T) + idx] = head_hash;
                }
            }
        }
        head_idx += config_.n_head_per_ngram;
    }
}

void EngramCPU::multi_head_embedding(const std::vector<int64_t>& hash_ids,
    int64_t B,
    int64_t T,
    torch::Tensor& output) {
    int64_t D = embed_dim_per_head_;
    int64_t num_heads = total_heads_;

    auto embedding_ptr = embedding_weights_.data_ptr<float>();
    auto output_ptr = output.data_ptr<float>();
    int64_t embedding_stride = embedding_weights_.size(1);
    int64_t embedding_rows = embedding_weights_.size(0);
    int64_t out_stride_bt = num_heads * D;

    #pragma omp parallel for collapse(2) if (B * T > 1000)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            int64_t base_out = (b * T + t) * out_stride_bt;
            int64_t base_idx = b * T + t;
            for (int64_t h = 0; h < num_heads; ++h) {
                int64_t offset_hash = hash_ids[h * (B * T) + base_idx] + offsets_[h];
                const float* src = embedding_ptr + offset_hash * embedding_stride;
                float* dst = output_ptr + base_out + (h * D);
                std::memcpy(dst, src, D * sizeof(float));
            }
        }
    }
}

std::pair<torch::Tensor, torch::Tensor> EngramCPU::forward(const torch::Tensor& input_ids) {
    // Ensure input is on CPU
    auto input_cpu = input_ids.to(torch::kCPU).contiguous();
    
    // Get dimensions
    auto sizes = input_cpu.sizes();
    int64_t B = sizes[0];
    int64_t T = sizes[1];
    
    const int64_t* input_ptr = input_cpu.data_ptr<int64_t>();
    
    // Step 1: Compressed tokenizer
    compressed_tokenizer(input_ptr, B, T, compressed_ids_buffer_);
    // std::cout << "Compressed tokenizer." << std::endl;
    
    // Step 2: Get n-gram hashes
    get_ngram_hashes(compressed_ids_buffer_, B, T, hash_ids_buffer_);
    // std::cout << "Get n-gram hashes." << std::endl;
    
    // Step 3: Multi-head embedding
    if (!embeddings_buffer_.defined() || embeddings_buffer_.sizes() != std::vector<int64_t>({B, T, total_heads_ * embed_dim_per_head_})) {
        embeddings_buffer_ = torch::empty({B, T, total_heads_ * embed_dim_per_head_},
                                          torch::TensorOptions().dtype(embedding_weights_.dtype()).device(torch::kCPU));
    }
    multi_head_embedding(hash_ids_buffer_, B, T, embeddings_buffer_);
    auto embeddings = embeddings_buffer_;
    //auto emb_slice = embeddings.index({torch::indexing::Slice(0, 3), 0, torch::indexing::Slice(0, 8)});
    // std::cout << "embeddings[0:3, 0, 0:8]:\n" << emb_slice << std::endl;
    
    auto total_elements = B * T;
    auto& cache = get_or_create_cache(total_elements);
    // 1. 计算value投影

    // 准备输出tensor
    auto value = torch::empty({B, T, 1, config_.hidden_size}, 
                              torch::TensorOptions().dtype(embeddings.dtype()).device(torch::kCPU));

    // 创建输入内存（使用行优先布局）
    memory embeddings_mem(cache.src_md, engine_, const_cast<float_t*>(embeddings.data_ptr<float_t>()));
    
    // 处理value
    {
        // 创建输出内存描述符
        memory value_output_mem(cache.dst_md, engine_, value.data_ptr<float_t>());
        
        // 执行
        value_matmul_args_[DNNL_ARG_SRC] = embeddings_mem;
        value_matmul_args_[DNNL_ARG_WEIGHTS] = value_weight_.memory;
        value_matmul_args_[DNNL_ARG_BIAS] = value_bias_.memory;
        value_matmul_args_[DNNL_ARG_DST] = value_output_mem;
        
        cache.value_prim.execute(stream_, value_matmul_args_);
    }
    

    auto normed_keys = torch::empty({config_.hc_mult, B, T, config_.hidden_size}, 
                              torch::TensorOptions().dtype(embeddings.dtype()).device(torch::kCPU));

    // 处理每个head的key
    for (int hc_idx = 0; hc_idx < config_.hc_mult; ++hc_idx) {
        // 准备key输出内存
        float_t* key_output_ptr = normed_keys.data_ptr<float_t>() + hc_idx * B * T * config_.hidden_size;
        memory key_output_mem(cache.dst_md, engine_, key_output_ptr);
        
        // 1. key matmul
        {
            key_matmul_args_[hc_idx][DNNL_ARG_SRC] = embeddings_mem;
            key_matmul_args_[hc_idx][DNNL_ARG_WEIGHTS] = key_weights_[hc_idx].memory;
            key_matmul_args_[hc_idx][DNNL_ARG_BIAS] = key_biases_[hc_idx].memory;
            key_matmul_args_[hc_idx][DNNL_ARG_DST] = key_output_mem;
            
            cache.key_prims[hc_idx].execute(stream_, key_matmul_args_[hc_idx]);
        }
        // 2. norm (使用LayerNorm近似RMSNorm)
        {
            norm_args_[hc_idx][DNNL_ARG_SRC] = key_output_mem;
            norm_args_[hc_idx][DNNL_ARG_SCALE] = key_norm_weights_[hc_idx].memory;
            norm_args_[hc_idx][DNNL_ARG_MEAN] = cache.mean_mems[hc_idx];
            norm_args_[hc_idx][DNNL_ARG_VARIANCE] = cache.var_mems[hc_idx];
            norm_args_[hc_idx][DNNL_ARG_DST] = key_output_mem;  // 原地操作
            
            cache.norm_prims[hc_idx].execute(stream_, norm_args_[hc_idx]);
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
