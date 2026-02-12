"""
================================================================================
[Engram Architecture Demo Implementation]

DISCLAIMER:
1. Demo Purpose Only: 
   This code is a demonstration version intended to illustrate the core logic and 
   data flow of the Engram module.

2. Production Readiness: 
   This implementation requires further optimization for actual production use 
   (e.g., custom CUDA kernels, distributed training support).

3. Simplifications: 
   Standard components (Normalization, Attention, MoE) and complex Hyper-connection 
   mechanisms are omitted or mocked in this version to focus exclusively on the 
   Engram module implementation.
================================================================================
"""

"""
pip install torch numpy transformers sympy
"""

# python engram_demo_v1_profile.py --batch-size 1 --seq-len 1024 --n-embed-per-ngram 2560 --hidden-size 4096 --dtype bf16

import os
import time
import glob
import argparse
os.environ['PT_HPU_LAZY_MODE'] = '1'
os.environ['HABANA_PROFILE'] = '1'

from contextlib import contextmanager
from typing import Dict

## built-in
from typing import List, Tuple, Any

from dataclasses import dataclass, field
import math
import threading

## third-party
from sympy import isprime
import numpy as np
import torch
import torch.nn as nn
import habana_frameworks.torch.hpu as ht
import habana_frameworks.torch as htorch
import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex

cpu_device = torch.device("cpu")
hpu_device = torch.device("hpu")
torch.set_printoptions(linewidth=800)

random_seed = 102
torch.manual_seed(random_seed)
np.random.seed(random_seed)

class StepTimer:
    def __init__(self):
        self.stats: Dict[str, Dict[str, float]] = {}
        self.total = 1

    @contextmanager
    def measure(self, name: str):
        start_ns = time.perf_counter_ns()
        yield
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6

        if name not in self.stats:
            self.stats[name] = {'time_ms': 0.0, 'count': 0}
        
        self.stats[name]['time_ms'] += elapsed_ms
        # if name == "embeddings.to":
        #     print(f"embeddings.to elapsed_ms: {elapsed_ms}  ({self.stats[name]['count']})")
        self.stats[name]['count'] += 1

    def __str__(self) -> str:
        lines = []
        for name, stats in self.stats.items():
            avg_time = stats['time_ms'] / stats['count'] if stats['count'] > 0 else 0
            lines.append(f"{name}: 总时间={stats['time_ms']:.2f}ms, "
                        f"\t次数={stats['count']}, "
                        f"\t平均={avg_time:.2f}ms"
                        f"\t占比={avg_time*100/self.total:5.2f}%")
        return "\n".join(lines)

def human_format(num):
    magnitude = 0
    #while abs(num) >= 1000:
    while magnitude < 3:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

def get_latest_trace_file(log_dir="./logs"):
    trace_files = glob.glob(os.path.join(log_dir, "*.pt.trace.json.gz"))
    if not trace_files:
        return None
    return max(trace_files, key=os.path.getctime)


''' # Original Config
@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
    engram_vocab_size: List[int] = field(default_factory=lambda: [129280*5, 129280*5])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4
'''

# 1-layer Engram for profile
@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = "/mnt/disk8/hf_models/DeepSeek-V3"
    max_ngram_size: int = 3
    n_head_per_ngram: int = 8
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4


    # original config 0.662B
    #layer_ids: List[int] = field(default_factory=lambda: [1])
    #layer_ids: List[int] = field(default_factory=lambda: [1,15])
    layer_ids: List[int] = field(default_factory=lambda: [1,3])
    engram_vocab_size: List[int] = field(default_factory=lambda: [129280*5, 129280*5])
    n_embed_per_ngram: int = 512

    '''
    # config 0.662B * 4 layers
    layer_ids: List[int] = field(default_factory=lambda: [0,1,2,3])
    engram_vocab_size: List[int] = field(default_factory=lambda: [129280*5, 129280*5])
    n_embed_per_ngram: int = 512
    '''
    '''
    # scaled-up config 100.8B
    layer_ids: List[int] = field(default_factory=lambda: [1])
    engram_vocab_size: List[int] = field(default_factory=lambda: [123000*50, 123000*50])
    n_embed_per_ngram: int = 8192
    '''

    
@dataclass
class BackBoneConfig:
    hidden_size: int = 1024
    hc_mult: int = 4
    vocab_size: int = 129280
    num_layers: int = 5 # 30
    
engram_cfg = EngramConfig()
backbone_config = BackBoneConfig()

            
class ShortConv(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        kernel_size: int = 4, 
        dilation: int = 1, 
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation
        
        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        ).to(hpu_device)

        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=norm_eps).to(hpu_device)
            for _ in range(hc_mult)
        ])
        
        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B,L,HC_MULT,D)
        Output: (B,L,HC_MULT,D)
        """
        B, T, G, C = x.shape
        
        assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))
        
        x_norm = torch.cat(normed_chunks, dim=-1)
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]

        if self.activation:
            y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()
        
        return y
    
def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1

    
class Engram_host(nn.Module):
    def __init__(self, layer_id,timer):
        super().__init__()
        
        torch.manual_seed(random_seed+layer_id)
        np.random.seed(random_seed+layer_id)
        self.timer = timer
        self.layer_id = layer_id

        # NgramHashMapping
        self.vocab_size_per_ngram = engram_cfg.engram_vocab_size
        self.max_ngram_size = engram_cfg.max_ngram_size
        self.n_embed_per_ngram = engram_cfg.n_embed_per_ngram
        self.n_head_per_ngram = engram_cfg.n_head_per_ngram
        self.pad_id = engram_cfg.pad_id
        self.layer_ids = engram_cfg.layer_ids

        # CompressedTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(engram_cfg.tokenizer_name_or_path, trust_remote_code=True)
        
        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])
        
        self.lookup_table, self.tokenizer_vocab_size = self._build_lookup_table()

        if self.pad_id is not None:
            self.pad_id = int(self.lookup_table[self.pad_id])

        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007
        
        base_seed = int(engram_cfg.seed + PRIME_1 * int(layer_id))
        g = np.random.default_rng(base_seed)
        r = g.integers(
            low=0,
            high=half_bound,
            size=(self.max_ngram_size,),
            dtype=np.int64
        )
        self.layer_multipliers = r * 2 + 1


        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

        # MultiHeadEmbedding
        list_of_N = [x for y in self.vocab_size_across_layers[self.layer_id] for x in y]
        D = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        
        total_N = sum(list_of_N)
        self.mhe_embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

        engram_hidden_size = (engram_cfg.max_ngram_size-1) * engram_cfg.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size,backbone_config.hidden_size).to(cpu_device)
        self.key_projs = nn.ModuleList(
            [nn.Linear(engram_hidden_size, backbone_config.hidden_size).to(cpu_device) for _ in range(backbone_config.hc_mult)]
        )
        self.norm1 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size).to(cpu_device) for _ in range(backbone_config.hc_mult)])

        self._init_cpp_engram()

        print(f"[Engram]: multi_head_embedding[{layer_id}]: num_embeddings={self.mhe_embedding.num_embeddings}, embedding_dim={self.mhe_embedding.embedding_dim}, params={human_format(self.mhe_embedding.num_embeddings*self.mhe_embedding.embedding_dim)}")
        
    def _init_cpp_engram(self):
        """初始化C++版本的Engram"""
        import engram_cpu
        
        # 准备配置
        config = engram_cpu.EngramConfig()
        config.engram_vocab_size = self.vocab_size_per_ngram
        config.max_ngram_size = self.max_ngram_size
        config.n_embed_per_ngram = self.n_embed_per_ngram
        config.n_head_per_ngram = self.n_head_per_ngram
        config.layer_ids = self.layer_ids
        config.pad_id = self.pad_id
        config.seed = engram_cfg.seed
        config.kernel_size = engram_cfg.kernel_size
        config.hidden_size = backbone_config.hidden_size
        config.engram_hidden_size = (engram_cfg.max_ngram_size-1) * engram_cfg.n_embed_per_ngram
        config.hc_mult = backbone_config.hc_mult
        config.tokenizer_vocab_size = self.tokenizer_vocab_size
        
        multipliers_list = self.layer_multipliers.tolist()
        
        # 获取当前层的vocab sizes
        if self.layer_id not in self.vocab_size_across_layers:
            raise ValueError(f"Layer ID {self.layer_id} not found in vocab_size_across_layers")
        vocab_sizes_for_layer = self.vocab_size_across_layers[self.layer_id]
        # 转换为list of list of int
        vocab_sizes_list = []
        for ngram_list in vocab_sizes_for_layer:
            vocab_sizes_list.append([int(x) for x in ngram_list])
        
        # 准备offsets
        offsets_np = self.offsets.cpu().numpy().tolist()
        
        # 获取embedding权重
        embedding_weights = self.mhe_embedding.weight.detach().cpu()
        
        # 获取projection层权重
        value_proj_weight = self.value_proj.weight.detach().cpu()
        value_proj_bias = self.value_proj.bias.detach().cpu()
        
        k_weights = []
        k_biases = []
        for proj in self.key_projs:
            k_weights.append(proj.weight.detach().cpu())
            k_biases.append(proj.bias.detach().cpu())
        key_proj_weights = torch.stack(k_weights)
        key_proj_biases = torch.stack(k_biases)
        
        k_norm_weights = []
        for norm in self.norm1:
            k_norm_weights.append(norm.weight.detach().cpu())
        key_norm_weight = torch.stack(k_norm_weights)
        
        # 保持张量生命周期
        self._value_proj_weight = value_proj_weight.contiguous()
        self._value_proj_bias = value_proj_bias.contiguous()
        self._key_proj_weights = key_proj_weights.contiguous()
        self._key_proj_biases = key_proj_biases.contiguous()
        self._key_norm_weight = key_norm_weight.contiguous()

        # 创建C++ Engram对象
        self.cpp_engram = engram_cpu.EngramCPU(
            layer_id=self.layer_id,
            config=config,
            lookup_table=self.lookup_table.tolist(),
            multipliers=multipliers_list,
            vocab_sizes_for_layer=vocab_sizes_list,
            offsets=offsets_np,
            embedding_weights=embedding_weights,
        )
        self.cpp_engram.set_weights(
            self._key_proj_weights,
            self._key_proj_biases,
            self._key_norm_weight,
            self._value_proj_weight,
            self._value_proj_bias
        )

    def _build_lookup_table(self):
        old2new = {}
        key2new = {}          
        new_tokens = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            
            if "�" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid
        
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)

    def calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}
        
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1
                
                for _ in range(num_head):
                    found_prime = find_next_prime(
                        current_prime_search_start, 
                        seen_primes
                    )
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime
                
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
            
        return vocab_size_across_layers

    def compressed_tokenizer(self, input_ids):
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        out[pos_mask] = self.lookup_table[valid_ids]
        return out   

    def _get_ngram_hashes(
        self,
        input_ids: np.ndarray,
    ) -> np.ndarray:
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        multipliers = self.layer_multipliers

        def shift_k(k: int) -> np.ndarray:
            if k == 0: return x
            shifted = np.pad(x, ((0, 0), (k, 0)),
                                mode='constant', constant_values=self.pad_id)[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []

        # original version, straightforward but redundant computation for 3-gram
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = (tokens[0] * multipliers[0])
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[self.layer_id][n_gram_index]

            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))
        '''
        # 版本1, 3-gram 复用 2-gram 计算重复部分
        # tokens[k] * multipliers[k]
        # bitwise_xor
        # 但缓存性能差
        max_n = self.max_ngram_size
        all_weighted_tokens = [base_shifts[i] * multipliers[i] for i in range(max_n)]
        mix_k = []
        mix = all_weighted_tokens[0]
        for k in range(1, max_n):
            mix = np.bitwise_xor(mix, all_weighted_tokens[k])
            mix_k.append(mix)
        for n in range(2, max_n + 1):
            head_vocab_sizes = self.vocab_size_across_layers[self.layer_id][n - 2]
            for j in range(self.n_head_per_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix_k[n-2] % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))
        '''

        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_one_layer = self._get_ngram_hashes(input_ids)
        return hash_ids_for_one_layer

    def multi_head_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        shifted_input_ids = input_ids + self.offsets
        output = self.mhe_embedding(shifted_input_ids)
        
        return output

    def hash_and_embed(self, input_ids):
        hash_input_ids = torch.from_numpy(self.hash(input_ids)).to(cpu_device)
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
        return embeddings

    def forward(self,input_ids):
        value, normed_keys = self.cpp_engram.forward(input_ids)
        #print(f"Engram_host forward output - value: {value}\nnormed_keys:  {normed_keys}")
        return value, normed_keys

        """
        hidden_states: [B, L, HC_MULT, D]
        input_ids: [B, L]
        """
        with self.timer.measure("engram_host  "):
            embeddings = self.hash_and_embed(input_ids)
            #embeddings = self.cpp_engram.forward(input_ids)
            #embeddings=embeddings.to(cpu_device)
        
            value = self.value_proj(embeddings).unsqueeze(2)
            #print(f"Engram_host forward - embeddings: {embeddings}")
            normed_keys = []
            for hc_idx in range(backbone_config.hc_mult):
                key = self.key_projs[hc_idx](embeddings)
                #print(f"Engram_host forward {hc_idx}- embeddings: {embeddings}")
                #print(f"Engram_host forward {hc_idx}- weight: {self.key_projs[hc_idx].weight}")
                #print(f"Engram_host forward {hc_idx}- bias: {self.key_projs[hc_idx].bias}")
                #print(f"Engram_host forward {hc_idx}- key: {key}")
                normed_key = self.norm1[hc_idx](key)
                normed_keys.append(normed_key)
            normed_keys = torch.stack(normed_keys)
        #print(f"Engram_host forward - value: {value}")
        #print(f"Engram_host forward output - value: {value}\nnormed_keys:  {normed_keys}")
        return value, normed_keys


@wrap_in_hpu_graph
class Engram_device(nn.Module):
    def __init__(self, layer_id,timer):
        super().__init__()
        torch.manual_seed(random_seed+layer_id)
        np.random.seed(random_seed+layer_id)
        self.timer = timer
        self.layer_id = layer_id

        self.short_conv = ShortConv(
            hidden_size = backbone_config.hidden_size,
            kernel_size = engram_cfg.kernel_size,
            dilation    = engram_cfg.max_ngram_size,
            hc_mult     = backbone_config.hc_mult,
        )
        self.norm2 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size).to(hpu_device) for _ in range(backbone_config.hc_mult)])
    
    def forward(self,hidden_states, value, normed_keys):
        gates = []
        for hc_idx in range(backbone_config.hc_mult):
            query = hidden_states[:,:,hc_idx,:]
            normed_query = self.norm2[hc_idx](query)
            gate = (normed_keys[hc_idx] * normed_query).sum(dim=-1) / math.sqrt(backbone_config.hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)
        gates = torch.stack(gates,dim=2)
        value = gates * value
        output = value + self.short_conv(value)
        htcore.mark_step()
        return output


class Embedding(nn.Module):
    def __init__(self,vocab_size,hidden_size):
        super().__init__()
        self.Embedding = nn.Embedding(vocab_size,hidden_size).to(hpu_device)
        # self.Fake_proj = nn.Linear(hidden_size,hidden_size).to(hpu_device)

    def forward(self,input_ids):
        hidden_states = self.Embedding(input_ids)
        # hidden_states = self.Fake_proj(hidden_states)
        return hidden_states

@wrap_in_hpu_graph
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = lambda x:x*0.3
        self.moe  = lambda x:x*0.3
        #self.Fake_proj = nn.Linear(backbone_config.hidden_size,backbone_config.hidden_size).to(hpu_device)
        #with torch.no_grad():
        #    self.Fake_proj.weight.copy_(torch.eye(backbone_config.hidden_size, backbone_config.hidden_size))
        #    self.Fake_proj.bias.copy_(torch.zeros_like(self.Fake_proj.bias))
    
    def forward(self, hidden_states):
        hidden_states = self.attn(hidden_states) + hidden_states
        #for _ in range(10):
        #    hidden_states = self.Fake_proj(hidden_states)
        hidden_states = self.moe(hidden_states) + hidden_states
        htcore.mark_step()
        return hidden_states

class GetEngram(nn.Module):
    def __init__(self,layer_id,timer,engram_manager):
        super().__init__()
        self.timer = timer
        self.layer_id = layer_id
        self.engram_manager = engram_manager
        self.engram_device = Engram_device(layer_id=layer_id,timer=timer)
    
    def forward(self, hidden_states):
        with self.timer.measure(f"layer_{self.layer_id}_join  "):
            value, normed_keys = self.engram_manager.get_engram_output(self.layer_id)
        with self.timer.measure(f"layer_{self.layer_id}_device"):
            hidden_states = self.engram_device(hidden_states, value, normed_keys) + hidden_states
        return hidden_states
    
class TransformerBlockWithEngram(nn.Module):
    def __init__(self,layer_id,timer,engram_manager):
        super().__init__()
        self.engram_device = Engram_device(layer_id=layer_id,timer=timer)
        self.get_engram = GetEngram(layer_id=layer_id,timer=timer,engram_manager=engram_manager)
        self.transformer_block = TransformerBlock()
    
    def forward(self, hidden_states):
        hidden_states = self.get_engram(hidden_states)
        hidden_states = self.transformer_block(hidden_states)
        return hidden_states

class EngramManager:
    def __init__(self, layer_ids, timer):
        self.layer_ids = layer_ids
        self.timer = timer
        self.engram_host_layer = nn.ModuleList(
            [Engram_host(layer_id=layer_id,timer=self.timer) for layer_id in self.layer_ids]
        )
        
        self.threads: Dict[int, threading.Thread] = {}
        self.results: Dict[int, Tuple[Any, Any]] = {layer_id: (None, None) for layer_id in self.layer_ids}

    def _compute_engram_host(self, layer_id, input_ids):
        layer = self.engram_host_layer[self.layer_ids.index(layer_id)]
        #print(f"Worker thread - PyTorch threads: {torch.get_num_threads()}")
        #print(f"Worker thread - CPU count: {os.cpu_count()}")
        #torch.set_num_threads(40)
        
        with self.timer.measure(f"layer_{layer_id}_cpu_compute"):
            value, normed_keys = layer(input_ids=input_ids)

        with self.timer.measure(f"layer_{layer_id}_to_hpu     "):
            value_hpu  = value.to("hpu")
            normed_keys_hpu  = normed_keys.to("hpu")

        self.results[layer_id] = (value_hpu , normed_keys_hpu)

    def start_async_computation(self, input_ids):
        # Synchronous version for reference
        input_ids = input_ids.to(cpu_device)
        for layer_id in self.layer_ids:
            self._compute_engram_host(layer_id, input_ids)
        '''
        self.threads.clear()
        input_ids = input_ids.to(cpu_device)
        for layer_id in self.layer_ids:
            thread = threading.Thread(
                target=self._compute_engram_host,
                args=(layer_id, input_ids),
                daemon=True
            )
            self.threads[layer_id] = thread
            thread.start()
        '''

    def get_engram_output(self, layer_id):
        '''
        # Commented out for synchronous version
        assert layer_id in self.layer_ids, f"Layer ID {layer_id} not in Engram layers {self.layer_ids}"
        assert layer_id in self.threads, f"Layer ID {layer_id} not in Engram threads {self.layer_ids}"
        if self.threads[layer_id].is_alive():
            self.threads[layer_id].join(timeout=10.0)
        # End of commented out section
        '''
        value, normed_keys = self.results[layer_id]
        return value, normed_keys

class LLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.timer = StepTimer()
        self.vocab_embed_tokens = Embedding(backbone_config.vocab_size,backbone_config.hidden_size)
        self.engram_manager = EngramManager(engram_cfg.layer_ids,timer=self.timer)
        self.decoder_layers = nn.ModuleList(
            [
                TransformerBlockWithEngram(layer_id=layer_id, timer=self.timer, engram_manager=self.engram_manager)
                if layer_id in engram_cfg.layer_ids
                else TransformerBlock()
                for layer_id in range(backbone_config.num_layers)
            ]
        )
        #self.lm_head = nn.Linear(backbone_config.hidden_size, backbone_config.vocab_size).to(hpu_device)

    def forward(self, input_ids, profile_engram=False):
        self.engram_manager.start_async_computation(input_ids)

        hidden_states = self.vocab_embed_tokens(input_ids)
        ## mock hyper-connection
        hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, backbone_config.hc_mult, -1)
        for layer in self.decoder_layers:
            hidden_states = layer(hidden_states=hidden_states)
        ## mock hyper-connection
        output = hidden_states[:,:,0,:]
        #output = self.lm_head(output)
        return output, self.timer

# python engram_demo_v1_profile.py --batch-size 1 --seq-len 1024 --n-embed-per-ngram 2560 --hidden-size 4096 --dtype bf16
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size.')
    parser.add_argument('--seq-len', type=int, default=1, help='Sequence length.')
    parser.add_argument('--n-embed-per-ngram', type=int, default=512, help='Embedding dims per ngram.')
    parser.add_argument('--hidden-size', type=int, default=1024, help='Embedding dims for Transformer (Hidden size).')
    parser.add_argument('--dtype', type=str, default="fp32", help='data type.')
    args = parser.parse_args()

    backbone_config.hidden_size = int(args.hidden_size)
    engram_cfg.n_embed_per_ngram = int(args.n_embed_per_ngram)
    if args.dtype == "fp16":
        torch.set_default_dtype(torch.float16)
    elif args.dtype == "bf16":
        torch.set_default_dtype(torch.bfloat16)
    else:
        torch.set_default_dtype(torch.float32)

    '''
    text = "Only Alexander the Great could tame the horse Bucephalus."
    tokenizer = AutoTokenizer.from_pretrained(engram_cfg.tokenizer_name_or_path,trust_remote_code=True)
    input_ids = tokenizer(text,return_tensors='pt').input_ids
    '''
    #input_b, input_seq = 16, 1
    #input_b, input_seq = 1, 8*1024
    #input_b, input_seq = 1, 1
    input_b, input_seq = int(args.batch_size), int(args.seq_len)
    input_ids = torch.randint(0, 127000, size=(input_b, input_seq), dtype=torch.long)
    input_ids = input_ids.to(hpu_device)

    B,L = input_ids.shape
    print(f"\n******** Engram Configuration ********")
    print(f"[Engram]: n_embed_per_ngram = {engram_cfg.n_embed_per_ngram}")
    print(f"[Engram]: hidden_size       = {backbone_config.hidden_size}")
    print(f"[Engram]: input_ids: {input_ids.shape} {input_ids.device}")
    print(f"[Engram]: embeddings: [B, T, (N-1)*n_embed_per_ngram] = [{B}, {L}, {engram_cfg.max_ngram_size-1}*{engram_cfg.n_embed_per_ngram}] = [{B}, {L}, {(engram_cfg.max_ngram_size-1)*engram_cfg.n_embed_per_ngram}]")
    print(f"[Engram]: key_projs : [hc][B, T, H] = [{backbone_config.hc_mult}][{B}, {L}, {backbone_config.hidden_size}]")
    print(f"[Engram]: value_proj:     [B, T, H] =    [{B}, {L}, {backbone_config.hidden_size}]")
    print(f"[Engram]: default dtype: {torch.get_default_dtype()}")

    llm = LLM()
    llm.eval()

    warmup_iters = 50
    with torch.no_grad():
        for i in range(int(warmup_iters)):  ## warm-up
            input_ids_local = input_ids + i
            output, _ =  llm(input_ids_local)
            htcore.mark_step()
    print(f"**************************************\n")

    loop_iters = 1000
    output=None
    start_time = time.perf_counter()
    with torch.no_grad():
        for i in range(loop_iters):
            input_ids_local = input_ids + i
            output_new, _ =  llm(input_ids_local)
            output = output_new + output if output is not None else output_new
            htcore.mark_step()
    end_time = time.perf_counter()
    execution_time = (end_time - start_time) * 1000 / loop_iters
    print(output)
    print(f"Inference {torch.get_default_dtype()} Time: {execution_time} ms per loop")

    loop_iters = 1000
    output=None
    with torch.no_grad():
        for i in range(loop_iters):
            input_ids_local = input_ids + i
            output_new, times_ms = llm(input_ids_local, profile_engram=True)
            output = output_new + output if output is not None else output_new
            htcore.mark_step()
    times_ms.total = execution_time
    print(f"\nForward Complete with Timing!\nProfiling times_ms:\n{times_ms}")

    output = None
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=5, active=5, repeat=1),
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
        #record_shapes=True,
        #profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile_logs', use_gzip=True)
    ) as profiler:
        with torch.no_grad():
            for i in range(int(10)):
                output, _ =  llm(input_ids)
                htcore.mark_step()
                htcore.hpu.synchronize()
                profiler.step()
    trace_file = get_latest_trace_file(log_dir="./profile_logs")
    print(f"\nForward Complete with Profiling!\nTrace file saved at: {trace_file}")

