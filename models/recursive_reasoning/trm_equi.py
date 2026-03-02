from typing import Tuple, List, Dict
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from models.common import trunc_normal_init_
from models.layers import rms_norm, CastedLinear, SwiGLU, RotaryEmbedding, RotaryEmbedding2d, CosSin, apply_rotary_pos_emb
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class SERRM_InnerCarry:
    z: torch.Tensor

@dataclass
class SERRM_Carry:
    inner_carry: SERRM_InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


class SERRM_Config(BaseModel):
    batch_size: int
    seq_len: int
    # treated as bool (0 = False), int because of backward compatibility
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int  # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    num_heads_t: int
    head_dim: int
    head_dim_t: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    equivariant_symbols: bool = True
    add_tokens: int = 0
    dropout: float = 0.0


class Attention(nn.Module):
    '''Axial Attention module: gets a 4 dimensional tensor (batch_size, first_axis, second_axis, hidden_dim)
    and performs axial attention. First axis is in the batch-dimension, mha is performed over second axis.
    If cos_sin is not None, rope or rope2d is only performed on the sequence an not on any additional tokens
    (puzzle_emb,...) '''
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False, dropout=0.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.dropout = dropout

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
                                     bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, mask=None) -> torch.Tensor:
        B, C, S, H = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        # first axis to batch-dimension
        qkv = qkv.view(B*C, S, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        if cos_sin is not None:
            cos, sin = cos_sin
            num_enc = cos.size(0)
            # only perform rope(2d) on the original seq_len
            query_enc = query[:, :num_enc]
            key_enc = key[:, :num_enc]
            query_enc, key_enc = apply_rotary_pos_emb(query_enc, key_enc, cos, sin)
            query = torch.cat([query_enc, query[:, num_enc:]], dim=1)
            key = torch.cat([key_enc, key[:, num_enc:]], dim=1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=mask,  # or your mask (broadcastable to [B, Hq, L, S])
            dropout_p=self.dropout,
            is_causal=False,
        ).transpose(1, 2)
        # attn_output: [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.view(B, C, S, self.output_size)  # type: ignore

        return self.o_proj(attn_output)

class SERRMBlock(nn.Module):
    def __init__(self, config: SERRM_Config) -> None:
        super().__init__()

        # first attention is only over positions (common attention)
        self.attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
            dropout=config.dropout)
        # second block for attention over symbols
        self.attn_t = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim_t,
            num_heads=config.num_heads_t,
            num_key_value_heads=config.num_heads_t,
            causal=False,
            dropout=config.dropout)

        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(
            hidden_states + self.attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(
            hidden_states + self.attn_t(cos_sin=None, hidden_states=hidden_states.transpose(1, 2)).transpose(1, 2),
            variance_epsilon=self.norm_eps,
        )
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class SERRM_ReasoningModule(nn.Module):
    def __init__(self, layers: List[SERRMBlock]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class SERRM_Inner(nn.Module):
    def __init__(self, config: SERRM_Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = (torch.ones(1) * self.config.vocab_size).sqrt().to(self.forward_dtype)

        if config.puzzle_emb_ndim > 0:
            # puzzle embeddings cannot be equivariant for every individual puzzle:
            # for e.g. tasks as color every yellow field blue
            # this is still equivariant for the whole task
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers,
                                                    self.config.vocab_size + self.config.hidden_size,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # additional tokens that are appended to the sequence (same as puzzle embeddings)
        if config.add_tokens > 0:
            if self.config.equivariant_symbols:
                # Zero init puzzle embeddings
                self.add_tokens_special = nn.Parameter(torch.zeros(1, 2, self.config.add_tokens, self.config.hidden_size, dtype=self.forward_dtype))
                self.add_tokens = nn.Parameter(torch.zeros(1, 1, self.config.add_tokens, self.config.hidden_size, dtype=self.forward_dtype))
                trunc_normal_init_(self.add_tokens_special)
                trunc_normal_init_(self.add_tokens)
            else:
                # Zero init puzzle embeddings
                self.add_tokens = nn.Parameter(
                    torch.zeros(1, self.config.vocab_size, self.config.add_tokens, self.config.hidden_size, dtype=self.forward_dtype))
                trunc_normal_init_(self.add_tokens)

        if config.equivariant_symbols:
            self.lm_head = CastedLinear(self.config.hidden_size, 1, bias=False)
        else:
            # if the symbols are not equivariant, all symbols can be used to predict the output
            self.lm_head = CastedLinear(self.config.hidden_size * self.config.vocab_size, self.config.vocab_size, bias=False)

        if self.config.pos_encodings == 'rope':
            self.rotary_emb = RotaryEmbedding(dim=self.config.head_dim,
                                              max_position_embeddings=self.config.seq_len,
                                              base=self.config.rope_theta)
            self.pos_embedding = torch.zeros((1, 1, config.seq_len, config.hidden_size), dtype=self.forward_dtype)
        elif self.config.pos_encodings == 'rope2d':
            self.rotary_emb = RotaryEmbedding2d(dim=self.config.head_dim,
                                              max_position_embeddings=self.config.seq_len,
                                              base=self.config.rope_theta)
            self.pos_embedding = torch.zeros((1, 1, config.seq_len, config.hidden_size), dtype=self.forward_dtype)
        else:
            # if not specified otherwise, every input embedding is a learnable parameter
            self.pos_embedding = nn.Parameter(
                torch.empty((1, 1, config.seq_len, config.hidden_size), dtype=self.forward_dtype))
            trunc_normal_init_(self.pos_embedding)

        if config.equivariant_symbols:
            # for Sudoku, Maze, ARC the first two tokens are special tokens (padding, empty, EOS,..)
            # adjust for different datasets
            self.embed_special_tokens = nn.Parameter(
                torch.empty(size=(1, 2, 1, config.hidden_size), dtype=self.forward_dtype))
            trunc_normal_init_(self.embed_special_tokens, std=1)
            self.embed_symbol_tokens = nn.Parameter(
                torch.empty(size=(1, 1, 1, config.hidden_size), dtype=self.forward_dtype))
            trunc_normal_init_(self.embed_symbol_tokens, std=1)
        else:
            self.embed_symbols = nn.Parameter(
                torch.empty(size=(1, config.vocab_size, 1, config.hidden_size), dtype=self.forward_dtype))
            trunc_normal_init_(self.embed_symbols, std=1)

        # Reasoning Layers
        self.L_level = SERRM_ReasoningModule(
            layers=[SERRMBlock(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.Z_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers):
        # Indicate where to put the embedding for the symbol
        # there is only one symbol per position
        embedding = F.one_hot(input.to(torch.long), num_classes=self.config.vocab_size).to(
            self.forward_dtype)

        if self.config.equivariant_symbols:
            symbol_embedding = torch.cat([self.embed_special_tokens,
                                   self.embed_symbol_tokens.repeat(1, self.config.vocab_size - 2, 1, 1)
                                   ], dim=1)
        else:
            symbol_embedding = self.embed_symbols

        embedding = embedding.transpose(1, 2) * self.embed_scale.to(embedding.device)
        embedding = embedding.unsqueeze(dim=-1)
        # only the token where a symbol is present gets the symbol_embedding
        # empty tokens are initialized with 0
        # all tokens get a positional embedding (0 in case of rope)
        embedding = (embedding * symbol_embedding + self.pos_embedding)
        B, C, S, D = embedding.shape

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            puzzle_embedding = puzzle_embedding.view(B, 1, 1, D + C)
            puzzle_embedding_color = puzzle_embedding[:, :, :, D:].transpose(1, 3)
            puzzle_embedding_global = puzzle_embedding[:, :, :, :D]
            embedding = embedding + puzzle_embedding_color + puzzle_embedding_global

        if self.config.add_tokens > 0:
            if self.config.equivariant_symbols:
                add_tokens = torch.cat([self.add_tokens_special.repeat(B, 1, 1, 1),
                                              self.add_tokens.repeat(B, C - 2, 1, 1)], dim=1)
                embedding = torch.cat([embedding, add_tokens], dim=2)
            else:
                add_tokens = self.add_tokens.repeat(B, 1, 1, 1)
                embedding = torch.cat([embedding, add_tokens], dim=2)

        # Scale
        return embedding

    def empty_carry(self, batch_size: int):
        return SERRM_InnerCarry(
            z=torch.empty(batch_size, self.config.vocab_size,
                          self.config.seq_len + self.config.add_tokens,
                          self.config.hidden_size, dtype=self.forward_dtype)
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: SERRM_InnerCarry):
        return SERRM_InnerCarry(
            z=torch.where(reset_flag.view(-1, 1, 1, 1), self.Z_init, carry.z),
        )

    def forward(self, carry: SERRM_InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[
        SERRM_InnerCarry, torch.Tensor, torch.Tensor]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        B, C, S, D = input_embeddings.shape

        # Forward iterations
        # there is only one hidden variable z
        # no distinction between higher and lower modules
        z = carry.z
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z = self.L_level(z, input_embeddings, **seq_info)
        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z = self.L_level(z, input_embeddings, **seq_info)

        new_carry = SERRM_InnerCarry(z=z.detach())  # New carry no grad

        # LM Outputs
        # cut away all additional tokens
        z = z[:, :self.config.vocab_size, :self.config.seq_len]
        if self.config.equivariant_symbols:
            output = self.lm_head(z).squeeze(dim=-1)
            output = output.transpose(1, 2)
        else:
            output = z.transpose(1, 2).reshape(B, self.config.seq_len, C * D)
            output = self.lm_head(output).squeeze(dim=-1)

        # the probability of all tokens being correct is estimated
        # no need for an additional token and additional loss
        with torch.no_grad():
            output_prob = F.softmax(output.to(torch.float32), dim=-1)
            output_prob = torch.clamp(output_prob.amax(dim=-1).amin(dim=-1), min=0, max=1)

        return new_carry, output, output_prob


class SERRM(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = SERRM_Config(**config_dict)
        self.inner = SERRM_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return SERRM_Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            # Empty is expected, it will be reseted in first pass as all sequences are halted.

            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),  # Default to halted

            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(self, carry: SERRM_Carry, batch: Dict[str, torch.Tensor]) -> Tuple[
        SERRM_Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v) for k, v
                            in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, probs = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": probs,
            #"q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # if training, and ACT is enabled
            # max steps are still considered
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                # Exploration
                rand_stop = torch.rand_like(probs) < self.config.halt_exploration_prob
                halted = halted | rand_stop

        return SERRM_Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
