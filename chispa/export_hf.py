"""Convierte un checkpoint de CHISPA-lab (sin memoria) a LlamaForCausalLM de Hugging Face y verifica la equivalencia.

Uso: python -m chispa.export_hf --modelo RUN/final.pt --datos DATOS --out hf/chispa-20m-python
Diferencias que se resuelven: RoPE intercalado (pares 2i, 2i+1) -> rotate_half (i, i+hd/2) permutando las salidas de
q y k dentro de cada cabeza; SwiGLU fusionada w12 -> gate_proj / up_proj; cabeza atada -> tie_word_embeddings.
"""
from __future__ import annotations

import argparse, json, shutil
from pathlib import Path

import torch

from .model import cargar_modelo


def permutacion(h, hd):
    idx = torch.cat([torch.arange(0, hd, 2), torch.arange(1, hd, 2)])
    return torch.cat([idx + i * hd for i in range(h)])


def convertir(ruta, datos, out, tol=2e-3):
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    m, ck = cargar_modelo(ruta, "cpu"); m.eval()
    c = m.c
    assert not c.memoria, "solo modelos base (sin cabeza de memoria)"
    hd = c.d // c.n_head
    cfg = LlamaConfig(vocab_size=c.vocab, hidden_size=c.d, intermediate_size=c.ffn, num_hidden_layers=c.n_layer,
                      num_attention_heads=c.n_head, num_key_value_heads=c.n_head, head_dim=hd, hidden_act="silu",
                      max_position_embeddings=c.ctx, rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=True,
                      attention_bias=False, mlp_bias=False, torch_dtype="float32")
    sd = m.state_dict(); P = permutacion(c.n_head, hd); new = {"model.embed_tokens.weight": sd["emb.weight"],
                                                                 "model.norm.weight": sd["nf.w"]}
    for i in range(c.n_layer):
        q, k, v = sd[f"blocks.{i}.qkv.weight"].split(c.d, 0)
        g, u = sd[f"blocks.{i}.w12.weight"].split(c.ffn, 0)
        pre = f"model.layers.{i}."
        new.update({pre + "self_attn.q_proj.weight": q[P], pre + "self_attn.k_proj.weight": k[P],
                    pre + "self_attn.v_proj.weight": v, pre + "self_attn.o_proj.weight": sd[f"blocks.{i}.o.weight"],
                    pre + "mlp.gate_proj.weight": g, pre + "mlp.up_proj.weight": u, pre + "mlp.down_proj.weight": sd[f"blocks.{i}.w3.weight"],
                    pre + "input_layernorm.weight": sd[f"blocks.{i}.n1.w"], pre + "post_attention_layernorm.weight": sd[f"blocks.{i}.n2.w"]})
    hf = LlamaForCausalLM(cfg)
    faltan, sobran = hf.load_state_dict(new, strict=False)
    faltan = [k for k in faltan if k != "lm_head.weight"]
    assert not faltan and not sobran, (faltan, sobran)
    hf.tie_weights(); hf.eval()
    # verificación numérica
    x = torch.randint(0, c.vocab, (2, 200), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        a = m.logits(m(x)); b = hf(x).logits
    dif = float((a - b).abs().max())
    assert dif < tol, f"los logits no coinciden: {dif}"
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(out, safe_serialization=True)
    tk = PreTrainedTokenizerFast(tokenizer_file=str(Path(datos) / "tok" / "tokenizer.json"),
                                 eos_token="<|endoftext|>", bos_token="<|endoftext|>", pad_token="<pad>",
                                 additional_special_tokens=["<fim_prefix>", "<fim_suffix>", "<fim_middle>"])
    tk.model_max_length = c.ctx
    tk.save_pretrained(out)
    json.dump({"origen": str(ruta), "paso": ck.get("step"), "dif_max_logits": dif}, open(out / "chispa_export.json", "w"), indent=1)
    print(f"exportado a {out}; diferencia máxima de logits {dif:.2e}", flush=True)
    return dif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True); ap.add_argument("--datos", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    convertir(a.modelo, a.datos, a.out)


if __name__ == "__main__":
    main()
