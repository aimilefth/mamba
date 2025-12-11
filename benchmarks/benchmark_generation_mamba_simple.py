import argparse
import time
import json

import torch
import torch.nn.functional as F

from einops import rearrange

from transformers import AutoTokenizer, AutoModelForCausalLM

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
# <--- Added imports for manual config loading
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from mamba_ssm.models.config_mamba import MambaConfig


parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--model-name", type=str, default="state-spaces/mamba-130m")
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--promptlen", type=int, default=100)
parser.add_argument("--genlen", type=int, default=100)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=1.0)
parser.add_argument("--minp", type=float, default=0.0)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--use-mamba2-simple", action="store_true", help="Use Mamba2Simple implementation") # <--- New Arg
args = parser.parse_args()

repeats = 3
device = "cuda"
dtype = torch.float16
torch.manual_seed(0)

print(f"Loading model {args.model_name}")
is_mamba = args.model_name.startswith("state-spaces/mamba") or args.model_name.startswith("state-spaces/transformerpp")
if is_mamba:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    # <--- Modified loading logic to support swapping the layer type
    if args.use_mamba2_simple:
        print("Using Mamba2Simple implementation.")
        config_data = load_config_hf(args.model_name)
        # Override the layer type in ssm_cfg
        if 'ssm_cfg' not in config_data or config_data['ssm_cfg'] is None:
            config_data['ssm_cfg'] = {}
        config_data['ssm_cfg']['layer'] = "Mamba2Simple"
        
        config = MambaConfig(**config_data)
        model = MambaLMHeadModel(config, device=device, dtype=dtype)
        
        # Load weights manually
        state_dict = load_state_dict_hf(args.model_name, device=device, dtype=dtype)
        model.load_state_dict(state_dict)
    else:
        model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map={"": device}, torch_dtype=dtype)
model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

if args.prompt is None:
    input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device="cuda")
    attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
else:
    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)
    attn_mask = tokens.attention_mask.to(device=device)
max_length = input_ids.shape[1] + args.genlen

if is_mamba:
    fn = lambda: model.generate(
        input_ids=input_ids,
        max_length=max_length,
        cg=True,
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=False,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        min_p=args.minp,
        repetition_penalty=args.repetition_penalty,
    )
else:
    fn = lambda: model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_length=max_length,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        repetition_penalty=args.repetition_penalty,
    )
out = fn()
print(out)
if args.prompt is not None:
    print(tokenizer.batch_decode(out.sequences.tolist()))

torch.cuda.synchronize()
start = time.time()
for _ in range(repeats):
    fn()
torch.cuda.synchronize()
print(f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}")
print(f"{args.model_name} prompt processing + decoding time: {(time.time() - start) / repeats * 1000:.0f}ms")