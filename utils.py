from fileinput import filename
import json
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/mnt/ssd1/zhanghanxiu/cache/huggingface'

import csv
import pandas as pd
import ast
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig, AutoTokenizer, LlamaTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import os 
import torch.nn.functional as F
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../simnet/')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

k = 256 # number of eigenvalues/singular values to keep
select_index = [0,1,2,3,4,5,6,7] # layers to select for fingerprinting
m = 8 # total number of layers to consider for fingerprinting

ev_filename = "ev.csv" # filename to save eigenvalues
sv_filename = "sv.csv" # filename to save singular values
file_list = [ev_filename, sv_filename]

def get_fp(model_path, file_list=file_list, k=k, m=m, select_index=select_index, mid=False, mid_num=0, quantize=None):
    if quantize is not None:
        try:
            fp = read_fp(model_path+quantize, file_list=file_list, k=k, m=m)
            return fp
        except:
            if quantize == '4bit':
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True
            )
            elif quantize == '8bit':
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            try:
                model = AutoModel.from_pretrained(model_path, quantization_config=bnb_config,device_map="cpu")
            except:
                model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config,device_map="cpu")
            q_list, k_list= extract_q_k_from_llama_model(model)
            head_dim = get_head_dim_from_model_name(model_path)
            qk_ev = cal_ev_from_q_k(q_list, k_list, select_index=select_index, mid=mid, mid_num=mid_num, head_dim=head_dim)
            qk_sv = cal_sv_from_q_k(q_list, k_list, select_index=select_index, mid=mid, mid_num=mid_num, head_dim=head_dim)
            save_half_fp(model_path+quantize, qk_ev, filename=file_list[0])
            save_half_fp(model_path+quantize, qk_sv, filename=file_list[1])
            save_half_fp(model_path+quantize, vo_ev, filename=file_list[2])
            save_half_fp(model_path+quantize, vo_sv, filename=file_list[3])
            fp = read_fp(model_path+quantize, file_list=file_list, k=k, m=m)
            return fp

    else:
        try:
            fp = read_fp(model_path, file_list=file_list, k=k, m=m)
            return fp
        except:
            q_list, k_list= extract_q_k(model_path)
            head_dim = get_head_dim_from_model_name(model_path)
            qk_ev = cal_ev_from_q_k(q_list, k_list, select_index=select_index, mid=mid, mid_num=mid_num, head_dim=head_dim)
            qk_sv = cal_sv_from_q_k(q_list, k_list, select_index=select_index, mid=mid, mid_num=mid_num, head_dim=head_dim)
            
            save_half_fp(model_path, qk_ev, filename=file_list[0])
            save_half_fp(model_path, qk_sv, filename=file_list[1])
            
            fp = read_fp(model_path, file_list=file_list, k=k, m=m)
        return fp

def read_fp(model_path, file_list, k, m):
    fp_parts = []
    for filename in file_list:
        part = read_half_fp(model_path, filename=filename)[:m,:k]
        part = F.normalize(part, p=2, dim=-1)
        fp_parts.append(part)
    return torch.cat(fp_parts, dim=0)
    

# Get half fingerprint (either eigenvalues or singular values) from saved file
def read_half_fp(model_path, filename):
    df_fp = pd.read_csv(filename, index_col=0, header=None)
    df_fp = df_fp[~df_fp.index.duplicated(keep='first')]
    try:
        sv = torch.tensor(df_fp.loc[model_path].values)
    except:
        tensor_str_list = df_fp.loc[model_path].dropna().values
        tensor_list = [ast.literal_eval(s) for s in tensor_str_list]
        sv = torch.tensor(tensor_list)
    return sv

def get_head_dim_from_model_name(model_name):
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except:
        print(f"Failed to load config for {model_name}. Using default head_dim=128")
        return 128
    if hasattr(config, "num_attention_heads") and hasattr(config, "hidden_size"):
        head_dim = config.hidden_size // config.num_attention_heads
        return head_dim
    else:
        print(f"{model_name} does not have num_attention_heads or hidden_size attribute in its config. Using default head_dim=128")
        return 128

# Compute L1 distance between two fingerprints
def get_fp_distance(fp1, fp2):
    return torch.norm(fp1.to(device) - fp2.to(device), p=1).item()

# get similarity score from SimNet based on fingerprint
def get_sim(fp, simnet_path):
    model = (torch.load(simnet_path, weights_only=False))
    model.eval()
    with torch.no_grad():
        sim = model(fp.unsqueeze(0).to(device).float()).item()
    return sim


def fp_dist_from_path(model_path_1, model_path_2, file_list=file_list, k=k, m=m, select_index=select_index, mid=False, mid_num=0):  
    fp1 = get_fp(model_path_1, file_list=file_list, k=k, m=m, select_index = select_index, mid=mid, mid_num=mid_num)
    fp2 = get_fp(model_path_2, file_list=file_list, k=k, m=m, select_index = select_index, mid=mid, mid_num=mid_num)
    dist = get_fp_distance(fp1, fp2)
    return dist


# Save half fingerprint (either eigenvalues or singular values) to file
def save_half_fp(model_path, sv, filename):
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([model_path] + sv.tolist())

# Extract Q and K weights from the model based on its architecture
def extract_q_k(model_name):
    
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    except:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    q_list, k_list = [], []
    print(model_name)
    if "TableGPT" in model_name or "MiniGPT" in model_name or "alpaca" in model_name or "baize" in model_name or "Sheared-LLaMA" in model_name or "prune_sparsegpt" in model_name:
        return extract_q_k_from_llama_model(model)
    elif 'finetune_self' in model_name:  # for finetuned model
        config_file = os.path.join(model_name, "adapter_config.json")
        with open(config_file, "r") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config["base_model_name_or_path"]
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
        if "llama" in model_name:
            tokenizer = LlamaTokenizer.from_pretrained(base_model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        try:
            peft_model = PeftModel.from_pretrained(
                base_model,
                model_name,
                torch_dtype=torch.float16,
            )
        except:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            base_model.resize_token_embeddings(len(tokenizer))
            base_model.config.vocab_size = len(tokenizer)
            peft_model = PeftModel.from_pretrained(
                base_model,
                model_name,
                torch_dtype=torch.float16,
            )
        peft_model.merge_and_unload()
        model = peft_model.model.model
        return extract_q_k_from_llama_model(model)
    elif "Baichuan" in model_name:
        print("Baichuan model detected")
        for layer in model.model.layers:
            Q = layer.self_attn.W_pack.weight.data[0:4096,:]
            K = layer.self_attn.W_pack.weight.data[4096:8192,:]
            q_list.append(Q)
            k_list.append(K)
    elif "internlm" in model_name:
        head_dim = 128
        group_num = 8
        head_num = 32
        for layer in model.model.layers:
            Q = layer.attention.wqkv.weight.data[0:head_dim*head_num,:]
            K = layer.attention.wqkv.weight.data[head_dim*head_num:head_dim*head_num+head_dim*group_num,:]
            q_list.append(Q)
            k_list.append(K)
    elif "gpt" in model_name or "GPT" in model_name:
        nf = model.h[0].attn.c_attn.weight.data.shape[1] // 3
        for layer in model.h:
            Q = layer.attn.c_attn.weight.data.T[0:nf,:]
            K = layer.attn.c_attn.weight.data.T[nf:2*nf,:]
            q_list.append(Q)
            k_list.append(K)
    elif "glm" in model_name:
        for layer in model.transformer.encoder.layers:
            head_dim = 128
            group_num = 2
            head_num = 32
            Q = layer.self_attention.query_key_value.weight.data[0:head_dim*head_num,:]
            K = layer.self_attention.query_key_value.weight.data[head_dim*head_num:head_dim*head_num+head_dim*group_num,:]
            q_list.append(Q.float())
            k_list.append(K.float())
    elif "opt" in model_name:
        for layer in model.decoder.layers:
            Q = layer.self_attn.q_proj.weight.data
            K = layer.self_attn.k_proj.weight.data
            q_list.append(Q)
            k_list.append(K)
    elif "pythia" in model_name:
        for layer in model.layers:
            Q = layer.attention.query_key_value.weight.data[0:4096,:]
            K = layer.attention.query_key_value.weight.data[4096:8192,:]
            q_list.append(Q)
            k_list.append(K)
    elif "mpt" in model_name:
        for layer in model.transformer.blocks:
            Q = layer.attn.Wqkv.weight.data[0:4096,:]
            K = layer.attn.Wqkv.weight.data[4096:8192,:]
            q_list.append(Q)
            k_list.append(K)
    elif "TinyLlama" in model_name:
        for layer in model.layers:
            Q = layer.self_attn.q_proj.weight.data
            K = layer.self_attn.k_proj.weight.data
            q_list.append(Q)
            k_list.append(K)
    else:
        return extract_q_k_from_llama_model(model)
    return q_list, k_list
    
def extract_q_k_from_llama_model(model):
    q_list, k_list = [], []
    for layer in model.layers:
        Q = layer.self_attn.q_proj.weight.data
        K = layer.self_attn.k_proj.weight.data
        q_list.append(Q)
        k_list.append(K)
    return q_list, k_list


# Important note: The Q, K matrices obtained .weight.data are of shape (out_features, in_features), 
# which is the transpose of the usual notation (in_features, out_features)!!!

def construct_qk_sv_invariant_matrix(q, k, head_dim=128):
    if q.shape[0] == k.shape[0]:
        return q.T.float() @ k.float()
    else:
        print("GQA/MQA detected, extending K")
        k_ext = extend_kv(k, kv_head_num=k.shape[0]//head_dim, q_head_num=q.shape[0]//head_dim, head_dim=head_dim).to(device)
        return q.T.float() @ k_ext.float()
    
def construct_qk_ev_invariant_matrix(q, k, head_dim=128):
    if q.shape[0] == k.shape[0]:
        # return q @ q.T @ k @ k.T
        return q.float() @ k.T.float()
    else:
        print("GQA/MQA detected, extending K")
        k_ext = extend_kv(k, kv_head_num=k.shape[0]//head_dim, q_head_num=q.shape[0]//head_dim, head_dim=head_dim).to(device)
        # return q @ q.T @ k_ext @ k_ext.T
        return q.float() @ k_ext.T.float()

        return o.T.float() @ v_ext.T.float()
def extend_kv(kv, kv_head_num, q_head_num, head_dim):
    kv_ext = torch.zeros(q_head_num * head_dim, kv.shape[1])
    copy_num = q_head_num // kv_head_num
    for h in range(kv_head_num):
        kv_head = kv[h*head_dim:(h+1)*head_dim, :]
        for c in range(copy_num):
            kv_ext[(h*copy_num + c)*head_dim:(h*copy_num + c + 1)*head_dim, :] = kv_head
    return kv_ext

# Calculate singular values from singular value invariant matrix
def cal_sv_from_q_k(q_list, k_list, select_index, mid=False, mid_num=2,head_dim=128):
    QK_list = []
    for i in select_index:
        QK_list.append(construct_qk_sv_invariant_matrix(q_list[i].to(device), k_list[i].to(device), head_dim=head_dim))
        torch.cuda.empty_cache()
    if mid:
        mid_layer_start = len(q_list) // 2 - mid_num // 2
        mid_layer_end = mid_layer_start + mid_num
        for mid_layer in range(mid_layer_start, mid_layer_end):
            print(f'Calculating {mid_layer}-th layer')
            QK_list.append(construct_qk_sv_invariant_matrix(q_list[mid_layer].to(device), k_list[mid_layer].to(device), head_dim=head_dim))
            torch.cuda.empty_cache()
    svs = []
    for mat in tqdm(QK_list, desc="Calculating singular values"):
        _, sv, _ = torch.linalg.svd(mat.float().to(device), full_matrices=False)
        svs.append(sv)
    svs = torch.stack(svs, dim=0).to(device)
    torch.cuda.empty_cache()
    return svs

# Calculate eigenvalues from eigenvalue invariant matrix
def cal_ev_from_q_k(q_list, k_list, select_index, mid=False, mid_num=2, head_dim=128):
    QK_list = []
    for i in select_index:
        QK_list.append(construct_qk_ev_invariant_matrix(q_list[i].to(device), k_list[i].to(device), head_dim=head_dim))
    if mid:
        mid_layer_start = len(q_list) // 2 - mid_num // 2
        mid_layer_end = mid_layer_start + mid_num
        for mid_layer in range(mid_layer_start, mid_layer_end):
            QK_list.append(construct_qk_ev_invariant_matrix(q_list[mid_layer].to(device), k_list[mid_layer].to(device), head_dim=head_dim))
    evs = []
    QK_list = [mat for mat in QK_list]
    for mat in tqdm(QK_list, desc="Calculating eigenvalues"):
        ev = torch.linalg.eigvals(mat.float().to(device))
        ev = ev.abs()
        ev, _ = torch.sort(ev, descending=True)
        evs.append(ev)
    evs = torch.stack(evs, dim=0).to(device)
    torch.cuda.empty_cache()
    return evs