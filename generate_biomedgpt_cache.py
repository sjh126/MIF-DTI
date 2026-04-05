import torch
from transformers import OFATokenizer, OFAModel
from tqdm import tqdm

# ======== 配置 ========
model_path = '/home/sjh/DTI/BiomedGPT/models/BiomedGPT-Base-Pretrained'
input_txt = '/home/sjh/DTI/MIF-DTI/DataSets/Davis.txt'
output_drug = '/home/sjh/DTI/MIF-DTI/DataSets/Preprocessed/biomedgpt_drugs.pt'
output_target = '/home/sjh/DTI/MIF-DTI/DataSets/Preprocessed/biomedgpt_targets.pt'
device = "cuda" if torch.cuda.is_available() else "cpu"

# ======== 加载 BiomedGPT ========
print("Loading BiomedGPT model...")
tokenizer = OFATokenizer.from_pretrained(model_path)
model = OFAModel.from_pretrained(model_path).to(device)
model.eval()

@torch.no_grad()
def get_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.encoder(input_ids=inputs["input_ids"].to(device))
    hidden = outputs.last_hidden_state
    emb = hidden.mean(dim=1).squeeze(0).cpu()
    return emb  # [768]

# ======== 读取 Davis.txt，提取唯一 SMILES 和 蛋白序列 ========
smiles_set = set()
seq_set = set()
with open(input_txt, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        smiles_set.add(parts[2])
        seq_set.add(parts[3])

print(f"🧪 Unique drugs: {len(smiles_set)} | 🧬 Unique targets: {len(seq_set)}")

# ======== 提取 Drug 表征 ========
drug_dict = {}
for smiles in tqdm(smiles_set, desc="Encoding drugs"):
    try:
        drug_dict[smiles] = get_embedding(smiles)
    except Exception as e:
        print(f"❌ Drug encode error: {e}")

torch.save(drug_dict, output_drug)
print(f"✅ Saved {len(drug_dict)} drug embeddings to {output_drug}")

# ======== 提取 Target 表征 ========
target_dict = {}
for seq in tqdm(seq_set, desc="Encoding targets"):
    try:
        target_dict[seq] = get_embedding(seq)
    except Exception as e:
        print(f"❌ Target encode error: {e}")

torch.save(target_dict, output_target)
print(f"✅ Saved {len(target_dict)} target embeddings to {output_target}")