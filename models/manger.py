# models/manager.py
import torch
import torch.nn as nn
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model

from models.gcn import RGCNEncoder           # 앞서 만든 GCN
from models.graph_builder import EDGE_TYPE   # 관계 개수 = 5

# ─────────────────────────────────────────────────────────────
class ManagerModel(nn.Module):
    """
    RGCN 768d  → pool → Linear(768→4096) → ChatGLM2 prefix
                    ↘────────── BCE Head (0/1)
    """

    def __init__(
        self,
        glm_ckpt: str = "THUDM/chatglm2-6b",
        lora_r: int   = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        gcn_layers: int = 2,
    ):
        super().__init__()

        # 1) R-GCN
        self.gcn = RGCNEncoder(
            in_dim=768, num_rel=len(EDGE_TYPE), num_layers=gcn_layers
        )

        # 2) 768 → 4096 prefix 프로젝션
        self.proj_up = nn.Linear(768, 4096, bias=False)

        # 3) ChatGLM2 (8-bit 로드 권장)
        self.tokenizer = AutoTokenizer.from_pretrained(glm_ckpt, trust_remote_code=True)
        self.glm = AutoModel.from_pretrained(
            glm_ckpt,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            trust_remote_code=True,
        )

        # 4) LoRA 적용 (query/key/value/output proj)
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.glm = get_peft_model(self.glm, lora_cfg)

        # 5) BCE 분류 헤드
        self.cls_head = nn.Linear(4096, 1)

    # ────────────────────────────────────────────────────────
    def graph_prefix(self, data: Data) -> torch.Tensor:
        """GCN + mean-pool + 4096 proj → [1,4096]"""
        node_h = self.gcn(data.x, data.edge_index, data.edge_type)   # [N,768]
        g_repr = node_h.mean(dim=0, keepdim=True)                    # [1,768]
        return self.proj_up(g_repr)                                  # [1,4096]

    # ────────────────────────────────────────────────────────
    def forward(self, data: Data, person: str) -> torch.Tensor:
        """
        returns: logits (before sigmoid) of shape [1]
        """
        device = next(self.parameters()).device

        # ① 그래프 prefix
        prefix = self.graph_prefix(data).to(device)                  # [1,4096]

        # ② LLM 프롬프트 (person 이름 포함)
        prompt = (
            f"Please predict whether Google search volume for {person} "
            f"will show an anomaly within 3 days after this video. "
            f"Answer 0 or 1."
        )
        tok = self.tokenizer(prompt, return_tensors="pt").to(device)
        embed_matrix = self.glm.get_input_embeddings()(tok.input_ids)  # [1,L,4096]

        # ③ Prefix + prompt embeds
        inputs_embeds = torch.cat([prefix.unsqueeze(1), embed_matrix], dim=1)  # [1,L+1,4096]

        # ④ ChatGLM2 forward  (labels X, hidden states O)
        out = self.glm(inputs_embeds=inputs_embeds, output_hidden_states=True)
        rep = out.hidden_states[-1][:, 0, :]      # prefix 위치의 4096-d

        # ⑤ 분류 헤드
        logit = self.cls_head(rep).squeeze(-1)    # [1]
        return logit
