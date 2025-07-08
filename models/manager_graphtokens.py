# models/manager_graphtokens.py
import torch
import torch.nn as nn
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model

from models.gcn import RGCNEncoder
from models.graph_builder import EDGE_TYPE  # 5 relations

# ─────────────────────────────────────────────────────────────
class GraphTokenManager(nn.Module):
    """
    1. RGCN (768)  →  proj_up (→4096) : 노드 수 N 만큼 '가상 토큰'
    2. prompt 임베딩과 concat          : [prompt_tokens | graph_tokens]
    3. ChatGLM2 (+LoRA)
    4. 분류 헤드 or LM decoding
    """

    def __init__(
        self,
        glm_ckpt: str = "THUDM/chatglm2-6b",
        gcn_layers: int = 5,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()

        # R-GCN (노드 768 → 768)
        self.gcn = RGCNEncoder(
            in_dim=768,
            num_rel=len(EDGE_TYPE),
            num_layers=gcn_layers,
        )

        # 768 → 4096 projection (각 노드별)
        self.proj_up = nn.Linear(768, 4096, bias=False)

        # ChatGLM2 (8-bit) + LoRA
        self.tokenizer = AutoTokenizer.from_pretrained(
            glm_ckpt, trust_remote_code=True
        )
        self.glm = AutoModel.from_pretrained(
            glm_ckpt,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            trust_remote_code=True,
        )
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.glm = get_peft_model(self.glm, lora_cfg)

        # 4096 → 1 로짓 (sigmoid 는 BCEWithLogitsLoss 내부 처리)
        self.cls_head = nn.Linear(4096, 1)

    # ────────────────────────────────────────────────────────
    def forward(
        self,
        data: Data,       # PyG 그래프 (GPU 로 .to(device) 해둘 것)
        person: str,      # 인물 이름 (str)
        label: torch.Tensor | None = None,  # [1] 0/1 float (옵션)
    ):
        """
        returns: (logits, loss_optional)
        """
        device = next(self.parameters()).device

        # 1) 노드 업데이트
        node_h = self.gcn(data.x, data.edge_index, data.edge_type)   # [N,768]

        # 2) 4096-d 가상 토큰 시퀀스
        graph_tokens = self.proj_up(node_h).unsqueeze(0)             # [1,N,4096]

        # 3) prompt → 임베딩
        prompt = (
            "Instruction: Please predict whether Google search volume for "
            f"{person} will exhibit an anomaly (0=no, 1=yes) within 3 days.\n"
            "Input: "
        )
        tok = self.tokenizer(prompt, return_tensors="pt").to(device)
        prompt_emb = self.glm.get_input_embeddings()(tok.input_ids)  # [1,L,4096]

        # 4) concat & attention mask
        inputs_embeds = torch.cat([prompt_emb, graph_tokens.to(device)], dim=1)  # [1,L+N,4096]
        attn_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=device)

        # 5) LLM forward (hidden states 필요)
        out = self.glm(inputs_embeds=inputs_embeds,
                       attention_mask=attn_mask,
                       output_hidden_states=True)
        # prefix: 그래프 첫 토큰은 prompt 뒤 첫 위치 = index prompt_len
        graph_start_idx = prompt_emb.size(1)
        # 평균 풀링으로 그래프 구간 요약 (4096d)
        graph_repr = out.hidden_states[-1][:, graph_start_idx:, :].mean(dim=1)   # [1,4096]

        # 6) 분류 헤드
        logits = self.cls_head(graph_repr).squeeze(-1)   # [1]

        if label is None:
            return logits, None

        loss = nn.BCEWithLogitsLoss()(logits, label.to(device))
        return logits, loss
