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
        # 모델 레이어들을 올릴 장치를 정의합니다.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # R-GCN 레이어를 device로 이동
        self.gcn = RGCNEncoder(
            in_dim=768,
            num_rel=len(EDGE_TYPE),
            num_layers=gcn_layers,
        ).to(device)

        # 프로젝션 레이어를 device로 이동
        self.proj_up = nn.Linear(768, 4096, bias=False).to(device)

        # ChatGLM2 + LoRA (FP16, no 8-bit quantization)
        self.tokenizer = AutoTokenizer.from_pretrained(
            glm_ckpt, trust_remote_code=True
        )
        # FP16 with automatic device placement
        self.glm = AutoModel.from_pretrained(
            glm_ckpt,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        # LoRA 설정
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query_key_value"],
        )
        self.glm = get_peft_model(self.glm, lora_cfg)

        # 분류 헤드를 device로 이동
        self.cls_head = nn.Linear(4096, 1).to(device)

    def forward(
        self,
        data: Data,       # PyG 그래프 (GPU로 .to(device) 해둘 것)
        person: str,      # 인물 이름
        label: torch.Tensor | None = None,
    ):
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
        inputs_embeds = torch.cat([
            prompt_emb,
            graph_tokens.to(device)
        ], dim=1)  # [1, L+N, 4096]
        attn_mask = torch.ones(
            inputs_embeds.size()[:-1], dtype=torch.long, device=device
        )

        # 5) LLM forward: ChatGLM requires input_ids
        batch_size, seq_len, _ = inputs_embeds.size()
        dummy_input_ids = torch.zeros(
            (batch_size, seq_len), dtype=torch.long, device=device
        )
        out = self.glm(
            input_ids=dummy_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )

        # prefix: prompt 뒤부터 graph tokens 영역
        graph_start_idx = prompt_emb.size(1)
        graph_repr = out.hidden_states[-1][:, graph_start_idx:, :].mean(dim=1)   # [1,4096]

        # 6) 분류 헤드 및 loss
        logits = self.cls_head(graph_repr).squeeze(-1)  # [1]
        if label is None:
            return logits, None

        # label이 [batch,1]인 경우 squeeze
        target = label.to(device).squeeze(-1)
        loss = nn.BCEWithLogitsLoss()(logits, target)
        return logits, loss
