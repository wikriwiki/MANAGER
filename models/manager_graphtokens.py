from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from models.gcn import RGCNEncoder
from models.graph_builder import EDGE_TYPE

class GraphTokenManager(nn.Module):
    def __init__(
        self,
        glm_ckpt: str = "THUDM/chatglm2-6b",
        gcn_layers: int = 5,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.glm = AutoModel.from_pretrained(
            glm_ckpt,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        self.base_device = self.glm.get_input_embeddings().weight.device
        self.tokenizer = AutoTokenizer.from_pretrained(
            glm_ckpt, trust_remote_code=True
        )
        self.gcn = RGCNEncoder(
            in_dim=768,
            num_rel=len(EDGE_TYPE),
            num_layers=gcn_layers,
        ).to(self.base_device)
        self.proj_up  = nn.Linear(768, 4096, bias=False).to(self.base_device)
        self.cls_head = nn.Linear(4096, 1).to(self.base_device)
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query_key_value"],
        )
        self.glm = get_peft_model(self.glm, lora_cfg)

    def forward(
        self,
        data: Data,
        person: str,
        label: torch.Tensor | None = None,
    ):
        device = self.base_device
        data = data.to(device)

        # 1) R-GCN을 통해 모든 노드 임베딩을 업데이트
        node_h = self.gcn(data.x, data.edge_index, data.edge_type)
        
        # 2) GraphBuilder가 생성한 '텍스트 토큰 노드(발화 노드)'만 추출
        # GraphBuilder의 build() 함수에서 `text_token_map`에 해당하는 노드 인덱스를
        # data 객체에 추가했다고 가정 (e.g., data.utterance_nodes)
        
        # `data.node_meta`를 사용하여 발화 노드 인덱스 추출
        # GraphBuilder 코드에 따르면 발화 텍스트 토큰이 가장 먼저 추가됨
        num_utterance_tokens = data.node_meta.get("text_tokens", 0)
        
        if num_utterance_tokens > 0:
            utterance_h = node_h[:num_utterance_tokens]
        else:
            # 발화 토큰이 없는 경우 (예외 처리)
            # 여기서는 빈 텐서를 반환하거나 오류를 발생시킬 수 있음
            utterance_h = torch.empty(0, 768, device=device)

        # 3) 추출된 발화 노드 임베딩을 가상 토큰으로 변환
        graph_tokens = self.proj_up(utterance_h).unsqueeze(0)  # [1, M, 4096]
        
        # 4) prompt 임베딩
        prompt = (
            "Instruction: Please predict whether Google search volume for "
            f"{person} will exhibit an anomaly (0=no, 1=yes) within 3 days.\n"
            "Input: "
        )
        tok = self.tokenizer(prompt, return_tensors="pt").to(device)
        prompt_emb = self.glm.get_input_embeddings()(tok.input_ids)

        # 5) prompt 임베딩과 발화 노드 가상 토큰을 결합
        inputs_embeds = torch.cat([prompt_emb, graph_tokens], dim=1)
        attn_mask = torch.ones(inputs_embeds.size()[:-1],
                               dtype=torch.long, device=device)

        # 6) GLM forward
        b, s, _ = inputs_embeds.size()
        dummy_ids = torch.zeros((b, s), dtype=torch.long, device=device)
        out = self.glm(
            input_ids=dummy_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )

        # 7) GLM 출력에서 가상 토큰에 해당하는 부분만 추출 후 분류
        g_start = prompt_emb.size(1)
        graph_repr = out.hidden_states[-1][:, g_start:, :].mean(dim=1)
        
        logits = self.cls_head(graph_repr).squeeze(-1)
        if label is None:
            return logits, None

        target = label.to(device).squeeze(-1)
        loss = nn.BCEWithLogitsLoss()(logits, target)
        return logits, loss