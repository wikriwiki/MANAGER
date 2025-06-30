import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
from peft import get_peft_model, LoraConfig, TaskType
import os
import time
import math
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from dotenv import load_dotenv
load_dotenv()

# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# model = AutoModel.from_pretrained('bert-base-uncased')
# model.eval()
# model.to('cuda')
access_token = os.getenv("LLM_token")

def get_chain_length(edge_index):
    sources = edge_index[0]
    targets = edge_index[1]
    
    length = 1  # 첫 엣지는 무조건 포함됨
    
    for i in range(1, edge_index.size(1)):
        if sources[i] == targets[i-1]:
            length += 1
        else:
            break  # 사슬이 끊기면 종료

    return length

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:,:x.size(0), :]

class CrossModalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CrossModalAttention, self).__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=2, batch_first=True)
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x1, x2): # x1 = alpha->beta  , x2 = beta
        x1_tilde = self.norm(x1)
        x2_tilde = self.norm(x2)
        attn_output, _ = self.attention(x1_tilde, x2_tilde, x2_tilde)
        output = self.fc(attn_output)
        output = output + self.norm(output)
        return output

class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(MyModel, self).__init__()
        
        self.audio_conv = nn.Conv1d(input_dim, hidden_dim1, kernel_size=3, padding=1)
        self.video_conv = nn.Conv1d(input_dim, hidden_dim1, kernel_size=3, padding=1)
        self.text_conv = nn.Conv1d(input_dim, hidden_dim1, kernel_size=3, padding=1)

        self.audio_projection = nn.Linear(hidden_dim1, hidden_dim2)
        self.video_projection = nn.Linear(hidden_dim1, hidden_dim2)
        self.text_projection = nn.Linear(hidden_dim1, hidden_dim2)
        self.question_projection = nn.Linear(hidden_dim1, hidden_dim2)

        self.pos_encoder = PositionalEncoding(hidden_dim1)
        self.text_modal_token = nn.Parameter(torch.randn(1, hidden_dim2))  # shape: [1, hidden_dim2]
        self.audio_modal_token = nn.Parameter(torch.randn(1, hidden_dim2))
        self.video_modal_token = nn.Parameter(torch.randn(1, hidden_dim2))
        self.question_modal_token = nn.Parameter(torch.randn(1, hidden_dim2))

        cls_token_id = tokenizer.cls_token_id
        cls_token = model.embeddings.word_embeddings(torch.tensor(cls_token_id).to('cuda'))  # shape: [1, hidden_dim]
        print(cls_token.shape)
        self.cls_token = nn.Parameter(cls_token.unsqueeze(0))  # shape: [1, hidden_dim]

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim1, dropout=0.2,nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=8)

        self.fc = nn.Linear(hidden_dim1, 1)

    def forward(self, a:torch.Tensor, v, t, q): # audio, video, text, question
        # z_a = self.audio_conv(a.permute(0, 2, 1)).permute(0, 2, 1)
        # z_a = nn.ReLU()(z_a)
        # z_v = self.video_conv(v.permute(0, 2, 1)).permute(0, 2, 1)
        # z_v = nn.ReLU()(z_v)
        # z_t = self.text_conv(t.permute(0, 2, 1)).permute(0, 2, 1)
        # z_t = nn.ReLU()(z_t)
        # z_q = self.text_conv(q.permute(0, 2, 1)).permute(0, 2, 1)
        # z_q = nn.ReLU()(z_q)

        z_a = self.audio_projection(a)
        z_a = nn.ReLU()(z_a)
        z_v = self.video_projection(v)
        z_v = nn.ReLU()(z_v)
        z_t = self.text_projection(t)
        z_t = nn.ReLU()(z_t)
        z_q = self.question_projection(q)
        z_q = nn.ReLU()(z_q)

        z_a = self.pos_encoder(z_a)
        z_v = self.pos_encoder(z_v)
        z_t = self.pos_encoder(z_t)
        z_q = self.pos_encoder(z_q)

        z_a = z_a + self.audio_modal_token
        z_v = z_v + self.video_modal_token
        z_t = z_t + self.text_modal_token
        z_q = z_q + self.question_modal_token

        z = torch.cat((z_q, z_t, z_a, z_v), dim=1)  # Concatenate along the sequence dimension

        z = torch.cat((self.cls_token.expand(z.size(0), -1, -1), z), dim=1)  # Add cls token
        z = self.encoder(z)  # Apply transformer encoder
        z = z[:, 0, :]  # Take the cls token output
        z = self.fc(z)  # Apply final linear layer
        # z = torch.sigmoid(z)  # Apply sigmoid activation
        return z


    
class Monopoly(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3):
        super(Monopoly, self).__init__()

        self.audio_conv = nn.Conv1d(input_dim, hidden_dim1, kernel_size=3, padding=1)
        self.video_conv = nn.Conv1d(input_dim, hidden_dim1, kernel_size=3, padding=1)
        self.text_conv = nn.Conv1d(input_dim, hidden_dim1, kernel_size=3, padding=1)

        self.pos_encoder = PositionalEncoding(hidden_dim1)
        
        self.v_a_attention = CrossModalAttention(hidden_dim1, hidden_dim2)
        self.t_a_attention = CrossModalAttention(hidden_dim1, hidden_dim2)
        self.v_t_attention = CrossModalAttention(hidden_dim1, hidden_dim2)
        self.a_t_attention = CrossModalAttention(hidden_dim1, hidden_dim2)
        self.a_v_attention = CrossModalAttention(hidden_dim1, hidden_dim2)
        self.t_v_attention = CrossModalAttention(hidden_dim1, hidden_dim2)

        self.audio_transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim2*2, nhead=2, batch_first=True)
        self.video_transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim2*2, nhead=2, batch_first=True)
        self.text_transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim2*2, nhead=2, batch_first=True)

        self.audio_transformer = nn.TransformerEncoder(self.audio_transformer_layer, num_layers=2)
        self.video_transformer = nn.TransformerEncoder(self.video_transformer_layer, num_layers=2)
        self.text_transformer = nn.TransformerEncoder(self.text_transformer_layer, num_layers=2)

        self.fc_a = nn.Linear(hidden_dim2*2, hidden_dim3)
        self.fc_v = nn.Linear(hidden_dim2*2, hidden_dim3)
        self.fc_t = nn.Linear(hidden_dim2*2, hidden_dim3)
        self.softmax = nn.Softmax(dim=hidden_dim3)

        self.output = nn.Linear(hidden_dim3*3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, a, v, t):
        XA = self.audio_conv(a.permute(0, 2, 1)).permute(0, 2, 1)
        XV = self.video_conv(v.permute(0, 2, 1)).permute(0, 2, 1)
        XT = self.text_conv(t.permute(0, 2, 1)).permute(0, 2, 1)

        XAPE = self.pos_encoder(XA)
        XVPE = self.pos_encoder(XV)
        XTPE = self.pos_encoder(XT) # => [B, S, hidden_dim1]

        x1 = self.v_a_attention(XAPE, XVPE)
        x2 = self.v_t_attention(XTPE, XVPE)
        x3 = self.t_a_attention(XAPE, XTPE)
        x4 = self.a_v_attention(XVPE, XAPE)
        x5 = self.a_t_attention(XTPE, XAPE)
        x6 = self.t_v_attention(XVPE, XTPE)

        ZA = torch.cat((x1,x3), dim=2)
        ZT = torch.cat((x2,x5), dim=2)
        ZV = torch.cat((x4,x6), dim=2)

        ZA = self.audio_transformer(ZA)
        ZV = self.video_transformer(ZV)
        ZT = self.text_transformer(ZT)

        WA_tilde = F.softmax(self.fc_a(ZA))
        WV_tilde = F.softmax(self.fc_v(ZV))
        WT_tilde = F.softmax(self.fc_t(ZT))

        Wall_tilde = WA_tilde + WV_tilde + WT_tilde

        WA = WA_tilde / Wall_tilde
        WV = WV_tilde / Wall_tilde
        WT = WT_tilde / Wall_tilde

        Zfused = WA*ZA + WV*ZV + WT*ZT

        out = self.output(Zfused)
        out = self.sigmoid(out)

        return out



class MANAGER(nn.Module):
    def __init__(self, model_name, input_dim, hidden_dim1, hidden_dim2,device):
        """
        Args:
            input_dim (int): Input dimension.
            hidden_dim1 (int): First hidden dimension.
            hidden_dim2 (int): Second hidden dimension.
            output_dim (int): Output dimension.
            num_encoder_layers (int): Number of encoder layers in the transformer.
            num_decoder_layers (int): Number of decoder layers in the transformer.
            dropout_p (float): Dropout probability.
        """
        super(MANAGER, self).__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim1).to(device)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim1).to(device)
        
        self.linear = nn.Linear(hidden_dim1, hidden_dim2).to('cuda')

        # ChatGLM2 with LoRA
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4"
        # )
                
        self.chatglm = AutoModelForCausalLM.from_pretrained(
            model_name,
            # quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
            use_auth_token=access_token
        )
        self.chatglm.gradient_checkpointing_enable()
        
        # self.chatglm.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,use_auth_token=access_token)
        self.tokenizer.model_max_length = 8192
        
        # lora_config = LoraConfig(
        #     r=8,
        #     lora_alpha=16,
        #     target_modules=["query_key_value"],
        #     lora_dropout=0.05,
        #     task_type=TaskType.CAUSAL_LM
        # )

        # self.chatglm = get_peft_model(base_model, lora_config)
        # self.chatglm.print_trainable_parameters()  # 확인용
        self.device=device
        
    def chatglm_generate(self, query, gnn_embed):
        max_length = 8192
        num_beams = 1
        do_sample = True
        top_p = 0.8
        temperature = 0.8
        
        # 1. GNN 출력 → 의미 있는 프롬프트 텍스트로 변환
        gnn_vec = gnn_embed.squeeze(0).tolist()
        gnn_snippet = ", ".join([f"{v:.2f}" for v in gnn_vec]) 

        gnn_context = f"[Graph Context] : {gnn_snippet}"
        
        # 2. 프롬프트 구성: GNN → 사용자 질문
        full_prompt = gnn_context + f"\n\n[User]: {query}"  
        
        print(full_prompt)
        
        # 3. 토크나이즈 및 generate
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # print(input_ids)
        
        outputs = self.chatglm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            do_sample=do_sample
        )

        response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response

    def forward(self,src,edge,query):
        utt_len = get_chain_length(edge)

        # 1. GCN forward
        x = self.conv1(src, edge)
        x = F.relu(x)
        x = self.conv2(x, edge)
        x = F.relu(x)
        x = x[:utt_len]
        x = self.linear(x)

        # 1. 임베딩
        prompt = f"\n\n[User]: According the context, answer the question '{query}'\n\n[Answer]:"
        gnn_embed = x.unsqueeze(0)
        text_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        text_embeds = self.chatglm.get_input_embeddings()(text_inputs["input_ids"])
        input_embeds = torch.cat([gnn_embed, text_embeds], dim=1)
        
        # GCN 임베딩은 전부 실제 토큰으로 처리 (mask=1)
        gnn_mask = torch.ones((1, gnn_embed.shape[1]), dtype=torch.long).to(self.device)

        # 기존 텍스트의 attention_mask
        text_mask = text_inputs["attention_mask"]  # shape: [1, text_len]

        # 이어붙이기
        attention_mask = torch.cat([gnn_mask, text_mask], dim=1)

        # 5. Output
        outputs = self.chatglm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=128,  # 생성 최대 길이
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=0.1
        )

        response = self.tokenizer.decode(outputs[0][:], skip_special_tokens=True) # input_embeds.shape[1]
        return response
    
    def compute_loss(self, query: str, answer: str, src: torch.Tensor, edge_index: torch.Tensor):
        # 1. GCN forward
        utt_len = get_chain_length(edge_index)  # 선택적
        x = self.conv1(src, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = x[:utt_len]
        x = self.linear(x)
        
        # 정답 토큰
        answer_tokens = self.tokenizer(answer, return_tensors="pt").input_ids.to(self.device)
        eos_token = torch.tensor([[self.tokenizer.eos_token_id]], device=self.device)
        answer_tokens = torch.cat([answer_tokens, eos_token], dim=1)  # shape: [1, T+1]
        
        answer_embeds = self.chatglm.get_input_embeddings()(answer_tokens)

        # 1. 임베딩
        prompt = f"\n\n[User]: According the context, answer the question '{query}'\n\n[Answer]:"
        gnn_embed = x.unsqueeze(0)
        text_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        text_embeds = self.chatglm.get_input_embeddings()(text_inputs["input_ids"])
        input_embeds = torch.cat([gnn_embed, text_embeds,answer_embeds], dim=1)
        
        # GCN 임베딩은 전부 실제 토큰으로 처리 (mask=1)
        gnn_mask = torch.ones((1, gnn_embed.shape[1]), dtype=torch.long).to(self.device)

        # 기존 텍스트의 attention_mask
        text_mask = text_inputs["attention_mask"]  # shape: [1, text_len]
        
        answer_mask = torch.zeros((1, answer_embeds.shape[1]), dtype=torch.long).to(self.device)

        # 이어붙이기
        attention_mask = torch.cat([gnn_mask, text_mask, answer_mask], dim=1)
        

        # label = [-100 for GCN + 질문], 정답 토큰만 label로 학습
        label_pad_len = gnn_embed.shape[1] + text_inputs["input_ids"].shape[1]
        label_pad = torch.full((1, label_pad_len), -100, dtype=torch.long).to(self.device)
        labels = torch.cat([label_pad, answer_tokens], dim=1)

        # 5. loss 계산
        output = self.chatglm(
            # input_ids=input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
#         logits = output.logits  # shape: [batch, seq_len, vocab_size]
        
#         loss = F.cross_entropy(
#         logits.view(-1, logits.size(-1)),   # [batch * seq_len, vocab]
#         labels.view(-1),                    # [batch * seq_len]
#         ignore_index=-100                   # 앞부분 무시
# )

        return output.loss

    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz, device = torch.device('cpu'))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
if __name__ == "__main__":
    # Example usage
    input_dim = 768
    hidden_dim1 = 768
    hidden_dim2 = 2048
    output_dim = 1
    num_encoder_layers = 2
    dropout_p = 0.1

    model = MANAGER("meta-llama/Llama-3.2-1B",input_dim, hidden_dim1, hidden_dim2,'cuda')
    
    # Create a dummy input tensor
    src = torch.load('data/embedding/emb_0001.pt').to('cuda')

    graph = torch.load('data/graph/graph_0001.pt', weights_only=False).to('cuda')
    
    # Forward pass
    output = model(src,graph.edge_index,"Will Pete Hegseth be Trump's Defense Secretary?")
    
    print(output)

    # print(output.shape)
