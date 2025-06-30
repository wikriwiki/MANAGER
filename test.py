import torch
from data import DataLoader_Graph as MyDataLoader
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGLoader
from tqdm import tqdm
from model import MANAGER, MyModel
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_file = "manager_qwen1.5B_Instruct.pth"
annotation_path = "data/data_annotation_balanced_test.csv"

model = MANAGER("Qwen/Qwen2.5-1.5B-Instruct", 768,768,1536,device)

model.load_state_dict(torch.load(model_file))

dataset = MyDataLoader(annotation_path)
dataloader = DataLoader(dataset)

outputs = []
answers = []

for step, (emb, edge_index, question, answer) in tqdm(enumerate(dataloader)):
    query = "Does the Graph Context affect the answer to the following question?\n\n"+question[0]
    output = model(
        query=query,
        src=emb[0].to(device),
        edge=edge_index[0].to(device)
    )
    print(output)
    outputs.append(output)
    answers.append(answer[0])
    
outputs_logits = []
for o in outputs:
    if o == "Low volatility is expected.":
        outputs_logits.append(0)
    elif o == "High volatility is expected.":
        outputs_logits.append(1)
    else:
        outputs_logits.append(2)
        
answers_logits = []
for a in answers:
    if a == "Low volatility is expected.":
        answers_logits.append(0)
    elif a == "High volatility is expected.":
        answers_logits.append(1)
    else:
        answers_logits.append(2)
        
# confusion matrix 계산
cm = confusion_matrix(answers_logits, outputs_logits)

# 시각화
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Low", "High"], yticklabels=["Low", "High"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 추가로 정밀도, 재현율 등도 출력
print(classification_report(answers_logits, outputs_logits, target_names=["Low", "High"]))