import torch
from data import DataLoader_Graph as MyDataLoader
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGLoader
from tqdm import tqdm
from model import MANAGER, MyModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import bitsandbytes as bnb

annotation_path = "data/data_annotation_balanced_train.csv" #"data_annotation_label.csv"
batch_size = 1
shuffle = True                                                      
learning_rate = 0.001
epochs = 10

def train_chatglm(model, loader, epochs, optimizer, device, file_name):
    # for p in model.chatglm.parameters():
    #     p.requires_grad = False
    # model.chatglm.eval()
    model.train()
    loss_history = []
    for epoch in tqdm(range(epochs),desc="Training"):
        total_loss = 0
        for step, (emb, edge_index, question, answer) in enumerate(loader):
            query = "Does the Graph Context affect the answer to the following question?\n\n"+question[0]
            loss = model.compute_loss(
                query=query,
                answer=answer[0],
                src=emb[0].to(device),
                edge_index=edge_index[0].to(device)
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step} - Loss: {loss.item():.4f}")
                loss_history.append(loss.item())

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} Completed - Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), file_name)
        # loss_history.append(avg_loss)
    
    return loss_history
    
def train_batch(model:MyModel, loader:DataLoader, epochs, optimizer, criterion, device):
    
    loss_history = []
    for epoch in tqdm(range(epochs),desc="Training"):
        total_loss = 0
        model.train()
        for n, (t_emb,a_emb,v_emb,label) in enumerate(loader):
            t_emb,a_emb,v_emb,label = t_emb.to(device), a_emb.to(device), v_emb.to(device), label.to(device)
            # print(t_emb.shape, a_emb.shape, v_emb.shape, q_emb.shape, label.shape)
            optimizer.zero_grad()
            out = model(a_emb, v_emb, t_emb) #batch.edge_index,batch.batch, batch.question)
            loss = criterion(out.squeeze(), label.squeeze().float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader.dataset)
        loss_history.append(avg_loss)

        print(f"{epoch+1}/{epochs} Avg. Loss: {avg_loss}")

        model.eval()
        with torch.no_grad():
            all_labels = []
            all_preds = []
            for n, (t_emb,a_emb,v_emb,label) in enumerate(loader):
                t_emb,a_emb,v_emb,label = t_emb.to(device), a_emb.to(device), v_emb.to(device), label.to(device)
                out = model(a_emb, v_emb, t_emb)
                preds = torch.sigmoid(out).squeeze().cpu().numpy()
                preds = [1 if preds > 0.5 else 0]
                all_labels.extend(label.cpu().numpy())
                all_preds.extend(preds)

            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)

            print(f"Accuracy: {acc}, F1: {f1}, Precision: {precision}, Recall: {recall}")

        torch.save(model.state_dict(), "model.pth")
    
    return loss_history

if __name__ == "__main__":
    # Example usage
    input_dim = 768
    hidden_dim1 =768
    hidden_dim2 = 768
    output_dim = 1
    num_encoder_layers = 24
    dropout_p = 0.25

    pos_weight = torch.tensor(0.192)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = MyModel(input_dim, hidden_dim1, hidden_dim2).to(device)   #.half()

    # # import time
    # # time.sleep(30)

    # # exit()

    # myData = MyDataLoader(annotation_path)
    # loader = DataLoader(myData, batch_size=batch_size, shuffle=True)
    
    # # data = myData() # Data object(x=emb, edge_index=graph, y=label)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

    # avg_loss = train_batch(model, loader, epochs, optimizer, criterion, device)
    # print("Average Loss:", avg_loss)

    # torch.save(model.state_dict(), "model.pth")
    
    # "meta-llama/Llama-3.2-1B" 2048
    # "Qwen/Qwen2.5-1.5B" 1536
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    file_name = "manager_qwen1.5B_Instruct.pth"
    model = MANAGER(model_name, 768,768,1536,device)
    
    dataset = MyDataLoader(annotation_path)
    dataloader = DataLoader(dataset,1,False)
    
    llm_params = list(model.chatglm.parameters())
    gcn_params = list(model.conv1.parameters()) + list(model.conv2.parameters()) +list(model.linear.parameters())

    # optimizer = torch.optim.AdamW([
    #     {"params": gcn_params, "lr": 1e-4},
    # ], weight_decay=0.01)
    
    optimizer = bnb.optim.AdamW8bit([
        {"params": gcn_params, "lr": 1e-4},
        {"params": llm_params, "lr": 2e-5},
    ], weight_decay=0.01)
    
    avg_loss = train_chatglm(model,dataloader,3,optimizer,device,file_name)
    
    print("Average Loss:", avg_loss)
    
