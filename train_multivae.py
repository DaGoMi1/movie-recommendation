import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import sparse
import torch.nn.functional as F
import os
import pandas as pd
import pickle

# 모델 정의
class MultiVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=600, latent_dim=200, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.drop = nn.Dropout(dropout)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        h = F.normalize(x, p=2, dim=1)
        h = self.drop(h)
        h = self.encoder(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

class DeepMultiVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024, 512], latent_dim=256, dropout=0.5):
        super(DeepMultiVAE, self).__init__()
        
        # Encoder: 더 깊은 계층 구조
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.Tanh())
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder: Encoder의 역순
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.Tanh())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.drop = nn.Dropout(dropout)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        h = F.normalize(x, p=2, dim=1)
        h = self.drop(h)
        h = self.encoder(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# Loss 함수
def loss_fn(recon_x, x, mu, logvar, anneal=1.0):
    log_softmax_var = torch.log_softmax(recon_x, dim=1)
    neg_ll = -torch.mean(torch.sum(log_softmax_var * x, dim=1))
    kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return neg_ll + anneal * kld

# Recall 계산
def get_recall(recon_batch, test_matrix, k=100):
    indices = np.argpartition(-recon_batch, k, axis=1)[:, :k]
    
    recall_list = []
    
    for i in range(len(indices)):
        # 해당 유저의 실제 정답 아이템 인덱스
        actual = test_matrix[i].indices
        if len(actual) == 0: continue # 정답이 없으면 계산 제외
        
        # 상위 k개 인덱스 중 정답과 겹치는 개수
        hits = len(set(indices[i]) & set(actual))
        
        # 유저별 Recall = 맞춘 아이템 수 / 해당 유저의 전체 정답 수
        recall_list.append(hits / len(actual))
        
    return np.mean(recall_list) if len(recall_list) > 0 else 0

def split_data(matrix):
    train_rows = []
    train_cols = []
    val_rows = []
    val_cols = []
    
    matrix = matrix.tocsr() # 행 단위 접근을 위해 변환
    for i in range(matrix.shape[0]):
        items = matrix.getrow(i).indices
        if len(items) > 1:
            # 마지막 아이템은 검증용, 나머지는 학습용
            val_rows.append(i)
            val_cols.append(items[-1])
            
            for item in items[:-1]:
                train_rows.append(i)
                train_cols.append(item)
        else:
            # 아이템이 하나뿐이면 그냥 학습용에 넣음
            train_rows.append(i)
            train_cols.append(items[0])
            
    train_matrix = sparse.csr_matrix(([1]*len(train_rows), (train_rows, train_cols)), shape=matrix.shape)
    val_matrix = sparse.csr_matrix(([1]*len(val_rows), (val_rows, val_cols)), shape=matrix.shape)
    
    return train_matrix, val_matrix

def run_train(model, train_matrix, val_matrix, epochs=300, batch_size=512, device='cuda', patience=20):
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    model.to(device)
    num_users = train_matrix.shape[0]
    
    best_recall = 0
    counter = 0  # Early Stopping을 위한 카운터
    
    # 검증 시 메모리 부족 방지를 위해 미리 dense 변환을 피함
    val_true = val_matrix.toarray()
    train_dense_for_val = train_matrix.toarray()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        idxlist = np.arange(num_users)
        np.random.shuffle(idxlist)
        
        for batch_idx in range(0, num_users, batch_size):
            end_idx = min(batch_idx + batch_size, num_users)
            batch_users = idxlist[batch_idx:end_idx]
            x = torch.FloatTensor(train_matrix[batch_users].toarray()).to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(x)
            # Annealing을 조금 더 천천히 진행 (200 에포크까지 0.2에 도달하도록 수정 가능)
            anneal = min(0.15, epoch / 100) 
            loss = loss_fn(recon_batch, x, mu, logvar, anneal)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
            
        # 검증 단계
        model.eval()
        with torch.no_grad():
            # 유저가 많을 경우 검증도 배치 단위로 나누어 진행하는 것이 안전합니다.
            recon_val_list = []
            for i in range(0, num_users, batch_size):
                end_i = min(i + batch_size, num_users)
                batch_val_x = torch.FloatTensor(train_dense_for_val[i:end_i]).to(device)
                recon_batch, _, _ = model(batch_val_x)
                
                recon_batch = recon_batch.cpu().numpy()
                # 이미 본 아이템 제외 (Masking)
                recon_batch[train_dense_for_val[i:end_i] > 0] = -1e9
                recon_val_list.append(recon_batch)
            
            recon_val = np.concatenate(recon_val_list, axis=0)
            current_recall_10 = get_recall(recon_val, val_matrix.tocsr(), k=10)
            current_recall_100 = get_recall(recon_val, val_matrix.tocsr(), k=100)
            
        print(f"Epoch {epoch+1:3d}, Loss: {total_loss / (num_users/batch_size):.4f}, "
              f"Recall@10: {current_recall_10:.4f}, Recall@100: {current_recall_100:.4f}, LR: {current_lr:.6f}")

        # Recall@100 기준으로 Early Stopping 및 모델 저장
        if current_recall_100 > best_recall:
            best_recall = current_recall_100
            torch.save(model.state_dict(), './output/multivae_best_model.pt')
            print(f"   ==> Best Recall@100 Updated! Saved Model.")
            counter = 0 
        else:
            counter += 1
            if counter >= patience:
                print(f"Early Stopping triggered. No improvement in Recall@100 for {patience} epochs.")
                break

# 추론 및 제출 파일 생성 함수 (단일 Multi-VAE용)
def run_inference(model, train_matrix, mapping, device, batch_size=512, top_k=10):
    model.eval()
    num_users = train_matrix.shape[0]
    id2user = {v: k for k, v in mapping['user2id'].items()}
    id2item = mapping['id2item']
    
    print("Inference for Submission...")
    pred_list = []
    
    with torch.no_grad():
        for i in range(0, num_users, batch_size):
            end_idx = min(i + batch_size, num_users)
            batch_x = torch.FloatTensor(train_matrix[i:end_idx].toarray()).to(device)
            
            recon_batch, _, _ = model(batch_x)
            
            # 이미 본 아이템 제외
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[train_matrix[i:end_idx].toarray() > 0] = -1e9
            
            # 상위 K개 추출 (정렬 포함)
            ind = np.argpartition(recon_batch, -top_k, axis=1)[:, -top_k:]
            arr_ind = recon_batch[np.arange(len(recon_batch))[:, None], ind]
            arr_ind_argsort = np.argsort(arr_ind, axis=1)[:, ::-1]
            batch_pred_list = ind[np.arange(len(recon_batch))[:, None], arr_ind_argsort]
            
            pred_list.append(batch_pred_list)
            
    pred_list = np.concatenate(pred_list, axis=0)
    
    # 결과 포맷팅
    submission_data = []
    for user_idx, item_indices in enumerate(pred_list):
        original_user_id = id2user[user_idx]
        for item_id in item_indices:
            original_item_id = id2item[item_id]
            submission_data.append([original_user_id, original_item_id])
            
    df_sub = pd.DataFrame(submission_data, columns=['user', 'item'])
    df_sub.to_csv('./output/submission.csv', index=False)
    print("Submission File Generated at ./output/submission.csv")

# 추론 및 제출 파일 생성 함수 (Multi-VAE + CatBoost용)
def run_inference_for_catboost(model, train_matrix, mapping, device, batch_size=512, top_k=100):
    model.eval()
    num_users = train_matrix.shape[0]
    id2user = {v: k for k, v in mapping['user2id'].items()}
    id2item = mapping['id2item']
    
    print(f"Generating Top-{top_k} Candidates for CatBoost...")
    
    results = [] # user, item, score를 담을 리스트

    with torch.no_grad():
        for i in range(0, num_users, batch_size):
            end_idx = min(i + batch_size, num_users)
            batch_x = torch.FloatTensor(train_matrix[i:end_idx].toarray()).to(device)
            
            recon_batch, _, _ = model(batch_x)
            
            recon_batch = recon_batch.cpu().numpy()
            # recon_batch[train_matrix[i:end_idx].toarray() > 0] = -1e9
            
            # 상위 K개(100개) 추출
            ind = np.argpartition(recon_batch, -top_k, axis=1)[:, -top_k:]
            
            # 추출된 인덱스들의 실제 VAE score(확률값) 가져오기
            rows = np.arange(len(recon_batch))[:, None]
            scores = recon_batch[rows, ind]
            
            # 각 유저별로 결과 저장
            for b_idx in range(len(recon_batch)):
                user_idx = i + b_idx
                original_user_id = id2user[user_idx]
                
                for k_idx in range(top_k):
                    item_idx = ind[b_idx, k_idx]
                    score = scores[b_idx, k_idx]
                    
                    original_item_id = id2item[item_idx]
                    results.append([original_user_id, original_item_id, score])
            
    # 데이터프레임 생성 및 저장
    df_candidates = pd.DataFrame(results, columns=['user', 'item', 'vae_score'])
    df_candidates.to_csv('./output/vae_candidates_top100.csv', index=False)
    print(f"Candidate file saved! Total rows: {len(df_candidates)}")
    return df_candidates

# 메인 함수
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 로드
    full_matrix = sparse.load_npz('../data/train/train_matrix.npz')
    with open('../data/train/mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)
    
    # 데이터 분리
    print("Splitting data into Train/Val...")
    train_matrix, val_matrix = split_data(full_matrix)
    
    num_items = train_matrix.shape[1]
    model = MultiVAE(input_dim=num_items, hidden_dim=800, latent_dim=300, dropout=0.2).to(device)
    # model = DeepMultiVAE(input_dim=num_items, hidden_dims=[1024, 512, 256], latent_dim=200, dropout=0.3).to(device)

    # 학습 실행 (검증 데이터 포함)
    run_train(model, train_matrix, val_matrix, epochs=300, batch_size=128, device=device, patience=30)
    
    # 추론 실행 (Best 모델 로드 후 제출용은 전체 데이터 다시 사용 권장)
    print("Loading best model for inference...")
    model.load_state_dict(torch.load('./output/multivae_best_model.pt'))
    # run_inference(model, full_matrix, mapping, device)
    run_inference_for_catboost(model, full_matrix, mapping, device)

if __name__ == "__main__":
    main()