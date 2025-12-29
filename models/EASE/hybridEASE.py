import pandas as pd
import numpy as np
import torch
import gc
import os
import argparse
from sklearn.preprocessing import LabelEncoder

### MANUAL of hybridEASE.py
### put text into your terminal
# python hybridEASE.py --ITEM_BEST 660 --USER_BEST 4573

parser = argparse.ArgumentParser()
parser.add_argument('--ITEM_BEST', type=int)
parser.add_argument('--USER_BEST', type=int)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ITEM_BEST_L2 = args.ITEM_BEST
USER_BEST_L2 = args.USER_BEST

def force_clear_gpu():
    torch.cuda.empty_cache()
    gc.collect()

df = pd.read_csv("../data/train/train_ratings.csv")
df = df.drop('time', axis=1)

user_enc, item_enc = LabelEncoder(), LabelEncoder()
df['u_idx'] = user_enc.fit_transform(df['user'])
df['i_idx'] = item_enc.fit_transform(df['item'])

n_users = len(user_enc.classes_)
n_items = len(item_enc.classes_)
user_ids = user_enc.classes_
item_ids = item_enc.classes_


rows = torch.from_numpy(df['u_idx'].values)
cols = torch.from_numpy(df['i_idx'].values)
X = torch.zeros((n_users, n_items), dtype=torch.float32)
X[rows, cols] = 1.0
X_gpu = X.to(device)

G_item = X_gpu.t() @ X_gpu
G_item[torch.arange(n_items), torch.arange(n_items)] += ITEM_BEST_L2
P_item = torch.linalg.inv(G_item)
B_item = P_item / (-torch.diag(P_item))
B_item.diagonal().fill_(0)

scores_item = X_gpu @ B_item

del G_item, P_item, B_item
force_clear_gpu()

G_user = X_gpu @ X_gpu.t()
G_user[torch.arange(n_users), torch.arange(n_users)] += USER_BEST_L2
P_user = torch.linalg.inv(G_user)
B_user = P_user / (-torch.diag(P_user)).view(-1, 1)
B_user.diagonal().fill_(0)

scores_user = B_user @ X_gpu

del G_user, P_user, B_user
force_clear_gpu()

# ensemble_and_save_submission
def ensemble_and_save_submission(final_scores, filename):
    print(f"Saving {filename}...")
    final_scores[X_gpu > 0] = -1e9
    
    _, topk = torch.topk(final_scores, k=10, dim=1)
    topk_np = topk.cpu().numpy()
    
    rec_list = []
    for u_idx in range(n_users):
        u_id = user_ids[u_idx]
        for i in range(10):
            i_id = item_ids[topk_np[u_idx, i]]
            rec_list.append([u_id, i_id])
            
    pd.DataFrame(rec_list, columns=['user', 'item']).to_csv(filename, index=False)
    print(f"Done: {filename}")

# itme-base, user-based, iu55-based, iu73-based, iu37based
ensemble_and_save_submission(scores_item.clone(), "submission_item_only.csv")
ensemble_and_save_submission(scores_user.clone(), "submission_user_only.csv")
ensemble_and_save_submission(0.5 * scores_item + 0.5 * scores_user, "submission_ensemble_55.csv")
ensemble_and_save_submission(0.7 * scores_item + 0.3 * scores_user, "submission_ensemble_73.csv")
ensemble_and_save_submission(0.3 * scores_item + 0.7 * scores_user, "submission_ensemble_37.csv")

print("\nAll submission.csv created successfully!")