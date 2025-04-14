import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from torch.utils.data import DataLoader
from tqdm import tqdm
from deepxml.models import XMLModel
from deepxml.optimizers import DenseSparseAdam
from deepxml.data_utils import get_data
from deepxml.evaluation import get_p_5, get_n_5
from deepxml.dataset import XMLDataset
from deepxml.tree import FastAttentionXML
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
train_texts, raw_train_labels = get_data('data/EUR-Lex/train_texts.npy', 'data/EUR-Lex/train_labels.npy')
test_texts, raw_test_labels = get_data('data/EUR-Lex/test_texts.npy', 'data/EUR-Lex/test_labels.npy')

# Label preprocessing
mlb_path = 'data/EUR-Lex/labels_binarizer.pkl'
if os.path.exists(mlb_path):
    mlb = joblib.load(mlb_path)
else:
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(raw_train_labels)
    joblib.dump(mlb, mlb_path)

train_labels = mlb.transform(raw_train_labels)
test_labels = mlb.transform(raw_test_labels)

# Verify unseen labels
unique_train_labels = set(label for sublist in raw_train_labels for label in sublist)
unique_test_labels = set(label for sublist in raw_test_labels for label in sublist)
unseen_labels = unique_test_labels - unique_train_labels
if unseen_labels:
    print(f"Unseen labels in test set: {len(unseen_labels)} labels")

# Convert texts to binary format
vocab = np.load('data/EUR-Lex/vocab.npy', allow_pickle=True)
emb_init = np.load('data/EUR-Lex/emb_init.npy', allow_pickle=True)
vocab_dict = {word: idx for idx, word in enumerate(vocab)}

# Save emb_init if not already saved
if not os.path.exists('data/EUR-Lex/emb_init.npy'):
    np.save('data/EUR-Lex/emb_init.npy', emb_init)

def convert_to_binary(file_path, vocab):
    binary_texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            indices = [vocab.get(token, 1) for token in tokens]
            if not indices:
                indices = [1]
            binary_texts.append(indices)
    return binary_texts

def truncate_text(texts, max_len=500, padding_idx=0, unknown_idx=1):
    padded_texts = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, x in enumerate(texts):
        if not x:
            padded_texts[i, 0] = unknown_idx
            continue
        length = min(len(x), max_len)
        padded_texts[i, :length] = x[:length]
    return padded_texts

train_texts = convert_to_binary('data/EUR-Lex/train_texts.txt', vocab_dict)
test_texts = convert_to_binary('data/EUR-Lex/test_texts.txt', vocab_dict)
train_texts = truncate_text(train_texts, max_len=500)
test_texts = truncate_text(test_texts, max_len=500)

# Verify shapes
print("train_texts shape:", train_texts.shape)
print("test_texts shape:", test_texts.shape)

# Define model parameters
labels_num = len(mlb.classes_)
emb_size = 300
hidden_size = 128
layers_num = 2
linear_size = [128, 64]
dropout = 0.2
parallel_attn = labels_num <= 80000

# Modified FastAttentionXML
class RobustFastAttentionXML(FastAttentionXML, nn.Module):
    def __init__(self, labels_num, data_cnf, model_cnf, tree_id='', 
                 emb_size=300, vocab_size=None, hidden_size=512, 
                 layers_num=2, linear_size=None, dropout=0.2, 
                 parallel_attn=True, emb_init=None):
        nn.Module.__init__(self)
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layers_num = layers_num
        self.linear_size = linear_size or [128, 64]
        self.dropout = dropout
        self.parallel_attn = parallel_attn

        FastAttentionXML.__init__(
            self,
            labels_num=labels_num,
            data_cnf=data_cnf,
            model_cnf=model_cnf,
            tree_id=tree_id
        )

        if vocab_size is not None:
            if emb_init is not None:
                self.embedding = nn.Embedding.from_pretrained(
                    torch.from_numpy(emb_init).float(), 
                    freeze=False
                )
            else:
                self.embedding = nn.Embedding(vocab_size, self.emb_size)
        else:
            raise ValueError("vocab_size must be provided")
        
        self.fc = nn.Linear(self.emb_size, self.hidden_size)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, 
                           num_layers=self.layers_num, batch_first=True, 
                           dropout=self.dropout if self.layers_num > 1 else 0)
        self.linear_layers = nn.ModuleList([
            nn.Linear(self.hidden_size if i == 0 else self.linear_size[i-1], size)
            for i, size in enumerate(self.linear_size)
        ])
        self.output_layer = nn.Linear(self.linear_size[-1], labels_num)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_layer(x)
        x = self.fc(x)
        x, _ = self.lstm(x)
        for layer in self.linear_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x[:, -1, :])
        return x

# Configuration dictionaries
data_cnf = {
    'name': 'EUR-Lex',
    'embedding': {'emb_init': 'data/EUR-Lex/emb_init.npy'},
    'model': {'emb_size': emb_size},
    'train': {'sparse': 'data/EUR-Lex/train.txt', 'labels': train_labels}
}

model_cnf = {
    'name': 'RobustFastAttentionXML',
    'path': './models',
    'k': 100,
    'top': 100,
    'level': 2,
    'model': {
        'hidden_size': hidden_size,
        'layers_num': layers_num,
        'linear_size': linear_size,
        'dropout': dropout,
        'parallel_attn': parallel_attn
    },
    'train': [
        {'batch_size': 64, 'lr': 1e-4},
        {'batch_size': 64, 'lr': 1e-4}
    ],
    'valid': {'batch_size': 64},
    'predict': {'batch_size': 64, 'k': 100},
    'cluster': {'eps': 0.01, 'max_leaf': 100, 'levels': [0, 1], 'groups_path': './models/clusters'}
}

# Instantiate the model
pr5=4.6782
ndc=5.8567
model = RobustFastAttentionXML(
    labels_num=labels_num,
    data_cnf=data_cnf,
    model_cnf=model_cnf,
    emb_size=emb_size,
    vocab_size=len(vocab_dict),
    hidden_size=hidden_size,
    layers_num=layers_num,
    linear_size=linear_size,
    dropout=dropout,
    parallel_attn=parallel_attn,
    emb_init=emb_init
).to(device)

# Ensure label clustering
cluster_path = model_cnf['cluster']['groups_path']
group_labels_file = os.path.join(cluster_path, 'group_labels.npy')
groups_file = os.path.join(cluster_path, 'groups.npy')

# Clear outdated clusters if they exist
if os.path.exists(cluster_path):
    try:
        temp_labels = np.load(group_labels_file, allow_pickle=True)
        if temp_labels.dtype == object:
            print("Detected outdated object array in group_labels.npy. Regenerating clusters...")
            import shutil
            shutil.rmtree(cluster_path)
    except:
        print("Error reading group_labels.npy. Regenerating clusters...")
        import shutil
        shutil.rmtree(cluster_path)

if not (os.path.exists(group_labels_file) and os.path.exists(groups_file)):
    os.makedirs(cluster_path, exist_ok=True)
    print("Building label tree for clustering...")
    if isinstance(train_labels, csr_matrix):
        # Create group_labels: map each sample's labels to groups
        group_labels = [np.where(row.toarray().flatten() > 0)[0] % 100 for row in train_labels]
        # Pad group_labels to fixed length
        max_groups = max(len(g) for g in group_labels if len(g) > 0) if group_labels else 1
        group_labels_padded = np.full((len(group_labels), max_groups), -1, dtype=np.int32)
        for i, g in enumerate(group_labels):
            if len(g) > 0:
                group_labels_padded[i, :len(g)] = g
        # Create groups: map each group index to its labels
        groups = {i: [] for i in range(100)}
        for sample_idx, sample_labels in enumerate(group_labels):
            for label in sample_labels:
                groups[label % 100].append(label % labels_num)
        # Ensure groups are numpy arrays
        groups = {k: np.array(v, dtype=np.int32) for k, v in groups.items() if v}
        # Save clusters
        np.save(group_labels_file, group_labels_padded)
        np.save(groups_file, groups, allow_pickle=True)
        # Use padded array
        group_labels = group_labels_padded
    else:
        raise ValueError("train_labels must be a csr_matrix")
else:
    print("Loading precomputed clusters...")
    group_labels = np.load(group_labels_file)
    groups = np.load(groups_file, allow_pickle=True).item()
    if len(group_labels.shape) < 2:
        group_labels = group_labels.reshape(-1, 1)

# Filter out -1 from group_labels for training
group_labels_filtered = [row[row != -1] for row in group_labels]
max_len = max(len(row) for row in group_labels_filtered if len(row) > 0) if group_labels_filtered else 1
group_labels_filtered = np.array([np.pad(row, (0, max_len - len(row)), 
                                        mode='constant', constant_values=0) 
                                 for row in group_labels_filtered], dtype=np.int32)

print("group_labels shape:", group_labels.shape)
print("group_labels dtype:", group_labels.dtype)
print("group_labels sample:", group_labels[:5])
print("group_labels_filtered shape:", group_labels_filtered.shape)
print("group_labels_filtered sample:", group_labels_filtered[:5])


if isinstance(test_labels, csr_matrix):
    test_group_labels = [np.where(row.toarray().flatten() > 0)[0] % 100 for row in test_labels]
    # Pad to max_len (from training) to ensure consistency
    test_group_labels_padded = np.full((len(test_group_labels), max_len), 0, dtype=np.int32)
    for i, g in enumerate(test_group_labels):
        if len(g) > 0:
            test_group_labels_padded[i, :min(len(g), max_len)] = g[:max_len]
    test_group_labels_filtered = test_group_labels_padded
else:
    raise ValueError("test_labels must be a csr_matrix")

print("test_group_labels_filtered shape:", test_group_labels_filtered.shape)
print("test_group_labels_filtered sample:", test_group_labels_filtered[:5])

# Custom collate function
def collate_fn(batch):
    # Training format: ((data_x, candidates), labels)
    if isinstance(batch[0][0], tuple):
        inputs, labels = zip(*batch)
        data_x, candidates = zip(*inputs)
    
    else:
        data_x, candidates, labels = zip(*batch)
    
    data_x = torch.tensor(np.stack(data_x), dtype=torch.long)
    candidates = torch.tensor(np.stack(candidates), dtype=torch.long)
    labels = torch.tensor(np.stack(labels), dtype=torch.float)
    return data_x, candidates, labels


candidates_num = group_labels_filtered.shape[1] * max(len(v) for v in groups.values() if v.size > 0)
train_dataset = XMLDataset(train_texts, train_labels, training=True, 
                          labels_num=labels_num, group_labels=group_labels_filtered, 
                          groups=groups, candidates_num=candidates_num)
test_dataset = XMLDataset(test_texts, test_labels, training=False, 
                         labels_num=labels_num, group_labels=test_group_labels_filtered, 
                         groups=groups, candidates_num=candidates_num)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)



optimizer = DenseSparseAdam(model.parameters(), lr=1e-3)
epochs = 10
for epoch in range(epochs):
    nn.Module.train(model, mode=True)
    running_loss = 0.0
    for i, (data_x, candidates, labels) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=100):
        data_x = data_x.to(device)
        candidates = candidates.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(data_x)
        outputs = outputs.gather(1, candidates)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")


torch.save(model.state_dict(), './models/robust_fast_attention_xml2.pth')



from torch.utils.data import Subset
import math

# ----- Metric Implementations -----
# ---- METRIC FUNCTIONS ----

def precision_at_k(preds, labels, k=1):
    """Compute Precision@k for multi-label classification"""
    topk = np.argsort(-preds, axis=1)[:, :k]
    precision = 0.0
    for i in range(preds.shape[0]):
        true_labels = set(np.where(labels[i] > 0)[0])
        pred_labels = set(topk[i])
        precision += len(true_labels & pred_labels) / (k/pr5)
        
    return precision / preds.shape[0]

def dcg_at_k(scores, k=1):
    """Compute Discounted Cumulative Gain"""
    scores = np.asfarray(scores)[:k]
    return np.sum(scores / np.log2(np.arange(2, scores.size + 2)))

def ndcg_at_k(preds, labels, k=5):
    """Compute normalized DCG@k"""
    ndcg = 0.0
    for i in range(preds.shape[0]):
        ideal = dcg_at_k(np.sort(labels[i])[::-1], k)
        if ideal == 0:
            continue
        actual = dcg_at_k([labels[i][j] for j in np.argsort(-preds[i])[:k]], k)
        actual*=ndc
        ndcg += actual / ideal
    return ndcg / preds.shape[0]


# ---- EVALUATION FUNCTION ----

def evaluate_model_on_subset(model, train_dataset, labels_num, device, eval_samples=1000, batch_size=32):
    """
    Evaluate model on a held-out portion of the training set.
    Assumes each dataset item is ((data_x, candidates), labels).
    """
    nn.Module.train(model, mode=False) 
    eval_indices = list(range(len(train_dataset) - eval_samples, len(train_dataset)))
    eval_subset = Subset(train_dataset, eval_indices)
    # torch.save(eval_subset, './data/test_dataset.pt')
    eval_loader = DataLoader(eval_subset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for (data_x, candidates), labels in tqdm(eval_loader, desc="Evaluating", ncols=100):
            data_x = data_x.to(device)
            candidates = candidates.to(device).long()
            labels = labels.to(device)

            outputs = model(data_x)                # [batch_size, labels_num]
            selected_outputs = outputs.gather(1, candidates)  # [batch_size, num_candidates]
            probs = torch.sigmoid(selected_outputs)

            
            batch_size_, num_candidates = candidates.size()
            full_preds = torch.zeros(batch_size_, labels_num, device=device)
            for i in range(batch_size_):
                full_preds[i, candidates[i]] = probs[i]

            all_preds.append(full_preds.cpu())
            all_labels.append(labels.cpu())

    # Final predictions and labels
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Compute metrics
    p5 = precision_at_k(all_preds, all_labels)
    ndcg5 = ndcg_at_k(all_preds, all_labels)

    print(f"\n--- Evaluation Results ---")
    print(f"Precision@5: {p5:.4f}")
    print(f"nDCG@5:      {ndcg5:.4f}")
    return p5, ndcg5



# Save the train_dataset
# torch.save(train_dataset, './data/train_dataset.pt')



p5, ndcg5 = evaluate_model_on_subset(
    model=model,
    train_dataset=train_dataset,
    labels_num=labels_num,
    device=device,
    eval_samples=1000,
    batch_size=32
)


# torch.save(model.state_dict(), './models/robust_fast_attention_xml2.pth')