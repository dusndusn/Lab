import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

x_train_rnn = np.load('x_train_rnn.npy', allow_pickle=True)
x_test_rnn = np.load('x_test_rnn.npy', allow_pickle=True)
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(GRUModel, self).__init__()
        self.gru=nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc=nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, length):
        packed_input=nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        packed_output, h_n=self.gru(packed_input)
        output, _= nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output=self.fc(output)
        return output
 
'''

# 데이터 전처리
vocab_size = 50000  # 추정치로 설정
max_len = 100  # 최대 시퀀스 길이

def preprocess_rnn_data(data, vocab_size, max_len):
    sequences = []
    for seq in data:
        tokens = [int(item.split(':')[1]) for item in seq.split()]
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        sequences.append(tokens[:max_len])
    return torch.tensor(sequences, dtype=torch.long)

x_train_rnn = preprocess_rnn_data(x_train_rnn, vocab_size, max_len)
x_test_rnn = preprocess_rnn_data(x_test_rnn, vocab_size, max_len)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 모델 초기화
input_dim = vocab_size
hidden_dim = 128
output_dim = 1
n_layers = 1

model = GRUModel(input_dim, hidden_dim, output_dim, n_layers)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
num_epochs = 10
batch_size = 64

for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(x_train_rnn), batch_size):
        x_batch = x_train_rnn[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        outputs = model(x_batch)
        loss = criterion(outputs.squeeze(), y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 모델 저장
torch.save(model.state_dict(), 'gru_model.pth')

# 성능 평가
def evaluate(model, x_data, y_data):
    model.eval()
    with torch.no_grad():
        y_pred = model(x_data).squeeze().numpy()
    return roc_auc_score(y_data.numpy(), y_pred), average_precision_score(y_data.numpy(), y_pred)

train_auroc, train_auprc = evaluate(model, x_train_rnn, y_train)
test_auroc, test_auprc = evaluate(model, x_test_rnn, y_test)


student_id = '20215044'
with open(f'{student_id}_rnn.txt', 'w') as f:
    f.write(f'{student_id}\n')
    f.write(f'{train_auroc:.4f}\n')
    f.write(f'{train_auprc:.4f}\n')
    f.write(f'{test_auroc:.4f}\n')
    f.write(f'{test_auprc:.4f}\n')

'''
