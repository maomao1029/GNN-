import copy
import os.path as osp
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.datasets import Flickr
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# 数据集配置
dataset_path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', 'dataset','Flickr')
dataset = Flickr(dataset_path)

# 数据预处理
data = dataset[0].to(device, 'x', 'y')
print(f'Dataset Info:\n'
      f'- Nodes: {data.num_nodes}\n'
      f'- Edges: {data.num_edges}\n'
      f'- Features: {dataset.num_features}\n'
      f'- Classes: {dataset.num_classes}')

# 邻居采样配置
kwargs = {
    'batch_size': 512,        # 减小batch_size适应显存
    'num_workers': 4,
    'persistent_workers': True
}
train_loader = NeighborLoader(
    data,
    input_nodes=data.train_mask,
    num_neighbors=[15, 10],   # 调整邻居采样数
    shuffle=True,
    **kwargs
)

# 全图推理加载器
subgraph_loader = NeighborLoader(
    copy.copy(data),
    input_nodes=None,
    num_neighbors=[10,10],       # 使用全图邻居
    shuffle=False,
    **kwargs
)
del subgraph_loader.data.x, subgraph_loader.data.y
subgraph_loader.data.num_nodes = data.num_nodes
subgraph_loader.data.n_id = torch.arange(data.num_nodes)

# 模型定义
class FlickrSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            SAGEConv(in_channels, 512),    # 增大第一层维度
            SAGEConv(512, 256),
            SAGEConv(256, out_channels)    # 增加第三层
        ])
        self.dropout = 0.3

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index).relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        for conv in self.convs[:-1]:
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)]
                x = conv(x, batch.edge_index.to(device)).relu_()
                xs.append(x[:batch.batch_size].cpu())
            x_all = torch.cat(xs, dim=0)
        return self.convs[-1](x_all.to(device), torch.zeros(1).to(device))  # 最后层GPU计算

# 初始化模型
model = FlickrSAGE(
    in_channels=500,          # Flickr特征维度
    hidden_channels=512,
    out_channels=7            # 类别数
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# 训练循环
def train(epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch:02d}')
    
    for batch in pbar:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
        loss = F.cross_entropy(out, batch.y[:batch.batch_size])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)

@torch.no_grad()
def test():
    model.eval()
    out = model.inference(data.x, subgraph_loader)[0]
    y_pred = out.argmax(dim=-1)
    y_true = data.y.to(device)
    
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        acc = (y_pred[mask] == y_true[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

# 训练配置
best_val_acc = 0
for epoch in range(1, 20):
    train_loss = train(epoch)
    train_acc, val_acc, test_acc = test()
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f'flickr_sage_best_{epoch}.pth')
    
    print(f'Epoch {epoch:03d} | Loss: {train_loss:.4f} | '
          f'Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}')

print(f'Best Validation Accuracy: {best_val_acc:.4f}')