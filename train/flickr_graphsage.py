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
    'batch_size': 1024,        # 减小batch_size适应显存
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
    copy.copy(data),   # 强制数据在GPU上。
    input_nodes=None,
    num_neighbors=[10,10],       # 使用全图邻居
    shuffle=False,
    **kwargs
)

del subgraph_loader.data.x, subgraph_loader.data.y
subgraph_loader.data.num_nodes = data.num_nodes
subgraph_loader.data.n_id = torch.arange(data.num_nodes)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all
    

# 初始化模型
model = SAGE(
    in_channels=dataset.num_features,          # Flickr特征维度
    hidden_channels=512,
    out_channels=dataset.num_classes            # 类别数
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 训练循环

def train(epoch):
    model.train()
    

    pbar = tqdm(total=int(len(train_loader.dataset)))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = total_examples = 0
    for batch in train_loader:
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size
        pbar.update(batch.batch_size)
    pbar.close()

    return total_loss / total_examples, total_correct / total_examples

@torch.no_grad()
def test():
    model.eval()
    y_hat = model.inference(data.x, subgraph_loader).argmax(dim=-1)
    y = data.y.to(y_hat.device)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))
    return accs

times = []
max_acc = 0.0
for epoch in range(1, 2):
    start = time.time()
    loss, acc = train(epoch)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')
    times.append(time.time() - start)

    # 对于每个epoch都保存一次model_state_dict，最终只保存精度最高的那个。
    
    if acc > max_acc:
        max_acc = acc
        # 保存路径
        print(kwargs['batch_size'])
        model_save_path = osp.join(osp.dirname(osp.realpath(__file__)),f"flickr_graphSAGE_{kwargs['batch_size']}.pth")
        print(model_save_path)
        torch.save(model.state_dict(),model_save_path)
        

        print('new model state dict has saved!')
 

print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")