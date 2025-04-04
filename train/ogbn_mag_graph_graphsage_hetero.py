import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import ReLU
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import HGTLoader, NeighborLoader
from torch_geometric.nn import Linear, SAGEConv, Sequential, to_hetero

print('包导入完毕！')
parser = argparse.ArgumentParser()
parser.add_argument('--use_hgt_loader', action='store_true')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', 'dataset','ogbn-mag')

# 转换成无向图
print('开始转换成无向图！')
transform = T.ToUndirected(merge=True)   # 转换成无向图，实现边的双向化。
print('无向图转换完毕！')
dataset = OGB_MAG(path, preprocess='metapath2vec', transform=transform)

print('数据1')
# Already send node features/labels to GPU for faster access during sampling:
print(dataset[0])


'''
HeteroData(
  paper={
    x=[736389, 128],    # 特征
    year=[736389],
    y=[736389],         # 分类标签
    train_mask=[736389],
    val_mask=[736389],
    test_mask=[736389],
  },
  author={ x=[1134649, 128] },
  institution={ x=[8740, 128] },
  field_of_study={ x=[59965, 128] },
  (author, affiliated_with, institution)={ edge_index=[2, 1043998] },
  (author, writes, paper)={ edge_index=[2, 7145660] },
  (paper, cites, paper)={ edge_index=[2, 10792672] },
  (paper, has_topic, field_of_study)={ edge_index=[2, 7505078] },
  (institution, rev_affiliated_with, author)={ edge_index=[2, 1043998] },
  (paper, rev_writes, author)={ edge_index=[2, 7145660] },
  (field_of_study, rev_has_topic, paper)={ edge_index=[2, 7505078] }
)


'''
data = dataset[0].to(device, 'x', 'y')

train_input_nodes = ('paper', data['paper'].train_mask)
print(train_input_nodes)


# 原代码中的bincount实现方式（需要先转换为long类型）：
print('\nBincount方式统计:')
print('Train:', torch.bincount(data['paper'].train_mask.long()))
print('Val:  ', torch.bincount(data['paper'].val_mask.long()))
print('Test: ', torch.bincount(data['paper'].test_mask.long()))

# print('train_input_nodes:',train_input_nodes)
# print(torch.bincount(train_input_nodes[1]))
# train_input_nodes: ('paper', tensor([True, True, True,  ..., True, True, True]))

val_input_nodes = ('paper', data['paper'].val_mask)






kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}
print('36行！')
if not args.use_hgt_loader:
    train_loader = NeighborLoader(data, num_neighbors=[10] * 2, shuffle=True,
                                  input_nodes=train_input_nodes, **kwargs)
    val_loader = NeighborLoader(data, num_neighbors=[10] * 2,
                                input_nodes=val_input_nodes, **kwargs)
else:
    train_loader = HGTLoader(data, num_samples=[1024] * 4, shuffle=True,
                             input_nodes=train_input_nodes, **kwargs)
    val_loader = HGTLoader(data, num_samples=[1024] * 4,
                           input_nodes=val_input_nodes, **kwargs)
print('model')
model = Sequential('x, edge_index', [
    (SAGEConv((-1, -1), 64), 'x, edge_index -> x'),
    ReLU(inplace=True),
    (SAGEConv((-1, -1), 64), 'x, edge_index -> x'),
    ReLU(inplace=True),
    (Linear(-1, dataset.num_classes), 'x -> x'),
])
print('to_hetero')
model = to_hetero(model, data.metadata(), aggr='sum').to(device)


@torch.no_grad()
def init_params():
    # Initialize lazy parameters via forwarding a single batch to the model:
    batch = next(iter(train_loader))
    batch = batch.to(device, 'edge_index')
    model(batch.x_dict, batch.edge_index_dict)


def train():
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device, 'edge_index')
        batch_size = batch['paper'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)['paper'][:batch_size]
        loss = F.cross_entropy(out, batch['paper'].y[:batch_size])
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()

    total_examples = total_correct = 0
    for batch in tqdm(loader):
        batch = batch.to(device, 'edge_index')
        batch_size = batch['paper'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)['paper'][:batch_size]
        pred = out.argmax(dim=-1)

        total_examples += batch_size
        total_correct += int((pred == batch['paper'].y[:batch_size]).sum())

    return total_correct / total_examples


init_params()  # Initialize parameters.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



max_acc = 0.0
print('开始for循环')
for epoch in range(1, 2001):
    loss = train()
    val_acc = test(val_loader)
    
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')
    # 保存精度最高的模型参数


    if val_acc > max_acc:
        max_acc = val_acc
        print("kwargs['batch_size']:",kwargs['batch_size'])
        model_save_path = osp.join(osp.dirname(osp.realpath(__file__)),f"1ogbn_mags_GAT_{kwargs['batch_size']}.pth")

        torch.save(model.state_dict(),model_save_path)
        print('The New odel paramters have saved.')