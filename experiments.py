
from torch import manual_seed, zeros, device, autograd
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.cuda import is_available
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

from models.custom_gnn import Custom_GNN
from tasks.barbell import BarbellDataset
from tasks.path import PathDataset
from tasks.ring import RingDataset
from tasks.barbell_star import BarbellStarDataset

from tqdm import tqdm
import argparse

def test(model, test_loader, device):
    model.eval()

    running_acc = 0.0
    for data in test_loader:
        data = data.to(device)

        out = model(data)
        out = out[data.mask]

        pred = out.argmax(dim=1)
        acc = (pred == data.y).float().mean().item()
        running_acc += acc / len(test_loader)

    return running_acc

def train_epoch(model, optimizer, train_loader, loss_fn, device):
    running_loss = 0.0
    train_acc = 0.0
    model.train()
    for data in train_loader:
        data = data.to(device)

        optimizer.zero_grad()
        out = model(data)
        out = out[data.mask]
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() / len(train_loader)
        pred = out.argmax(dim=1)
        acc = (pred == data.y).float().mean().item()
        train_acc += acc / len(train_loader)
    
    return running_loss, train_acc

def train(model, train_loader, val_loader, test_loader, device, num_epochs=200):
    optimizer = AdamW(model.parameters())
    loss_fn = CrossEntropyLoss()

    losses = []
    train_accuracies = []
    val_accuracies = []

    pbar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")
    for epoch in pbar:
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, loss_fn, device)

        losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_acc = test(model, val_loader, device)

        if val_acc > 0.95:
            print("Validation accuracy exceeded 0.95, stopping training.")
            break

        val_accuracies.append(val_acc)

        pbar.set_postfix({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        })

    test_accuracy = test(model, test_loader, device)

    return losses, train_accuracies, val_accuracies, test_accuracy

def benchmark(models, dataset, ns):
    assert set(models).issubset(set(['GCN', 'GAT', 'HK', 'HK_mu', 'CTQW', 'CTQW_mu']))
    assert dataset in ['PathDataset', 'BarbellDataset', 'RingDataset', 'BarbellStarDataset']
    mu_with_features = dataset in ['RingDataset']

    dev = device('cuda' if is_available() else 'cpu')
    #autograd.set_detect_anomaly(True)

    train_size = int(2**13)
    val_size = int(2**10)
    test_size = int(2**10)

    dim_emb = 10
    dim_pe = 4
    hidden_channels = 10

    batch_size = 64

    results = zeros((len(models), len(ns), 10))
    for n in ns:
        print(f"Starting n={n}")
        for run in range(10):
            print(f"Running {dataset} with n={n}, run={run}")
            train_set = eval(dataset)('.', n=n, size=train_size, dim_emb=dim_emb, dim_pe=dim_pe)
            val_set = eval(dataset)('.', n=n, size=val_size, dim_emb=dim_emb, dim_pe=dim_pe)
            test_set = eval(dataset)('.', n=n, size=test_size, dim_emb=dim_emb, dim_pe=dim_pe)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

            for model_name in models:
                model = Custom_GNN(in_channels=dim_emb,
                                    hidden_channels=hidden_channels,
                                    out_channels=dim_emb,
                                    num_layers=n,
                                   model=model_name,
                                   mu_with_features=mu_with_features).to(dev)
                
                losses, train_accuracies, val_accuracies, test_accuracy = train(model, train_loader, val_loader, test_loader, dev)
                print(f'Model: {model_name}, Dataset: {dataset}, Test Acc: {test_accuracy:.4f}')
                results[models.index(model_name), ns.index(n), run] = test_accuracy

        #plot test_accuracy with respect to n for each model (contained in results)
        plt.figure(figsize=(10, 6))
        for i, model_name in enumerate(models):
            plt.plot(ns, results[i].mean(dim=-1), label=model_name)
        plt.xlabel('n')
        plt.ylabel('Test Accuracy')
        plt.title(f'Test Accuracy vs n for {dataset}')
        plt.legend()
        plt.savefig(f'{dataset}_test_accuracy.png')
        plt.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Benchmark GNN models on various datasets.")
    argparser.add_argument('--dataset', choices=['PathDataset', 'BarbellDataset', 'RingDataset', 'BarbellStarDataset'], required=True, help='Dataset to use for benchmarking.')
    args = argparser.parse_args()
    args = vars(args)

    manual_seed(42)
    models = ['GCN', 'GAT', 'HK', 'HK_mu', 'CTQW', 'CTQW_mu']
    dataset = args['dataset']
    ns = list(range(4, 32, 4))
    print("Starting benchmarks")
    benchmark(models, dataset, ns)
