from typing import List, Dict, Tuple
from setup import prepare_data
from plot_lib import plot_loss_curves
import torch
from torch import nn
import engine

def train_efficientnet(model,
                weights,
                in_features,
                data_path,
                device,
                optimizer = 'sgd',
                lr = 0.1,
                loss_fn = nn.CrossEntropyLoss(),
                BATCH_SIZE = 32,
                data_transforms = None,
                epochs = 10,
                random_state = 42):

    # Data Loader
    train_loader, test_loader, classes = prepare_data(data_path = data_path,
                     BATCH_SIZE = BATCH_SIZE,
                     data_transforms = data_transforms if data_transforms is not None else weights.transforms(),
                     random_state = random_state)

    # Freezing features
    for param in model.features.parameters():
        param.requires_grad = False

    # Adjusting classifier
    model.classifier = nn.Sequential(
                nn.Dropout(p = 0.2, inplace = True),
                nn.Linear(in_features = in_features, out_features = len(classes), bias = True)
                )

    # Training
    results = engine.train_loop(
        model = model,
        loss_fn = loss_fn,
        optimizer = torch.optim.Adam(params = model.parameters(), lr = lr) if optimizer.lower() == 'adam' else torch.optim.SGD(params = model.parameters(), lr = lr),
        train_loader = train_loader,
        test_loader = test_loader,
        epochs = epochs,
        device = device
    )


    fig = plot_loss_curves(epochs=results["Epoch"],
                     train_loss=results["Train Loss"],
                     test_loss= results["Test Loss"],
                     train_acc=results["Train Accuracy"],
                     test_acc= results["Test Accuracy"])

    return results, fig

def train_resnet(model,
                weights,
                in_features,
                data_path,
                device,
                optimizer = 'adam',
                lr = 0.1,
                loss_fn = nn.CrossEntropyLoss(),
                BATCH_SIZE = 32,
                data_transforms = None,
                epochs = 10,
                random_state = 42):

    # Data Loader
    train_loader, test_loader, classes = prepare_data(data_path = data_path,
                     BATCH_SIZE = BATCH_SIZE,
                     data_transforms = data_transforms if data_transforms is not None else weights.transforms(),
                     random_state = random_state)

    # Freezing features
    for param in model.parameters():
        param.requires_grad = False

    # Adjusting classifier
    model.fc = nn.Sequential(
                nn.Dropout(p = 0.2, inplace = True),
                nn.Linear(in_features = in_features, out_features = len(classes), bias = True)
                )

    # Training
    results = engine.train_loop(
        model = model,
        loss_fn = loss_fn,
        optimizer = torch.optim.Adam(params = model.parameters(), lr = lr) if optimizer.lower() == 'adam' else torch.optim.SGD(params = model.parameters(), lr = lr),
        train_loader = train_loader,
        test_loader = test_loader,
        epochs = epochs,
        device = device
    )


    fig = plot_loss_curves(epochs=results["Epoch"],
                     train_loss=results["Train Loss"],
                     test_loss= results["Test Loss"],
                     train_acc=results["Train Accuracy"],
                     test_acc= results["Test Accuracy"])

    return results, fig
