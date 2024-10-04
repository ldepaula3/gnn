# Databricks notebook source
# Reference to intall DGL with specific version
%pip install  dgl -f https://data.dgl.ai/wheels/torch-2.3/repo.html

# COMMAND ----------

import argparse
import pickle as pkl

import dgl

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_data
from TAHIN import TAHIN

import mlflow
from utils import (
    evaluate_acc,
    evaluate_auc,
    evaluate_f1_score,
    evaluate_logloss,
)

# COMMAND ----------

def main(gpu, dataset, batch, num_workers, path, in_size, out_size, num_heads, dropout, lr, wd, epochs, early_stop, model):
    # step 1: Check device
    if gpu >= 0 and torch.cuda.is_available():
        device = "cuda:{}".format(gpu)
    else:
        device = "cpu"

    # step 2: Load data
    (
        g,
        train_loader,
        eval_loader,
        test_loader,
        meta_paths,
        user_key,
        item_key,
    ) = load_data(dataset, batch, num_workers, path)
    g = g.to(device)
    print("Data loaded.")

    # step 3: Create model and training components
    model = TAHIN(
        g, meta_paths, in_size, out_size, num_heads, dropout
    )
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    print("Model created.")

    # step 4: Training
    print("Start training.")
    best_acc = 0.0
    kill_cnt = 0

# mlflow.pytorch.autolog()
    with mlflow.start_run(run_name="GNN") as run:
        
        for epoch in range(epochs):
            
            # Training and validation using a full graph
            model.train()
            train_loss = []
            for step, batch in enumerate(train_loader):
                user, item, label = [_.to(device) for _ in batch]
                logits = model.forward(g, user_key, item_key, user, item)

                # compute loss
                tr_loss = criterion(logits, label)
                train_loss.append(tr_loss)

                # backward
                optimizer.zero_grad()
                tr_loss.backward()
                optimizer.step()

            train_loss = torch.stack(train_loss).sum().cpu().item()

            model.eval()
            with torch.no_grad():
                validate_loss = []
                validate_acc = []
                for step, batch in enumerate(eval_loader):
                    user, item, label = [_.to(device) for _ in batch]
                    logits = model.forward(g, user_key, item_key, user, item)

                    # compute loss
                    val_loss = criterion(logits, label)
                    val_acc = evaluate_acc(
                        logits.detach().cpu().numpy(), label.detach().cpu().numpy()
                    )
                    validate_loss.append(val_loss)
                    validate_acc.append(val_acc)

                validate_loss = torch.stack(validate_loss).sum().cpu().item()
                validate_acc = np.mean(validate_acc)

                # validate
                if validate_acc > best_acc:
                    best_acc = validate_acc
                    best_epoch = epoch
                    torch.save(model.state_dict(), "TAHIN" + "_" + dataset)
                    kill_cnt = 0
                    print("saving model...")
                else:
                    kill_cnt += 1
                    if kill_cnt > early_stop:
                        print("early stop.")
                        print("best epoch:{}".format(best_epoch))
                        break
                mlflow.log_metric("validation_loss", val_loss )
                mlflow.log_metric("validation_acc", val_acc )
                print(
                    "In epoch {}, Train Loss: {:.4f}, Valid Loss: {:.5}\n, Valid ACC: {:.5}".format(
                        epoch, train_loss, validate_loss, validate_acc
                    )
                )

        # test use the best model
        model.eval()
        with torch.no_grad():
            model.load_state_dict(torch.load("TAHIN" + "_" + dataset))
            test_loss = []
            test_acc = []
            test_auc = []
            test_f1 = []
            test_logloss = []
            for step, batch in enumerate(test_loader):
                user, item, label = [_.to(device) for _ in batch]
                logits = model.forward(g, user_key, item_key, user, item)

                # compute loss
                loss = criterion(logits, label)
                acc = evaluate_acc(
                    logits.detach().cpu().numpy(), label.detach().cpu().numpy()
                )
                auc = evaluate_auc(
                    logits.detach().cpu().numpy(), label.detach().cpu().numpy()
                )
                f1 = evaluate_f1_score(
                    logits.detach().cpu().numpy(), label.detach().cpu().numpy()
                )
                log_loss = evaluate_logloss(
                    logits.detach().cpu().numpy(), label.detach().cpu().numpy()
                )

                test_loss.append(loss)
                test_acc.append(acc)
                test_auc.append(auc)
                test_f1.append(f1)
                test_logloss.append(log_loss)

            test_loss = torch.stack(test_loss).sum().cpu().item()
            test_acc = np.mean(test_acc)
            test_auc = np.mean(test_auc)
            test_f1 = np.mean(test_f1)
            test_logloss = np.mean(test_logloss)
            mlflow.log_metric("test_loss", test_loss )
            mlflow.log_metric("test_acc", test_acc )
            mlflow.log_metric("test_auc", test_auc )
            mlflow.log_metric("test_f1", test_f1 )
            mlflow.log_metric("test_logloss", test_logloss )

            print(
                "Test Loss: {:.5}\n, Test ACC: {:.5}\n, AUC: {:.5}\n, F1: {:.5}\n, Logloss: {:.5}\n".format(
                    test_loss, test_acc, test_auc, test_f1, test_logloss
                )
            )

# COMMAND ----------

main(dataset = "movielens",
path = "data",
model = "TAHIN",
batch = 128,
gpu = 0,
epochs = 1,
wd = 0,
lr = 0.001,
num_workers = 10,
early_stop = 15,
in_size = 128,
out_size = 128,
num_heads = 1,
dropout = 0.1)

# COMMAND ----------


