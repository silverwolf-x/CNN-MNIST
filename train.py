import torch
from tqdm import tqdm
import logging
import numpy as np
from utils import save_model


def trainer(train_loader, valid_loader, model, config):
    # ===prepare===
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.NAdam(model.parameters())
    # # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    criterion = config.criterion()
    optimizer = config.optimizer(model.parameters())
    scheduler = config.scheduler(
        optimizer, T_max=config.n_epoches
    )  #  希望 learning rate 每个epoch更新一次

    early_stop_count = 0
    record = {
        "train_loss": [],
        "valid_loss": [],
        "valid_acc": [],
        "best_loss": 1e5,
        "best_epoch": 0,
    }

    for epoch in range(config.n_epoches):
        # ===train mode===
        model.train()
        train_loss = 0
        train_loop = tqdm(
            train_loader,
            leave=0,
            mininterval=1,
            maxinterval=float("inf"),
            dynamic_ncols=True,
        )
        for batch, (x, y) in enumerate(train_loop):
            x, y = x.to(config.device), y.to(config.device)
            y_pred = model(x)
            # targets的类型是要求long(int64)，这里对齐
            loss = criterion(y_pred, y.long())
            # 清零梯度，反向传播，更新权重
            optimizer.zero_grad()
            loss.backward()
            # scheduler.step(epoch + batch / len(train_loop))
            """
            这里scheduler.step(epoch + batch / iters)的理解如下,如果是一个epoch结束后再.step
            那么一个epoch内所有batch使用的都是同一个学习率,为了使得不同batch也使用不同的学习率
            则可以在这里进行.step
            """
            optimizer.step()
            # 进度条设置
            L = loss.item()
            train_loop.set_description(f"Epoch [{epoch}/{config.n_epoches}]")
            train_loop.set_postfix(
                {"loss": L, "LR": optimizer.param_groups[-1]["lr"]}, refresh=False
            )
            train_loss += loss.item()
        scheduler.step()
        train_loss = train_loss / len(train_loader.dataset)
        record["train_loss"].append(train_loss)

        # ===evaluate mode===
        model.eval()
        valid_loss = 0
        correct = 0
        for x, y in valid_loader:
            x, y = x.to(config.device), y.to(config.device)
            with torch.no_grad():  # 减少内存损耗
                output = model(x)
                loss = criterion(output, y.long())
                pred = output.argmax(dim=1)

                correct += pred.eq(y).sum()
                valid_loss += loss.item()
        valid_accuracy = correct / len(valid_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)
        record["valid_loss"].append(valid_loss)
        record["valid_acc"].append(valid_accuracy)

        # ===early stopping===
        if record["valid_loss"][-1] < record["best_loss"]:
            record["best_loss"] = record["valid_loss"][-1]
            record["best_epoch"] = epoch
            logging.info(
                f"Now model with valid loss {record['best_loss']:.2e}, valid accuracy {record['valid_acc'][-1]:.4f}... from epoch {epoch}"
            )
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count >= config.early_stop:
            logging.info(
                f"Model is not improving for {config.early_stop} epoches. The last epoch is {epoch}."
            )
            break
    torch.save(model.state_dict(), save_model(record["best_loss"]))
    logging.info(
        f"Saving model with valid loss {record['best_loss']:.2e}... from epoch {record['best_epoch']}"
    )
    return record["train_loss"], record["valid_loss"], record["best_loss"]


def predict(test_data, model, config):
    """
    Predicts the output for the given test data.
    Returns:
        preds (numpy array), accuracy (float), incorrect_index (list of indices of incorrect predictions)
    """
    model.eval()
    preds = []
    incorrect_index = []

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config.batch_size, shuffle=False
    )

    with torch.no_grad():
        for batch, (x, y) in tqdm(enumerate(test_loader)):
            x = x.to(config.device)
            output = model(x)
            y_pred = output.argmax(dim=1).cpu().numpy()
            preds.extend(y_pred)

            incorrect_index.extend(np.where(y_pred != y.numpy())[0])

    accuracy = 1 - len(incorrect_index) / len(test_data)

    return preds, accuracy, incorrect_index
