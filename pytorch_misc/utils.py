import torch
from tqdm import tqdm
from os import path


def train_model(
    model,
    train_loader,
    optimizer,
    criterion,
    epochs,
    val_loader=None,
    device="cpu",
    save_folder=".",
    save_best=True,
):
    model = model.to(device)
    model.train()

    min_train_loss = float("inf")
    min_val_loss = float("inf")

    save_folder = path.abspath(save_folder)

    if val_loader is None:
        print("No validation loader provided. Skipping validation.")
        losses = dict(train=[])
    else:
        losses = dict(train=[], validation=[])

    for epoch in tqdm(range(epochs)):
        train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(
            tqdm(train_loader, desc=f"Training... Epoch {epoch+1}/{epochs}", ncols=100)
        ):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        if val_loader:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for batch_idx, (inputs, targets) in enumerate(
                    tqdm(
                        val_loader,
                        desc=f"Validation... Epoch {epoch+1}/{epochs}",
                        ncols=100,
                    )
                ):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            model.train()

        if val_loader is not None:
            losses["validation"].append(val_loss / len(val_loader))
            losses["train"].append(train_loss / len(train_loader))

            if train_loss < min_train_loss:
                min_train_loss = train_loss

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(model.state_dict(), path.join(save_folder, "weights.pt"))

                if save_best:
                    torch.save(model, path.join(save_folder, "best_model.pt"))

        else:
            losses["train"].append(train_loss / len(train_loader))

            if train_loss < min_train_loss:
                min_train_loss = train_loss
                torch.save(model.state_dict(), path.join(save_folder, "weights.pt"))

                if save_best:
                    torch.save(model, path.join(save_folder, "best_model.pt"))

        print(
            f"Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {val_loss / len(val_loader)}"
        )

    model.state_dict = torch.load(path.join(save_folder, "weights.pt"))

    return model, losses
