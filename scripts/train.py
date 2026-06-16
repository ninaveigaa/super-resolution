import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path

from models.edsr import EDSR
from models.hat import HAT
from src.dataset import get_dataloader


def get_model(model_name: str):
    if model_name == "edsr":
        return EDSR()
    elif model_name == "hat":
        return HAT()
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")


def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = get_model(args.model).to(device)
    dataloader = get_dataloader(args.data_dir, batch_size=args.batch_size)
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    output_dir = Path("outputs") / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for lr, hr in dataloader:
            lr, hr = lr.to(device), hr.to(device)

            y_hat = model(lr)
            loss = loss_fn(y_hat, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), output_dir / "weights.pth")
    print(f"Modelo salvo em {output_dir / 'weights.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["edsr", "hat"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_dir", type=str, default="data/processed/pairs")
    args = parser.parse_args()

    train(args)
