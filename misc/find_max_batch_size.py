import torch
from torch import nn
from model import generate_model  # make sure your model definition is imported
from your_data_loader import preprocess  # update as needed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = generate_model(121).to(device)
model.train()

sample = preprocess("your_image.nii.gz", "your_mask.nii.gz").to(device)
label = torch.tensor([1], dtype=torch.long).to(device)

criterion = nn.CrossEntropyLoss()

with open("batch_test_log.txt", "w") as f:
    for bs in [2, 4, 8, 16, 24, 32, 40, 48, 56, 64]:
        try:
            input_batch = sample.repeat(bs, 1, 1, 1, 1).to(device)
            label_batch = label.repeat(bs).to(device)
            output = model(input_batch)
            loss = criterion(output, label_batch)
            loss.backward()
            torch.cuda.synchronize()
            msg = f"Batch size {bs} worked!"
        except RuntimeError as e:
            msg = f"Batch size {bs} failed: {e}"
            torch.cuda.empty_cache()
        print(msg)
        f.write(msg + "\n")
