from model.load import FingerVideoDataset, build_finger_dataloader

import matplotlib.pyplot as plt


PROCESSED_DIR = "data/processed-finger-sam-2026-01-30T22-41-47-949Z"


# Single sample
dataset = FingerVideoDataset(PROCESSED_DIR)
coords, frame = dataset[10]

print("coords:", coords)
print("coords shape:", coords.shape)
print("frame shape:", frame.shape)


# Batch
loader = build_finger_dataloader(
    PROCESSED_DIR,
    batch_size=8,
    shuffle=True,
    require_finger=True,
    require_shirt=True,
    min_shirt_sample_count=1500,
)

batch_coords, batch_frames = next(iter(loader))
print("batch_coords shape:", batch_coords.shape)
print("batch_frames shape:", batch_frames.shape)


# Visualize one frame and its fingertip
img = frame.permute(1, 2, 0).cpu().numpy()
x = coords[0].item() * img.shape[1]
y = coords[1].item() * img.shape[0]

plt.figure(figsize=(8, 5))
plt.imshow(img)
plt.scatter([x], [y], c="red", s=60)
plt.title(f"finger tip: ({x:.1f}, {y:.1f})")
plt.axis("off")
plt.show()
