from manydepth.datasets.tartan_dataset import TartanDriveDataset

# 示例文件路径列表
filenames = [
    {"left": "/home/ubuntu22/下载/TarTan dataset/image_left_color/000000.png", "depth": "/home/ubuntu22/下载/TarTan dataset/depth_left/000000.jpg"},
    {"left": "/home/ubuntu22/下载/TarTan dataset/image_left_color/000001.png", "depth": "/home/ubuntu22/下载/TarTan dataset/depth_left/000001.jpg"}
]

# 初始化数据集
dataset = TartanDriveDataset(
    data_path="/home/ubuntu22/下载/TarTan dataset",
    filenames=filenames,
    height=192,              # 图像高度
    width=640,               # 图像宽度
    frame_idxs=[0, -1, 1],   # 帧索引
    num_scales=4             # 缩放级别数量
)

# 打印数据集信息
print(f"Dataset length: {len(dataset)}")
sample = dataset[0]
print(f"Sample keys: {sample.keys()}")
print(f"Image shape: {sample['image'].shape}")
print(f"Depth shape: {sample['depth'].shape}")
print(f"Intrinsics: {sample['intrinsics']}")


# 获取单个样本
sample = dataset[0]
print("Image shape:", sample["image"].shape)
print("Depth shape:", sample["depth"].shape)
print("Intrinsics:", sample["intrinsics"])

# 测试 DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

for batch in dataloader:
    print("Batch image shape:", batch["image"].shape)
    print("Batch depth shape:", batch["depth"].shape)
    print("Batch intrinsics:", batch["intrinsics"])
    break
