import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 加载图像文件
def load_image(file_path):
    transform = transforms.Compose([
        transforms.Resize((544, 1024)),  # 调整图像大小
        transforms.ToTensor(),          # 转换为张量
    ])
    image = Image.open(file_path).convert("RGB")  # 确保是 RGB 格式
    return transform(image).unsqueeze(0)  # 增加 batch 维度

# 加载深度图文件
def load_depth(file_path):
    depth = np.load(file_path)  # 假设深度图是 .npy 文件
    depth = torch.tensor(depth, dtype=torch.float32)  # 转换为张量
    if len(depth.shape) == 2:  # 如果是 (height, width)
        depth = depth.unsqueeze(0).unsqueeze(0)  # 增加 batch 和通道维度
    return depth

# 加载内参
K = np.array([
    [455.77496337890625, 0.0, 497.1180114746094],
    [0.0, 456.319091796875, 251.58502197265625],
    [0.0, 0.0, 1.0]
], dtype=np.float32)
K[0, :] *= 640 / 1024  # 缩放内参
K[1, :] *= 192 / 544
K_tensor = torch.tensor(K).unsqueeze(0)

# 加载位姿数据
def load_poses(file_path):
    poses = []
    current_pose = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                current_pose.append(list(map(float, line.split())))  # 尝试转换为浮点数
                if len(current_pose) == 4:  # 累积4行形成一个4x4矩阵
                    poses.append(torch.tensor(current_pose, dtype=torch.float32))
                    current_pose = []  # 清空，准备下一个矩阵
            except ValueError:
                continue  # 跳过无法解析的行
    return poses

# 示例：加载位姿数据
poses = load_poses("/Users/fengyingying/Downloads/2023-11-14-14-45-20_gupta/code/multisense_imu/poses.txt")
print(f"Loaded {len(poses)} poses")
print(poses[0])  # 打印第一个位姿矩阵

# BackprojectDepth 类
class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud"""

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = nn.Parameter(torch.from_numpy(np.stack(meshgrid, axis=0).astype(np.float32)),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        pix_coords = torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0).unsqueeze(0)
        pix_coords = pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        # 确保 depth 是 (batch_size, 1, height, width)
        depth = depth.view(self.batch_size, 1, -1)

        # 相机坐标点
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points

# Project3D 类
class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T"""

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        # 提取 T 的 3x4 部分
        T = T[:, :3, :]  # 只保留前 3 行

        # 投影计算
        P = torch.matmul(K, T)
        cam_points = torch.matmul(P, points)

        # 归一化像素坐标
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width).permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        return (pix_coords - 0.5) * 2  # 归一化到 [-1, 1]

# ImageIMUVerifier 类
class ImageIMUVerifier:
    def __init__(self, intrinsics, poses, batch_size=1, height=512, width=640):
        self.K = torch.tensor(intrinsics['K']).view(3, 3)
        self.inv_K = torch.inverse(self.K).unsqueeze(0)  # 批次维度
        self.poses = poses

        # 初始化 BackprojectDepth 和 Project3D
        self.backproject_depth_layer = BackprojectDepth(batch_size=batch_size, height=height, width=width)
        self.project_3d_layer = Project3D(batch_size=batch_size, height=height, width=width)

    def generate_images_pred(self, inputs, outputs, depth, current_pose, prev_pose):
        T = torch.matmul(torch.inverse(prev_pose), current_pose)
        cam_points = self.backproject_depth_layer(depth, self.inv_K)  # 深度反投影
        pix_coords = self.project_3d_layer(cam_points, self.K.unsqueeze(0), T.unsqueeze(0))  # 投影
        return F.grid_sample(inputs["color"], pix_coords, padding_mode="border", align_corners=True)

    def verify(self, inputs, outputs):
        depth = outputs["depth"]
        current_pose = self.poses[inputs["current_index"]]
        prev_pose = self.poses[inputs["prev_index"]]
        predicted_image = self.generate_images_pred(inputs, outputs, depth, current_pose, prev_pose)
        return predicted_image

class ReprojectionLoss(nn.Module):
    def __init__(self, use_ssim=True, device="cpu"):
        super(ReprojectionLoss, self).__init__()
        self.use_ssim = use_ssim
        self.device = device

    def forward(self, pred, target):
        """
        Computes reprojection loss between a batch of predicted and target images
        Args:
            pred (torch.Tensor): Predicted images, shape [B, C, H, W]
            target (torch.Tensor): Target images, shape [B, C, H, W]
        Returns:
            reprojection_loss (torch.Tensor): Reprojection loss
        """
        # L1 Loss
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)  # Reduce along channel dimension

        if self.use_ssim:
            # SSIM Loss
            ssim_loss = self.compute_ssim(pred, target).mean(1, True)  # Reduce along channel dimension
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        else:
            reprojection_loss = l1_loss

        return reprojection_loss

    def compute_ssim(self, img1, img2):
        """
        Computes SSIM loss between two images
        Args:
            img1 (torch.Tensor): Image 1, shape [B, C, H, W]
            img2 (torch.Tensor): Image 2, shape [B, C, H, W]
        Returns:
            ssim_loss (torch.Tensor): SSIM loss
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # 3x3 Average Pooling (smoothing kernel)
        kernel = torch.ones((1, 1, 3, 3), device=self.device) / 9.0  # Ensure kernel size matches filter size

        mu1 = F.conv2d(img1, kernel, padding=1, groups=img1.shape[1])
        mu2 = F.conv2d(img2, kernel, padding=1, groups=img2.shape[1])

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, kernel, padding=1, groups=img1.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, kernel, padding=1, groups=img2.shape[1]) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, kernel, padding=1, groups=img1.shape[1]) - mu1_mu2

        ssim_num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        ssim_den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim = ssim_num / (ssim_den + 1e-7)  # Add epsilon to avoid division by zero

        # SSIM Loss
        ssim_loss = torch.clamp((1 - ssim) / 2, 0, 1)
        return ssim_loss

# 示例运行
intrinsics = {"K": [[455.77496337890625, 0.0, 497.1180114746094],
                    [0.0, 456.319091796875, 251.58502197265625],
                    [0.0, 0.0, 1.0]]}

verifier = ImageIMUVerifier(intrinsics, poses)

inputs = {
    "color": load_image("/Users/fengyingying/Downloads/2023-11-14-14-45-20_gupta/image_left_color/001234.png"),
    "color_prev": load_image("/Users/fengyingying/Downloads/2023-11-14-14-45-20_gupta/image_left_color/001233.png"),
    "current_index": 1,
    "prev_index": 0,
}

outputs = {
    "depth": load_depth("/Users/fengyingying/Downloads/2023-11-14-14-45-20_gupta/depth_left/001234.npy")
}

# 验证深度值范围
outputs["depth"][outputs["depth"] <= 0] = 1e-3  # 修复无效深度
print(f"Depth range: {outputs['depth'].min()}, {outputs['depth'].max()}")

# 调用 verify
predicted_image = verifier.verify(inputs, outputs)
print("Predicted image generated successfully.")

# 将预测图像转换为 NumPy 数组
predicted_image_np = predicted_image[0].permute(1, 2, 0).cpu().detach().numpy()

# 归一化到 [0, 1]
predicted_image_np = (predicted_image_np - predicted_image_np.min()) / (predicted_image_np.max() - predicted_image_np.min())

# 显示图像
plt.imshow(predicted_image_np)
plt.axis('off')
plt.show()

# 保存图像
plt.imsave("/Users/fengyingying/Downloads/predicted_image.png", predicted_image_np)
print("Predicted image saved to /path/to/save/predicted_image.png")



###  验证位姿的准确性
# 实际上一帧图像
actual_image = inputs["color_prev"][0].permute(1, 2, 0).cpu().detach().numpy()
actual_image = (actual_image - actual_image.min()) / (actual_image.max() - actual_image.min())

# 预测图像
predicted_image_np = predicted_image[0].permute(1, 2, 0).cpu().detach().numpy()
predicted_image_np = (predicted_image_np - predicted_image_np.min()) / (predicted_image_np.max() - predicted_image_np.min())

# 显示比较
plt.subplot(1, 2, 1)
plt.title("Actual Previous Image")
plt.imshow(actual_image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Predicted Image")
plt.imshow(predicted_image_np)
plt.axis('off')

plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 确保 predicted_image 和 inputs["color_prev"] 尺寸一致
if predicted_image.shape != inputs["color_prev"].shape:
    inputs["color_prev"] = F.interpolate(inputs["color_prev"], size=predicted_image.shape[2:], mode="bilinear",
                                         align_corners=False)
# 实例化 ReprojectionLoss
reprojection_loss_fn = ReprojectionLoss(use_ssim=True)
# 计算重投影损失
reprojection_loss = reprojection_loss_fn(predicted_image, inputs["color_prev"])

# 打印损失
print("Reprojection Loss:", reprojection_loss.mean().item())