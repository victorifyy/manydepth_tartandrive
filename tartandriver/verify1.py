import torch
import numpy as np
import cv2
from manydepth.networks import DepthDecoder, ResnetEncoder
from manydepth.layers import disp_to_depth
import matplotlib.pyplot as plt

# Load image data
image_t = cv2.imread('/content/gdrive/My Drive/data/Tartan_data/sample/image_left_color/000001.png')  # Replace with actual path
image_t = cv2.cvtColor(image_t, cv2.COLOR_BGR2RGB)  # Convert to RGB format
image_t = cv2.resize(image_t, (640, 192))  # Resize to model input dimensions
image_t_tensor = torch.tensor(image_t).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Convert to tensor and normalize
print("Image tensor size:", image_t_tensor.shape)

# Load depth data
depth_map_t = np.load('/content/gdrive/My Drive/data/Tartan_data/sample/depth_left/000001.npy')  # Replace with actual path
depth_map_t_resized = cv2.resize(depth_map_t, (640, 192))  # Resize to match image dimensions
depth_map_t_tensor = torch.tensor(depth_map_t_resized).unsqueeze(0).unsqueeze(0).float()
depth_map_t_tensor = depth_map_t_tensor.clamp(0.1, 100.0)  # Clamp depth values to a valid range
print("Depth map tensor size:", depth_map_t_tensor.shape)

# Load camera intrinsics and rescale
K = np.array([
    [455.77496337890625, 0.0, 497.1180114746094],
    [0.0, 456.319091796875, 251.58502197265625],
    [0.0, 0.0, 1.0]
], dtype=np.float32)
K[0, :] *= 640 / 1024  # Rescale width
K[1, :] *= 192 / 544  # Rescale height
K_tensor = torch.tensor(K).unsqueeze(0)
print("Camera intrinsic matrix K:", K_tensor)

# Load relative pose
relative_pose = np.array([
    [0.989252,  0.146223, -0.000261,  0.000024],
    [-0.087943, 0.596392,  0.797861,  0.000229],
    [0.116822, -0.789263,  0.602841, -0.095673],
    [0.0,       0.0,       0.0,       1.0]
], dtype=np.float32)
relative_pose_tensor = torch.tensor(relative_pose).unsqueeze(0)

# Initialize models
encoder = ResnetEncoder(18, False)  # ResNet18 encoder
depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

# Load pretrained weights
encoder_weights = torch.load('/content/manydepth/models/KITTI_MR/encoder.pth', map_location='cpu')  # Replace with actual path
depth_decoder_weights = torch.load('/content/manydepth/models/KITTI_MR/depth.pth', map_location='cpu')  # Replace with actual path
encoder.load_state_dict(encoder_weights, strict=False)
depth_decoder.load_state_dict(depth_decoder_weights, strict=False)

encoder.eval()
depth_decoder.eval()

# Perform depth estimation
with torch.no_grad():
    features = encoder(image_t_tensor)  # Extract features
    outputs = depth_decoder(features)  # Generate depth map
    disp = outputs[("disp", 0)]
    _, predicted_depth = disp_to_depth(disp, 0.1, 100.0)
print("Predicted depth map size:", predicted_depth.shape)

# Project 3D points to 2D
def project_3d_to_2d(depth, K, pose, image):
    """Projects 3D points onto a 2D plane and uses grid_sample to generate the previous frame image."""
    batch_size, _, height, width = depth.shape

    # Create pixel grid
    i, j = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=depth.device),
        torch.arange(width, dtype=torch.float32, device=depth.device),
        indexing="ij"
    )
    pix_coords = torch.stack([j, i, torch.ones_like(i)], dim=0).unsqueeze(0)  # [1, 3, H, W]

    # Convert pixel coordinates to camera coordinates
    cam_points = torch.matmul(torch.linalg.inv(K), pix_coords.view(batch_size, 3, -1))
    cam_points = cam_points * depth.view(batch_size, 1, -1)  # Apply depth
    cam_points = torch.cat([cam_points, torch.ones_like(cam_points[:, :1])], dim=1)  # Convert to homogeneous coordinates

    # Convert camera coordinates to world coordinates
    world_points = torch.matmul(pose, cam_points)

    # Convert world coordinates to pixel coordinates
    pix_coords = torch.matmul(K, world_points[:, :3])  # [B, 3, N]
    pix_coords = pix_coords[:, :2] / pix_coords[:, 2:3]  # Normalize
    pix_coords = pix_coords.view(batch_size, 2, height, width).permute(0, 2, 3, 1)  # [B, H, W, 2]

    # Normalize to grid_sample range [-1, 1]
    pix_coords[..., 0] = 2.0 * pix_coords[..., 0] / (width - 1) - 1.0
    pix_coords[..., 1] = 2.0 * pix_coords[..., 1] / (height - 1) - 1.0

    # Generate projected image using grid_sample
    projected_image = torch.nn.functional.grid_sample(
        image, pix_coords, align_corners=True, mode="bilinear", padding_mode="zeros"
    )

    return projected_image

# Project depth map to generate previous frame image
predicted_image_t_minus_1 = project_3d_to_2d(predicted_depth, K_tensor, relative_pose_tensor, image_t_tensor)
print("Projected image generated successfully.")

# Visualize the predicted previous frame image
plt.imshow(predicted_image_t_minus_1.squeeze().permute(1, 2, 0).cpu().numpy())
plt.title("Predicted Image at t-1")
plt.show()

# Compute reprojection loss
def compute_reprojection_loss(predicted, target):
    return torch.mean(torch.abs(predicted - target))

# computation
real_image_t_minus_1 = torch.tensor(image_t).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Assume current frame as actual image
loss = compute_reprojection_loss(predicted_image_t_minus_1, real_image_t_minus_1)
print("Reprojection loss:", loss.item())

# Adjust depth values
predicted_depth = predicted_depth * 50  # Dynamically scale
print("Adjusted depth range:", predicted_depth.min().item(), predicted_depth.max().item())

# Validate relative pose (simple translation test)
relative_pose_tensor = torch.eye(4).unsqueeze(0)  # Identity matrix
relative_pose_tensor[0, :3, 3] = torch.tensor([0, 0, -1.0])  # Add simple translation

# Clamp projected coordinates
pix_coords[..., 0] = torch.clamp(pix_coords[..., 0], 0, image_t_tensor.shape[-1] - 1)
pix_coords[..., 1] = torch.clamp(pix_coords[..., 1], 0, image_t_tensor.shape[-2] - 1)
print("Projected coordinates range: x_min={}, x_max={}, y_min={}, y_max={}".format(
    pix_coords[..., 0].min().item(),
    pix_coords[..., 0].max().item(),
    pix_coords[..., 1].min().item(),
    pix_coords[..., 1].max().item()
))

# Check for invalid coordinates
print("Contains invalid coordinates:", torch.isnan(pix_coords).any().item())

# Visualize adjusted depth map
plt.imshow(predicted_depth.squeeze().cpu().numpy(), cmap='plasma')
plt.title("Adjusted Predicted Depth Map")
plt.colorbar()
plt.show()