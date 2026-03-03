import zmq
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# --- CONFIGURATION ---
GPU_SERVER_IP = "host.docker.internal" # Update to your server's IP if running on a separate machine
DINO_PORT = 5556

ROOT_DIR = "tasks"
SUCCESS_DIR = os.path.join(ROOT_DIR, "test_dist/arm_only1")
TOPPLE_DIR = os.path.join(ROOT_DIR, "test_dist/arm_only2")

FILES = {
    "init_front": "init_front.png",
    "init_wrist": "init_wrist.png",
    "final_front": "final_front.png",
    "final_wrist": "final_wrist.png"
}

def load_image_pair(folder, front_name, wrist_name):
    """Loads front and wrist images, formats to (2, 3, H, W)"""
    front_path = os.path.join(folder, front_name)
    wrist_path = os.path.join(folder, wrist_name)
    
    front_img = cv2.imread(front_path)
    wrist_img = cv2.imread(wrist_path)
    
    if front_img is None or wrist_img is None:
        raise FileNotFoundError(f"Missing images in {folder}. Did you name them correctly?")

    # OpenCV loads in BGR. Convert to RGB, then transpose to Channel-First
    front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    wrist_img = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    
    return np.stack([front_img, wrist_img]).astype(np.float32)

def get_latent_embeddings(socket, img_array):
    """Sends images to ZMQ server, returns PyTorch tensor (2, 196, 384)"""
    socket.send_pyobj({'images': img_array})
    response = socket.recv_pyobj()
    
    if 'error' in response:
        raise RuntimeError(f"Server Error: {response['error']}")
        
    return torch.from_numpy(response['z_visual'])

# --- DIVERGENCE METRICS ---
def calc_distances(z_nom, z_pert):
    """
    Calculates distance metrics AND their worst patch indices for BOTH cameras.
    Returns: Dict[metric_name, (value, camera_idx, list_of_patch_indices)]
    """
    metrics = {}
    
    # 1. Global Flattened L2 (Both Cameras combined - No specific patch/camera)
    l2_val = torch.norm(z_nom.flatten() - z_pert.flatten()).item()
    metrics["Global_Flattened_L2"] = (l2_val, None, []) 
    
    # 2. Max Feature Difference (L-Infinity Norm - Both Cameras combined)
    diff = torch.abs(z_nom.flatten() - z_pert.flatten())
    max_val = torch.max(diff).item()
    metrics["Global_L_Infinity"] = (max_val, None, []) 

    # Loop over both views to calculate view-specific metrics
    views = [("Front", 0), ("Wrist", 1)]
    
    for view_name, v_idx in views:
        z_v_nom = z_nom[v_idx]   # Shape: (196, 384)
        z_v_pert = z_pert[v_idx] # Shape: (196, 384)

        # Calculate per-patch distances
        patch_cosine_dists = 1.0 - F.cosine_similarity(z_v_nom, z_v_pert, dim=-1)
        patch_l2_dists = torch.norm(z_v_nom - z_v_pert, dim=-1)
        patch_max_dists = torch.abs(z_v_nom - z_v_pert).amax(dim=-1)

        # GAP Cosine (No specific patch)
        gap_nom = z_v_nom.mean(dim=0, keepdim=True)
        gap_pert = z_v_pert.mean(dim=0, keepdim=True)
        gap_val = (1.0 - F.cosine_similarity(gap_nom, gap_pert, dim=-1)).item()
        metrics[f"{view_name}_GAP_Cosine"] = (gap_val, v_idx, [])

        # Mean Patch Cosine
        mean_val = patch_cosine_dists.mean().item()
        max_cos_idx = torch.argmax(patch_cosine_dists).item()
        metrics[f"{view_name}_Mean_Patch_Cosine"] = (mean_val, v_idx, [max_cos_idx])

        # Max Patch Cosine
        max_cos_val = patch_cosine_dists.max().item()
        metrics[f"{view_name}_Max_Patch_Cosine"] = (max_cos_val, v_idx, [max_cos_idx])

        # Max Patch L2
        max_l2_val = patch_l2_dists.max().item()
        max_l2_idx = torch.argmax(patch_l2_dists).item()
        metrics[f"{view_name}_Max_Patch_L2"] = (max_l2_val, v_idx, [max_l2_idx])

        # Top-5 Patch Cosine
        top5_cos_vals, top5_cos_indices = torch.topk(patch_cosine_dists, k=5)
        metrics[f"{view_name}_Top5_Patch_Cosine"] = (top5_cos_vals.mean().item(), v_idx, top5_cos_indices.tolist())

        # Top-5 Patch Max
        top5_max_vals, top5_max_indices = torch.topk(patch_max_dists, k=5)
        metrics[f"{view_name}_Top5_Patch_Max"] = (top5_max_vals.mean().item(), v_idx, top5_max_indices.tolist())

    return metrics

def process_gt_image(img, target_size=224):
    """Mimics PyTorch's Resize(224) + CenterCrop(224) to match decoder outputs."""
    h, w = img.shape[:2]
    scale = target_size / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    start_x = (new_w - target_size) // 2
    start_y = (new_h - target_size) // 2
    return img_resized[start_y:start_y+target_size, start_x:start_x+target_size]

def draw_patch_boxes(img, patch_indices, label, color=(0, 0, 255)):
    """Draws multiple 16x16 bounding boxes on the image given a list of 1D patch indices."""
    img_copy = img.copy()
    
    for patch_idx in patch_indices:
        row = patch_idx // 14
        col = patch_idx % 14
        x_start = col * 16
        y_start = row * 16
        cv2.rectangle(img_copy, (x_start, y_start), (x_start + 16, y_start + 16), color, 2)
        
    cv2.putText(img_copy, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return img_copy

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{GPU_SERVER_IP}:{DINO_PORT}")
    print(f"Connected to DINO Encoder Server at {GPU_SERVER_IP}:{DINO_PORT}.")

    print("\nLoading Images...")
    init_nom_imgs = load_image_pair(SUCCESS_DIR, FILES['init_front'], FILES['init_wrist'])
    init_pert_imgs = load_image_pair(TOPPLE_DIR, FILES['init_front'], FILES['init_wrist'])
    
    final_nom_imgs = load_image_pair(SUCCESS_DIR, FILES['final_front'], FILES['final_wrist'])
    final_pert_imgs = load_image_pair(TOPPLE_DIR, FILES['final_front'], FILES['final_wrist'])

    print("Fetching Embeddings from Server...")
    z_init_nom = get_latent_embeddings(socket, init_nom_imgs)
    z_init_pert = get_latent_embeddings(socket, init_pert_imgs)
    z_final_nom = get_latent_embeddings(socket, final_nom_imgs)
    z_final_pert = get_latent_embeddings(socket, final_pert_imgs)

    print("Calculating Initial Distances d(0)...")
    d_start = calc_distances(z_init_nom, z_init_pert)
    
    print("Calculating Final Distances d(T)...")
    d_end = calc_distances(z_final_nom, z_final_pert)

    print("\n" + "="*80)
    print(f"{'METRIC':<30} | {'d(start)':<10} | {'d(end)':<10} | {'LYAPUNOV (λ)':<10}")
    print("="*80)
    
    # --- LOAD BASE IMAGES FOR BOTH CAMERAS ---
    # Front Camera Bases (Index 0)
    base_front_pert = process_gt_image(cv2.cvtColor(final_pert_imgs[0].transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2BGR), 224)
    base_front_nom = process_gt_image(cv2.cvtColor(final_nom_imgs[0].transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2BGR), 224)
    
    # Wrist Camera Bases (Index 1)
    base_wrist_pert = process_gt_image(cv2.cvtColor(final_pert_imgs[1].transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2BGR), 224)
    base_wrist_nom = process_gt_image(cv2.cvtColor(final_nom_imgs[1].transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2BGR), 224)
    
    display_images_top = []
    display_images_bottom = []

    for metric in d_start.keys():
        ds_val, _, _ = d_start[metric]
        de_val, cam_idx, worst_patch_indices = d_end[metric]
        
        ds = ds_val + 1e-8 
        de = de_val + 1e-8
        lyap_exp = np.log(de / ds)
        
        print(f"{metric:<30} | {ds:<10.4f} | {de:<10.4f} | {lyap_exp:<10.4f}")
        
        if len(worst_patch_indices) > 0:
            # Strip "Front_" or "Wrist_" from string so it fits on screen
            short_label = metric.split('_', 1)[1] 
            
            if cam_idx == 0:
                img_nom = base_front_nom
                img_pert = base_front_pert
                cam_label = "F"
            else:
                img_nom = base_wrist_nom
                img_pert = base_wrist_pert
                cam_label = "W"

            # Draw on Upright Image (Top Row)
            drawn_nom = draw_patch_boxes(img_nom, worst_patch_indices, f"{cam_label} Safe: {short_label}", color=(0, 255, 0))
            display_images_top.append(drawn_nom)

            # Draw on Toppled Image (Bottom Row) 
            drawn_pert = draw_patch_boxes(img_pert, worst_patch_indices, f"{cam_label} Fail: {short_label}", color=(0, 0, 255))
            display_images_bottom.append(drawn_pert)

    print("="*80)
    
    if len(display_images_top) > 0:
        top_row = np.hstack(display_images_top)
        bottom_row = np.hstack(display_images_bottom)
        full_viz = np.vstack([top_row, bottom_row])

        cv2.imshow("Top: Upright (Safe) | Bottom: Toppled (Fail)", full_viz)
        print("\nOpening OpenCV window. Press any key in the window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()