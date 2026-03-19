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
SUCCESS_DIR = os.path.join(ROOT_DIR, "test_dist/upright")
TOPPLE_DIR = os.path.join(ROOT_DIR, "test_dist/upright2")

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
    Calculates an array of distance metrics AND their worst patch indices.
    Returns: Dict[metric_name, (value, list_of_patch_indices)]
    """
    metrics = {}
    
    # 1. Global Flattened L2 (Both Cameras - No specific patch)
    l2_val = torch.norm(z_nom.flatten() - z_pert.flatten()).item()
    metrics["Global_Flattened_L2"] = (l2_val, []) 
    
    # 2. Max Feature Difference (L-Infinity Norm)
    diff = torch.abs(z_nom.flatten() - z_pert.flatten())
    max_val = torch.max(diff).item()
    metrics["Global_L_Infinity"] = (max_val, []) 

    # --- Isolate FIXED Front Camera (Index 0) ---
    z_front_nom = z_nom[0]   # Shape: (196, 384)
    z_front_pert = z_pert[0] # Shape: (196, 384)

    # Calculate per-patch distances
    patch_cosine_dists = 1.0 - F.cosine_similarity(z_front_nom, z_front_pert, dim=-1)
    patch_l2_dists = torch.norm(z_front_nom - z_front_pert, dim=-1)
    
    # Max feature difference inside each individual patch
    patch_max_dists = torch.abs(z_front_nom - z_front_pert).amax(dim=-1)

    # 3. Global Pool Average (GAP) Cosine (No specific patch)
    gap_nom = z_front_nom.mean(dim=0, keepdim=True)
    gap_pert = z_front_pert.mean(dim=0, keepdim=True)
    gap_val = (1.0 - F.cosine_similarity(gap_nom, gap_pert, dim=-1)).item()
    metrics["GAP_Cosine"] = (gap_val, [])

    # 4. Mean Patch Cosine
    mean_val = patch_cosine_dists.mean().item()
    max_cos_idx = torch.argmax(patch_cosine_dists).item()
    metrics["Mean_Patch_Cosine"] = (mean_val, [max_cos_idx])

    # 5. Most Divergent Patch (Max Cosine)
    max_cos_val = patch_cosine_dists.max().item()
    metrics["Max_Patch_Cosine"] = (max_cos_val, [max_cos_idx])

    # 6. Most Divergent Patch (Max L2)
    max_l2_val = patch_l2_dists.max().item()
    max_l2_idx = torch.argmax(patch_l2_dists).item()
    metrics["Max_Patch_L2"] = (max_l2_val, [max_l2_idx])

    # 7. Top-5 Patch Cosine (Returns list of 5 indices)
    top5_cos_vals, top5_cos_indices = torch.topk(patch_cosine_dists, k=3)
    metrics["Top5_Patch_Cosine"] = (top5_cos_vals.mean().item(), top5_cos_indices.tolist())

    # 8. Top-5 Patch Max (L-Infinity max per patch, average top 5)
    top5_max_vals, top5_max_indices = torch.topk(patch_max_dists, k=3)
    metrics["Top5_Patch_Max"] = (top5_max_vals.mean().item(), top5_max_indices.tolist())

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
        
    cv2.putText(img_copy, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
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

    print("\n" + "="*70)
    print(f"{'METRIC':<25} | {'d(start)':<10} | {'d(end)':<10} | {'LYAPUNOV (λ)':<10}")
    print("="*70)
    
    # --- LOAD BASE IMAGES FOR DRAWING ---
    # 1. Toppled Image (Bottom Row)
    raw_front_pert = final_pert_imgs[0].transpose(1, 2, 0).astype(np.uint8)
    raw_front_pert = cv2.cvtColor(raw_front_pert, cv2.COLOR_RGB2BGR)
    base_img_pert = process_gt_image(raw_front_pert, 224)

    # 2. Upright Image (Top Row)
    raw_front_nom = final_nom_imgs[0].transpose(1, 2, 0).astype(np.uint8)
    raw_front_nom = cv2.cvtColor(raw_front_nom, cv2.COLOR_RGB2BGR)
    base_img_nom = process_gt_image(raw_front_nom, 224)
    
    display_images_top = []
    display_images_bottom = []

    for metric in d_start.keys():
        ds_val, _ = d_start[metric]
        de_val, worst_patch_indices = d_end[metric]
        
        ds = ds_val + 1e-8 
        de = de_val + 1e-8
        
        lyap_exp = np.log(de / ds)
        
        print(f"{metric:<25} | {ds:<10.4f} | {de:<10.4f} | {lyap_exp:<10.4f}")
        
        # Only render the image if there are patches to highlight
        if len(worst_patch_indices) > 0:
            # Draw on Upright Image (Top Row) - using Green to show safety
            drawn_nom = draw_patch_boxes(base_img_nom, worst_patch_indices, f"Upright: {metric}", color=(0, 255, 0))
            display_images_top.append(drawn_nom)

            # Draw on Toppled Image (Bottom Row) - using Red to show failure
            drawn_pert = draw_patch_boxes(base_img_pert, worst_patch_indices, f"Toppled: {metric}", color=(0, 0, 255))
            display_images_bottom.append(drawn_pert)

    print("="*70)
    
    if len(display_images_top) > 0:
        # Stack the individual rows horizontally
        top_row = np.hstack(display_images_top)
        bottom_row = np.hstack(display_images_bottom)

        # Stack the two rows vertically
        full_viz = np.vstack([top_row, bottom_row])

        cv2.imshow("Top: Upright | Bottom: Toppled", full_viz)
        print("\nOpening OpenCV window. Press any key in the window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()