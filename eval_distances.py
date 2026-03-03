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
    Calculates an array of distance metrics. 
    Input shape: (2, 196, 384) -> [Front (0), Wrist (1)]
    """
    metrics = {}
    
    # 1. Global Flattened L2 (Both Cameras)
    metrics["Global_Flattened_L2"] = torch.norm(z_nom.flatten() - z_pert.flatten()).item()
    
    # 2. Max Feature Difference (L-Infinity Norm)
    metrics["Global_L_Infinity"] = torch.max(torch.abs(z_nom.flatten() - z_pert.flatten())).item()

    # --- For the rest, we isolate the FIXED Front Camera (Index 0) to avoid wrist ego-motion ---
    z_front_nom = z_nom[0]   # Shape: (196, 384)
    z_front_pert = z_pert[0] # Shape: (196, 384)

    # Calculate per-patch Cosine and L2 distances
    patch_cosine_dists = 1.0 - F.cosine_similarity(z_front_nom, z_front_pert, dim=-1)
    patch_l2_dists = torch.norm(z_front_nom - z_front_pert, dim=-1)

    # 3. Global Pool Average (GAP) Cosine
    # Averages all 196 patches into a single 384-D vector before comparing
    gap_nom = z_front_nom.mean(dim=0, keepdim=True)
    gap_pert = z_front_pert.mean(dim=0, keepdim=True)
    metrics["GAP_Cosine"] = (1.0 - F.cosine_similarity(gap_nom, gap_pert, dim=-1)).item()

    # 4. Mean Patch Cosine
    metrics["Mean_Patch_Cosine"] = patch_cosine_dists.mean().item()

    # 5. Most Divergent Patch (Max Cosine)
    metrics["Max_Patch_Cosine"] = patch_cosine_dists.max().item()

    # 6. Most Divergent Patch (Max L2)
    metrics["Max_Patch_L2"] = patch_l2_dists.max().item()

    # 7. Top-5 Patch Cosine
    metrics["Top5_Patch_Cosine"] = torch.topk(patch_cosine_dists, k=5)[0].mean().item()

    # 8. JENGA ZONE CROP (Top-3 Cosine)
    # Reshape to 14x14 grid
    grid_nom = z_front_nom.view(14, 14, 384)
    grid_pert = z_front_pert.view(14, 14, 384)
    
    # Crop out the top half where the arm hovers (Rows 7-14, Cols 4-10)
    # Note: Adjust these numbers based on your actual camera angle!
    roi_nom = grid_nom[7:14, 4:10, :].reshape(-1, 384)
    roi_pert = grid_pert[7:14, 4:10, :].reshape(-1, 384)
    
    roi_cosine_dists = 1.0 - F.cosine_similarity(roi_nom, roi_pert, dim=-1)
    metrics["Jenga_Zone_Crop_Top3"] = torch.topk(roi_cosine_dists, k=3)[0].mean().item()

    return metrics

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
    print(f"{'METRIC':<30} | {'d(start)':<10} | {'d(end)':<10} | {'LYAPUNOV (λ)':<10}")
    print("="*70)
    
    for metric in d_start.keys():
        ds = d_start[metric] + 1e-8 
        de = d_end[metric] + 1e-8
        
        # Calculate Exponent (Assume T=1 for simplified structural comparison)
        lyap_exp = np.log(de / ds)
        
        print(f"{metric:<30} | {ds:<10.4f} | {de:<10.4f} | {lyap_exp:<10.4f}")
        
    print("="*70)
    print("\nInterpretation Guide:")
    print("- A positive Lyapunov exponent (λ > 0) indicates chaotic divergence (failure).")
    print("- A negative Lyapunov exponent (λ < 0) indicates convergence (recovery/stability).")
    print("- The metric with the highest positive λ is your most sensitive failure detector.")

if __name__ == "__main__":
    main()