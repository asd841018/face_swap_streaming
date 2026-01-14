import cv2
import numpy as np
from scipy.interpolate import Rbf


def get_face_control_points_106(landmarks: np.ndarray):
    """
    Get face control points for deformation from InsightFace 106 landmarks
    
    Args:
        landmarks: (106, 2) numpy array, each row is (x, y) pixel coordinates
    
    Returns:
        src_points: Control point coordinates
        indices: Control point indices
        left_cheek_indices: Left cheek indices
        right_cheek_indices: Right cheek indices
        chin_indices: Chin indices
    """
    # Right cheek control points
    RIGHT_CHEEK = [1, 9, 10, 11, 12, 13, 14, 15, 16]
    # Left cheek control points
    LEFT_CHEEK = [17, 25, 26, 27, 28, 29, 30, 31, 32]
    # Chin control points
    CHIN = [2, 3, 4, 5, 6, 7, 8, 0, 24, 23, 22, 21, 20, 19, 18]
    # Nose (fixed)
    NOSE = [73, 74, 76, 77, 78, 79, 80, 85, 84, 83, 82, 86]
    # Eyes (fixed)
    RIGHT_EYE = [35, 36, 33, 37, 39, 42, 40, 41, 38]
    LEFT_EYE = [93, 96, 94, 95, 89, 90, 87, 91, 88]
    # Mouth (fixed)
    MOUTH = [71, 63, 64, 52, 55, 56, 53, 59, 58, 61, 68, 67, 62, 66, 65, 54, 60, 57, 69, 70]
    # Eyebrows (fixed)
    RIGHT_EYEBROWS = [43, 48, 49, 51, 50, 46, 47, 45, 44]
    LEFT_EYEBROWS = [101, 105, 104, 103, 102, 97, 98, 99, 100]
    # Between eyes (fixed)
    BETWEEN_EYES = [75, 72, 81]
    
    # Combine all control points
    all_indices = (LEFT_CHEEK + RIGHT_CHEEK + CHIN + 
                   NOSE + LEFT_EYE + RIGHT_EYE + MOUTH + RIGHT_EYEBROWS + LEFT_EYEBROWS + BETWEEN_EYES)
    
    # Remove duplicates
    all_indices = list(set(all_indices))
    all_indices.sort()
    
    src_points = landmarks[all_indices].copy()
    
    return src_points, all_indices, LEFT_CHEEK, RIGHT_CHEEK, CHIN


def apply_slim_deformation_106(src_points: np.ndarray, 
                                indices: list,
                                left_cheek_indices: list,
                                right_cheek_indices: list, 
                                chin_indices: list,
                                landmarks: np.ndarray,
                                cheek_strength: float = 0.15,
                                chin_strength: float = 0.10):
    """
    Apply slim face deformation
    
    Args:
        src_points: Control point coordinates
        indices: Control point indices
        left_cheek_indices: Left cheek indices
        right_cheek_indices: Right cheek indices
        chin_indices: Chin indices
        landmarks: Complete 106 landmarks
        cheek_strength: Cheek inward offset strength (0~1)
        chin_strength: Chin upward contraction strength (0~1)
    
    Returns:
        dst_points: Deformed control point coordinates
    """
    dst_points = src_points.copy()
    
    # Get nose tip as center reference point (landmark 86)
    nose_center = landmarks[86]
    nose_x, nose_y = nose_center[0], nose_center[1]
    
    # Get lowest chin point (landmark 0 is the lowest chin center point)
    chin_bottom = landmarks[0]
    chin_bottom_y = chin_bottom[1]
    
    # Calculate face width for normalization
    face_width = np.abs(landmarks[1][0] - landmarks[17][0])  # Right temple to left temple
    face_height = np.abs(landmarks[0][1] - landmarks[49][1])  # Chin to forehead
    
    # Offset left cheek points (toward nose)
    for idx in left_cheek_indices:
        if idx in indices:
            orig_idx = indices.index(idx)
            x, y = src_points[orig_idx]
            
            # Calculate vector toward nose
            dx = nose_x - x
            dy = nose_y - y
            
            # Adjust strength based on distance from nose (farther distance = stronger effect)
            distance = np.sqrt(dx**2 + dy**2)
            distance_factor = distance / face_width if face_width > 0 else 0
            
            # Adjust strength based on y coordinate (middle cheek area has stronger effect)
            # Middle cheek position is approximately near the nose
            y_factor = 1.0 - abs(y - nose_y) / (face_height * 0.5) if face_height > 0 else 1.0
            y_factor = max(0.3, min(1.0, y_factor))
            
            strength = cheek_strength * (0.5 + 0.5 * distance_factor) * y_factor
            
            dst_points[orig_idx][0] = x + dx * strength
            dst_points[orig_idx][1] = y + dy * strength * 0.3  # Smaller Y direction offset
    
    # Offset right cheek points (toward nose)
    for idx in right_cheek_indices:
        if idx in indices:
            orig_idx = indices.index(idx)
            x, y = src_points[orig_idx]
            
            dx = nose_x - x
            dy = nose_y - y
            
            distance = np.sqrt(dx**2 + dy**2)
            distance_factor = distance / face_width if face_width > 0 else 0
            
            y_factor = 1.0 - abs(y - nose_y) / (face_height * 0.5) if face_height > 0 else 1.0
            y_factor = max(0.3, min(1.0, y_factor))
            
            strength = cheek_strength * (0.5 + 0.5 * distance_factor) * y_factor
            
            dst_points[orig_idx][0] = x + dx * strength
            dst_points[orig_idx][1] = y + dy * strength * 0.3
    
    # Contract chin points upward
    for idx in chin_indices:
        if idx in indices:
            orig_idx = indices.index(idx)
            x, y = src_points[orig_idx]
            
            # Offset upward and toward center
            dy = chin_bottom_y - y
            dx = nose_x - x
            
            # Adjust strength based on distance from lowest chin point
            distance_factor = abs(dy) / face_height if face_height > 0 else 0
            strength = chin_strength * (1.0 - distance_factor * 0.5)
            
            # Contract chin upward
            dst_points[orig_idx][0] = x + dx * strength * 0.2
            dst_points[orig_idx][1] = y - abs(y - nose_y) * strength * 0.15
    
    return dst_points


def warp_face_rbf(img: np.ndarray, 
                  src_points: np.ndarray, 
                  dst_points: np.ndarray,
                  grid_resolution: int = 50):
    """
    Fast grid deformation using RBF (Radial Basis Function)
    
    Args:
        img: Input image
        src_points: Original control points
        dst_points: Target control points
        grid_resolution: Grid resolution (smaller = faster)
    
    Returns:
        warped: Warped image
    """
    h, w = img.shape[:2]
    
    # Add boundary control points to fix image edges
    margin = max(40, min(w, h) // 10)
    border_points = []
    
    # Four corners
    border_points.extend([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    
    # Border points
    for x in range(margin, w-margin, margin):
        border_points.extend([[x, 0], [x, h-1]])
    for y in range(margin, h-margin, margin):
        border_points.extend([[0, y], [w-1, y]])
    
    border_points = np.array(border_points, dtype=np.float32)
    
    # Combine face control points and border points
    all_src_points = np.vstack([src_points, border_points])
    all_dst_points = np.vstack([dst_points, border_points])
    
    # Build RBF interpolation (thin_plate function)
    try:
        rbf_x = Rbf(all_dst_points[:, 0], all_dst_points[:, 1], all_src_points[:, 0],
                    function='thin_plate', smooth=0)
        rbf_y = Rbf(all_dst_points[:, 0], all_dst_points[:, 1], all_src_points[:, 1],
                    function='thin_plate', smooth=0)
    except Exception as e:
        # If RBF fails, return original image
        return img
    
    # Use downsampled grid to accelerate computation
    grid_w = max(1, w // grid_resolution)
    grid_h = max(1, h // grid_resolution)
    
    grid_x_small = np.linspace(0, w-1, grid_w)
    grid_y_small = np.linspace(0, h-1, grid_h)
    grid_x_small, grid_y_small = np.meshgrid(grid_x_small, grid_y_small)
    
    # Calculate deformation on low resolution grid
    map_x_small = rbf_x(grid_x_small, grid_y_small).astype(np.float32)
    map_y_small = rbf_y(grid_x_small, grid_y_small).astype(np.float32)
    
    # Interpolate upscale to original resolution
    map_x = cv2.resize(map_x_small, (w, h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(map_y_small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Execute remapping
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    
    return warped

def warp_face_rbf_cropped(img: np.ndarray, 
                          src_points: np.ndarray, 
                          dst_points: np.ndarray,
                          bbox: np.ndarray,
                          grid_resolution: int = 50,
                          padding: float = 0.2):
    """
    只處理 bounding box 內的 RBF 變形
    
    Args:
        img: Input image
        src_points: Original control points (絕對座標)
        dst_points: Target control points (絕對座標)
        bbox: Bounding box [x1, y1, x2, y2]
        grid_resolution: Grid resolution
        padding: Bounding box 外擴比例
    
    Returns:
        warped: Warped image
    """
    h, w = img.shape[:2]
    
    # 計算帶 padding 的 bounding box
    x1, y1, x2, y2 = bbox[:4].astype(int)
    box_w, box_h = x2 - x1, y2 - y1
    pad_x, pad_y = int(box_w * padding), int(box_h * padding)
    
    # 擴展並限制在圖片範圍內
    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(w, x2 + pad_x)
    crop_y2 = min(h, y2 + pad_y)
    
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    
    # 將控制點轉換為裁剪區域的局部座標
    src_local = src_points.copy()
    dst_local = dst_points.copy()
    src_local[:, 0] -= crop_x1
    src_local[:, 1] -= crop_y1
    dst_local[:, 0] -= crop_x1
    dst_local[:, 1] -= crop_y1
    
    # 只在裁剪區域邊界加固定點
    margin = max(10, min(crop_w, crop_h) // 8)
    border_points = []
    
    # 四角
    border_points.extend([[0, 0], [crop_w-1, 0], [crop_w-1, crop_h-1], [0, crop_h-1]])
    
    # 邊界點
    for x in range(margin, crop_w-margin, margin): 
        border_points.extend([[x, 0], [x, crop_h-1]])
    for y in range(margin, crop_h-margin, margin):
        border_points.extend([[0, y], [crop_w-1, y]])
    
    border_points = np.array(border_points, dtype=np.float32)
    
    all_src = np.vstack([src_local, border_points])
    all_dst = np.vstack([dst_local, border_points])
    
    # 建立 RBF
    try:
        rbf_x = Rbf(all_dst[:, 0], all_dst[:, 1], all_src[:, 0], function='thin_plate', smooth=0)
        rbf_y = Rbf(all_dst[:, 0], all_dst[:, 1], all_src[:, 1], function='thin_plate', smooth=0)
    except Exception:
        return img
    
    # 計算小網格（相對於裁剪區域更小）
    grid_w = max(1, crop_w // grid_resolution)
    grid_h = max(1, crop_h // grid_resolution)
    
    grid_x_small = np.linspace(0, crop_w-1, grid_w)
    grid_y_small = np.linspace(0, crop_h-1, grid_h)
    grid_x_small, grid_y_small = np.meshgrid(grid_x_small, grid_y_small)
    
    map_x_small = rbf_x(grid_x_small, grid_y_small).astype(np.float32)
    map_y_small = rbf_y(grid_x_small, grid_y_small).astype(np.float32)
    
    map_x = cv2.resize(map_x_small, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(map_y_small, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    
    # 只對裁剪區域做 remap
    crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    warped_crop = cv2.remap(crop_img, map_x, map_y, 
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
    
    # 貼回原圖
    result = img.copy()
    result[crop_y1:crop_y2, crop_x1:crop_x2] = warped_crop
    
    return result


def reshape_face(img: np.ndarray,
                 landmarks: np.ndarray,
                 bbox: np.ndarray = None,
                 cheek_strength: float = 0.15,
                 chin_strength: float = 0.10,
                 grid_resolution: int = 50) -> np.ndarray:
    """
    Main function: Apply face slimming effect
    
    Args:
        img: Input image (BGR or RGB)
        landmarks: InsightFace 106 landmarks, shape is (106, 2)
        cheek_strength: Cheek contraction strength (0~1), larger = slimmer cheeks
        chin_strength: Chin contraction strength (0~1), larger = sharper chin
        grid_resolution: Grid resolution (20-80), smaller = faster but lower precision
    
    Returns:
        reshaped: Face-slimmed image
    """
    if landmarks is None or len(landmarks) != 106:
        return img
    
    # Ensure landmarks are float32
    landmarks = landmarks.astype(np.float32)
    
    # 1. Get control points
    src_points, indices, left_cheek, right_cheek, chin = get_face_control_points_106(landmarks)
    
    # 2. Calculate deformed target points
    dst_points = apply_slim_deformation_106(
        src_points, indices, left_cheek, right_cheek, chin,
        landmarks, cheek_strength, chin_strength
    )

    # 3. Apply RBF deformation
    if bbox is not None:
        reshaped = warp_face_rbf_cropped(
            img, src_points, dst_points, bbox,
            grid_resolution, padding=0.2
        )
    else:
        reshaped = warp_face_rbf(img, src_points, dst_points, grid_resolution)
    
    return reshaped


def reshape_faces(img: np.ndarray,
                  faces: list,
                  cheek_strength: float = 0.15,
                  chin_strength: float = 0.10,
                  grid_resolution: int = 50) -> np.ndarray:
    """
    Apply face slimming to all faces in the image
    
    Args:
        img: Input image (BGR or RGB)
        faces: List of faces detected by InsightFace
        cheek_strength: Cheek contraction strength (0~1)
        chin_strength: Chin contraction strength (0~1)
        grid_resolution: Grid resolution
    
    Returns:
        reshaped: Face-slimmed image
    """
    result = img.copy()
    
    for face in faces:
        landmarks = face.get("landmark_2d_106")
        bbox = face.get("bbox")
        if landmarks is not None and len(landmarks) == 106:
            # result = reshape_face(
            #     img=result,
            #     landmarks=landmarks,
            #     bbox=None,
            #     cheek_strength=cheek_strength, chin_strength=chin_strength, grid_resolution=grid_resolution
            # )
            result = reshape_face(
                result, landmarks, bbox,
                cheek_strength, chin_strength, grid_resolution
            )
    
    return result