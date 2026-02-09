import numpy as np

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Tính trung bình Binary Focal Loss.
    """
    # Chuyển đổi đầu vào thành numpy array để tính toán vector hóa
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Bước 1: Tính p_t
    # p_t = p nếu y = 1, ngược lại p_t = 1 - p nếu y = 0
    pt = np.where(targets == 1, predictions, 1 - predictions)

    # Bước 2: Tính Focal Loss cho từng mẫu theo công thức:
    # FL = -alpha * (1 - pt)^gamma * ln(pt)
    # Lưu ý: np.log là logarit tự nhiên (ln)
    individual_loss = -alpha * (1 - pt)**gamma * np.log(pt)

    # Bước 3: Trả về giá trị trung bình (mean loss)
    return np.mean(individual_loss)