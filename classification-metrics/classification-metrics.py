import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Tính toán accuracy, precision, recall, F1 cho phân loại đơn nhãn.
    Hỗ trợ các loại trung bình: 'micro', 'macro', 'weighted', 'binary'.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Lấy danh sách các lớp duy nhất xuất hiện trong cả thực tế và dự đoán
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # 1. Accuracy luôn tính chung cho toàn bộ tập dữ liệu
    accuracy = np.mean(y_true == y_pred)

    if average == "binary":
        # Logic cho phân loại nhị phân dựa trên pos_label
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    elif average == "micro":
        # Với Micro: Precision = Recall = Accuracy trong phân loại đơn nhãn
        precision = recall = f1 = accuracy

    else:
        # Logic cho Macro và Weighted: Tính toán theo từng lớp
        precisions = []
        recalls = []
        f1s = []
        counts = []

        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
            counts.append(np.sum(y_true == cls))

        if average == "macro":
            precision = np.mean(precisions)
            recall = np.mean(recalls)
            f1 = np.mean(f1s)
        elif average == "weighted":
            weights = np.array(counts) / len(y_true)
            precision = np.average(precisions, weights=weights)
            recall = np.average(recalls, weights=weights)
            f1 = np.average(f1s, weights=weights)

    return {
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6)
    }
