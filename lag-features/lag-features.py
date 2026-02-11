def lag_features(series, lags):
    """
    Tạo ma trận lag feature từ chuỗi thời gian.
    Chỉ bao gồm các bước thời gian mà tất cả các độ trễ đều khả dụng.
    """
    max_lag = max(lags)
    result = []

    # Bắt đầu từ t = max_lag để đảm bảo có đủ dữ liệu quá khứ cho mọi lag
    for t in range(max_lag, len(series)):
        row = []
        # Với mỗi thời điểm t, lấy giá trị tại các độ trễ tương ứng
        for lag in lags:
            row.append(series[t - lag])
        result.append(row)
        
    return result