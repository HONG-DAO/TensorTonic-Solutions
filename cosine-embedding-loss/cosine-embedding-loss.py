def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    dot_product = sum(a*b for a,b in zip(x1,x2))

    norm_x1 = math.sqrt(sum(a*a for a in x1))
    norm_x2 = math.sqrt(sum(b*b for b in x2))
    
    if norm_x1 == 0 or norm_x2 ==0:
        cos = 0.0
    else:
        cos = dot_product/(norm_x1*norm_x2)

    if label == 1:
        loss = 1 - cos
    elif label == -1:
        loss = max(0, cos - margin)
    else:
        raise ValueError("Label must be 1 or -1")

    return loss