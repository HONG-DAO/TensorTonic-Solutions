import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    """
    # Write code here
    mean_fake = np.mean(fake_scores)
    mean_real = np.mean(real_scores)

    loss = mean_fake - mean_real

    return float(loss)