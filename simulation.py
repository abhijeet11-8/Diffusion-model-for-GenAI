import numpy as np
import matplotlib.pyplot as plt

def Noising(data_tensor, num_steps, theta=4):
    """Adds logistic noise for multiple time steps."""
    x = data_tensor.copy()
    for _ in range(num_steps):
        x = theta * x * (1 - x)
    return x

def Reconstruction(data_tensor, num_steps, theta=4):
    """
    Approximates the inverse of the logistic map.
    Each pixel has two possible parents at every step, leading to 2^T paths.
    We instead compute both branches at each step, count occurrences of resulting
    pixel intensities, and pick the most frequent.
    """
    x_current = data_tensor.copy()
    shape = x_current.shape
    freq = np.zeros((256,) + shape)  # store counts for each possible pixel value

    for _ in range(num_steps):
        y = x_current
        inside = 0.25 - y / theta
        inside = np.clip(inside, 0, None)

        x_plus = 0.5 + np.sqrt(inside)
        x_minus = 0.5 - np.sqrt(inside)

        # convert to integer pixel values 0–255
        x_plus_int = np.clip((x_plus * 255).astype(int), 0, 255)
        x_minus_int = np.clip((x_minus * 255).astype(int), 0, 255)

        for i in range(shape[0]):
            for j in range(shape[1]):
                freq[x_plus_int[i, j], i, j] += 1
                freq[x_minus_int[i, j], i, j] += 1

        # Next iteration: use average of both branches (simplified evolution)
        x_current = (x_plus + x_minus) / 2.0

    x_reconstructed = np.argmax(freq, axis=0)
    return x_reconstructed / 255.0  # normalize back to [0,1]

if __name__ == "__main__":
    # Define a 3x3 binary image
    image = np.array([
        [255, 255, 255],
        [0, 255, 0],
        [255, 0, 0]
    ], dtype=float)

    # Normalize to [0,1]
    image /= image.max()
    print(image)

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original")

    # Add logistic noise
    noisy_image = Noising(image, num_steps=5, theta=4)
    print(noisy_image)
    plt.subplot(1, 3, 2)
    plt.imshow(noisy_image, cmap='gray')
    plt.title("Noisy (t=5)")

    # Reconstruct
    reconstructed = Reconstruction(noisy_image, num_steps=5, theta=4)
    print(reconstructed)
    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed, cmap='gray')
    plt.title("Reconstructed")

    plt.tight_layout()
    plt.show()