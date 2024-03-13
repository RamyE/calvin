import numpy as np

# Load the .npz file
npz_file = np.load('/mnt/vol1/ramy/calvin/dataset/task_D_D/training/episode_0297922.npz')
npz_file_2 = np.load('/mnt/vol1/ramy/calvin/dataset/task_D_D_new/training/episode_0297922.npz')
# Iterate over items and print their sizes
for key in npz_file_2:
    try:
        array1 = npz_file[key]
        array2 = npz_file_2[key]
        print(f"Array '{key}':")
        print(f" - Number of elements: {array1.size} vs {array2.size}")
        print(f" - Memory size (bytes): {array1.nbytes} vs {array2.nbytes}")
        print(f" - Shape: {array1.shape} vs {array2.shape}")
        print(f" - Data type: {array1.dtype} vs {array2.dtype}")
    except KeyError:
        print(f"Key '{key}' not found in npz_file.npz file.")
        array = npz_file_2[key]
        print(f"Array '{key}':")
        print(f" - Number of elements: {array.size}")
        print(f" - Memory size (bytes): {array.nbytes}")
        print(f" - Shape: {array.shape}")
        print(f" - Data type: {array.dtype}")

npz_file.close()
npz_file_2.close()
