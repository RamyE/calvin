from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

if __name__ == "__main__":
    parser = ArgumentParser(description="Interactive visualization of CALVIN dataset")
    parser.add_argument("path", type=str, help="Path to dir containing scene_info.npy")
    parser.add_argument("-d", "--data", nargs="*", default=["rgb_static", "rgb_gripper"], help="Data to visualize")
    args = parser.parse_args()

    if not Path(args.path).is_dir():
        print(f"Path {args.path} is either not a directory, or does not exist.")
        exit()

    # check if it is the training dataset, in which case there will be scene_info.npy
    if Path(f"{args.path}/scene_info.npy").is_file():
        indices = next(iter(np.load(f"{args.path}/scene_info.npy", allow_pickle=True).item().values()))
        indices = list(range(indices[0], indices[1] + 1))
    else:
        # if there is no scene_info, we will need to look for the indices in the filenames
        # the format should be episode_{index}.npz
        indices = [int(str(f.name).split("_")[1].split(".")[0]) for f in Path(args.path).iterdir() if f.is_file() and f.name.endswith(".npz")]
        print("No scene_info.npy found, using indices from filenames.")

    annotations = np.load(f"{args.path}/lang_annotations/auto_lang_ann.npy", allow_pickle=True).item()
    annotations = list(zip(annotations["info"]["indx"], annotations["language"]["ann"]))

    idx = 205 # 0
    ann_idx = -1

    while True:
        try:
            try:
                t = np.load(f"{args.path}/episode_{indices[idx]:07d}.npz", allow_pickle=True)
            except FileNotFoundError:
                # print(f"Transition with index {indices[idx]} cannot be found.")
                idx = (idx + 1) % len(indices)
                continue

            for d in args.data:
                if d not in t:
                    print(f"Data {d} cannot be found in transition with index {indices[idx]}")
                    continue
                else:
                    print(f"Data {d} found in transition with index {indices[idx]}")
                
                if d.startswith("rgb") or d.startswith("depth"):
                    if not "tactile" in d:
                        if d.startswith("depth"):
                            normalized_depth_image = cv2.normalize(t[d], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                            cv2.imshow(d, normalized_depth_image)
                        else:
                            cv2.imshow(d, t[d][:, :, ::-1])
                    else:
                        if d.startswith("rgb"):
                            tactile_left = t[d][:, :, :3]
                            tactile_right = t[d][:, :, 3:]
                            cv2.imshow(d, np.concatenate([tactile_left, tactile_right], axis=1))
                        elif d.startswith("depth"):
                            tactile_left = t[d][:, :, :1]
                            tactile_right = t[d][:, :, 1:]
                            # remove the last dimension for each
                            tactile_left = tactile_left.squeeze()
                            tactile_right = tactile_right.squeeze()
                            print(tactile_left.shape, tactile_right.shape)
                            # count the number of pixels that are not 0
                            print(np.count_nonzero(tactile_left), np.count_nonzero(tactile_right))
                            normalized_depth_image = cv2.normalize(np.concatenate([tactile_left, tactile_right], axis=1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                            cv2.imshow(d, normalized_depth_image)
                elif d.startswith("seg"):
                    # the segmentation image only has numbers, we need to assign each number to a different color before displaying it
                    seg = t[d]
                    # import numpy as np
                    # import matplotlib.pyplot as plt
                    # import matplotlib.colors as mcolors
                    # import matplotlib.ticker as ticker

                    # # Define the boundaries for each color segment
                    # boundaries = np.arange(-0.5, len(set(seg.flatten())), 1)  # -0.5 and 13.5 are the outer edges; 0-12 are the segment values

                    # # Create a color map and a normalization instance
                    # cmap = plt.get_cmap('tab20', np.max(seg)-np.min(seg)+1)
                    # norm = mcolors.BoundaryNorm(boundaries, cmap.N)

                    # # Display the image
                    # fig, ax = plt.subplots(figsize=(6, 6))
                    # cax = ax.imshow(seg, cmap=cmap, norm=norm, interpolation='nearest')

                    # # Create the color bar
                    # cbar = fig.colorbar(cax, ticks=np.arange(0, len(set(seg.flatten()))), spacing='proportional')
                    # cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Ensure integer ticks
                    # cbar.set_label('Segment Value')

                    # plt.show()

                    print(set(seg.flatten()))
                    # color_map = np.zeros((256, 3), dtype=np.uint8)
                    # for i in range(256):
                    #     color_map[i] = [(i+2)**6, (i+4)*2, (i+3)**3]
                    cmap = plt.get_cmap('tab20')
                    norm = mcolors.Normalize(vmin=0, vmax=20)

                    rgb_array = cmap(norm(seg))
                    cv2.imshow(d, rgb_array[..., :3])
                    
            
            for n, ((low, high), ann) in enumerate(annotations):
                if indices[idx] >= low and indices[idx] <= high:
                    if n != ann_idx:
                        print(f"{ann}")
                        ann_idx = n

            key = cv2.waitKey(0)
            if key == ord("q"):
                break
            elif key == 83 or key == ord("d"):  # Right arrow
                idx = (idx + 5) % len(indices)
            elif key == 81 or key == ord("a"):  # Left arrow
                idx = (len(indices) + idx - 5) % len(indices)
            else:
                print(f'Unrecognized keycode "{key}"')
        except KeyboardInterrupt:
            break