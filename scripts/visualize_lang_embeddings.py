from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import random
import hashlib
import matplotlib.colors
import wandb

wandb.init(project="Visualize Language Embeddings")

DATASET_PATH = Path("/mnt/vol1/ramy/calvin/dataset/task_D_D_new2/training")

EMBEDDING_TYPES = [
    "lang_one_hot",
    # "lang_all-distilroberta-v1",
    # "lang_all-MiniLM-L6-v2",
    # "lang_all-mpnet-base-v2",
    # "lang_annotations",
    # "lang_BERT",
    # "lang_clip_resnet50",
    # "lang_clip_ViTB32",
    # "lang_huggingface_distilroberta",
    # "lang_huggingface_mpnet",
    # "lang_msmarco-bert-base-dot-v5",
    # "lang_paraphrase-MiniLM-L3-v2"
]

for EMBEDDING_TYPE in EMBEDDING_TYPES:
    print("Visualizing embeddings for: ", EMBEDDING_TYPE)
    LANGUAGE_EMBEDDINGS_PATH = DATASET_PATH / EMBEDDING_TYPE / "auto_lang_ann.npy"

    lang_data = np.load(LANGUAGE_EMBEDDINGS_PATH, allow_pickle=True).item()['language']

    ann = lang_data['ann']
    emb = lang_data['emb']
    task = lang_data['task']

    task_embs_dict = {}
    for i, t in enumerate(task):
        if t not in task_embs_dict.keys():
            task_embs_dict[t] = [emb[i][0]]
        else:
            task_embs_dict[t].append(emb[i][0])
            
    # print(task_embs_dict.keys())

    # each emnbedding size is 1024, we need to display them so that each embedding beloning to the same task should have the same color
    # we can use tSNE to reduce the dimensionality of the embeddings to 2D and then plot them

    # first we need to reduce the dimensionality of the embeddings to 2D
    tsne = TSNE(n_components=2, random_state=0)
    task_embs = []
    task_labels = []
    for k, v in task_embs_dict.items():
        task_embs.extend(v)
        task_labels.extend([k] * len(v))
    task_embs = np.array(task_embs)
    print(task_embs.shape)
    task_labels = np.array(task_labels)
    task_embs_2d = tsne.fit_transform(task_embs)

    # we can see that the embeddings are not well separated, we can try to use a different method to visualize the embeddings
    # we can use UMAP to visualize the embeddings
    # create UMAP object
    umap_obj = umap.UMAP(n_components=2, random_state=0)
    # fit UMAP to the embeddings
    umap_embs = umap_obj.fit_transform(task_embs)

    def generate_distinct_colors(n):
        colors = []
        np.random.seed(0)  # Ensure reproducibility
        for i in range(n):
            hue = i / n
            saturation = 0.5 + np.random.rand() * 0.5  # Keep saturation between 0.5 and 1 to ensure colors are vibrant
            value = 0.5 + np.random.rand() * 0.5  # Keep value between 0.5 and 1 to ensure colors are not too dark
            colors.append(matplotlib.colors.hsv_to_rgb([hue, saturation, value]))
        return colors
    def hash_task_to_color(task, colors):
        hash_object = hashlib.sha256(task.encode())
        hash_digest = int(hash_object.hexdigest(), 16)
        color_index = hash_digest % len(colors)
        return colors[color_index]
    def generate_colors_for_tasks(tasks):
        num_unique_tasks = len(set(tasks))
        distinct_colors = generate_distinct_colors(num_unique_tasks)
        colors = [hash_task_to_color(task, distinct_colors) for task in tasks]
        print(colors)
        task_colors = {task: color for task, color in zip(tasks, colors)}
        return task_colors
    unique_tasks = list(task_embs_dict.keys())
    task_colors = generate_colors_for_tasks(unique_tasks)

    # display the embeddings using tSNE
    plt.figure(figsize=(20, 16))
    for task in unique_tasks:
        indices = np.where(task_labels == task)
        plt.scatter(task_embs_2d[indices, 0], task_embs_2d[indices, 1], c=[task_colors[task]], label=task, s=6)
    plt.title(f"tSNE visualization of {EMBEDDING_TYPE} embeddings")
    plt.legend()
    wandb.log({f"tSNE visualization of {EMBEDDING_TYPE}": plt})
    # save the plot because the server could be headless
    plt.savefig(f"task_embs_2d_tsne_{EMBEDDING_TYPE}.png")
    # plt.show()

    plt.figure(figsize=(20, 16))
    for task in unique_tasks:
        indices = np.where(task_labels == task)
        plt.scatter(umap_embs[indices, 0], umap_embs[indices, 1], c=[task_colors[task]], label=task, s=6)
    plt.title(f"UMAP visualization of {EMBEDDING_TYPE} embeddings")
    plt.legend()
    # save the plot because the server could be headless
    plt.savefig(f"task_embs_2d_umap_{EMBEDDING_TYPE}.png")
    wandb.log({f"UMAP visualization of {EMBEDDING_TYPE}": plt})
    # plt.show()
    
    print("Done visualizing embeddings for: ", EMBEDDING_TYPE)
