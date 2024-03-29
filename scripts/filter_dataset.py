from pathlib import Path
import numpy as np

DATASET_PATH = Path("/mnt/vol1/ramy/calvin/dataset/task_B_D/")
START_LIM = 0
END_LIM = 598909

# loop over all the folders in DATASET_PATH that start with "lang_"
for data_type in ["training"]:
    dataset = DATA_PATH / data_type
    for lang_folder in dataset.glob("lang_*"):
        print(lang_folder)
        # get the language embeddings file
        lang_path = lang_folder / "auto_lang_ann.npy"
        lang_data = np.load(lang_path, allow_pickle=True).item()

        ann = lang_data['language']['ann'] # list of strings
        emb = lang_data['language']['emb'] # numpy array of shape (num_episodes, 1, embedding_size)
        task = lang_data['language']['task'] # list of strings
        indx = lang_data['info']['indx'] # list of tuples

        # we will go over every item in the indx list and check if it should be filtered out or not
        # if we are to delete it then we have to delete the corresponding entry in the ann, emb, and task lists
        new_indx = []
        new_ann = []
        new_emb = []
        new_task = []
        for i, (start, end) in enumerate(indx):
            # check if the episode is to be deleted
            if (START_LIM <= start <= END_LIM or START_LIM <= end <= END_LIM):
                new_indx.append((start, end))
                new_ann.append(ann[i])
                new_emb.append(emb[i])
                new_task.append(task[i])
        new_emb = np.array(new_emb)
        lang_data['language']['ann'] = new_ann
        lang_data['language']['emb'] = new_emb
        lang_data['language']['task'] = new_task
        lang_data['info']['indx'] = new_indx
        
        print(f"Number of episodes in {lang_folder} before filtering: {len(indx)}")
        print(f"Number of episodes in {lang_folder} after filtering: {len(new_indx)}")
        print(lang_data['language']['emb'].shape)
        # save the new embeddings
        np.save(lang_path, lang_data)
        
# filter out the episodes in the training dataset in files ep_lens.npy and ep_start_end_ids.npy
for data_type in ["training"]:
    dataset = DATASET_PATH / data_type
    ep_lens = np.load(dataset / "ep_lens.npy")
    ep_start_end_ids = np.load(dataset / "ep_start_end_ids.npy")
    
    new_ep_start_end_ids = []
    for i, (start, end) in enumerate(ep_start_end_ids):
        if (START_LIM <= start <= END_LIM or START_LIM <= end <= END_LIM):
            new_ep_start_end_ids.append([start, end])
    new_ep_lens = ep_lens[:len(new_ep_start_end_ids)]
    new_ep_start_end_ids = np.array(new_ep_start_end_ids)
    
    print(f"Number of episodes in {data_type} before filtering: {ep_lens.shape}")
    print(f"Number of episodes in {data_type} after filtering: {ep_lens.shape}")
    
    np.save(DATASET_PATH / "ep_lens.npy", new_ep_lens)
    np.save(DATASET_PATH / "ep_start_end_ids.npy", new_ep_start_end_ids)
                
            
                