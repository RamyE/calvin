from pathlib import Path
import numpy as np

DATASET_PATH = Path("/mnt/vol1/ramy/calvin/dataset/task_A_D/training")
ORIG_LANGUAGE_EMBEDDINGS_PATH = DATASET_PATH / "lang_annotations" / "auto_lang_ann.npy"
NEW_LANGUAGE_EMBEDDINGS_PATH = DATASET_PATH / "lang_one_hot" / "auto_lang_ann.npy"
#create the new directory if it doesn't exist
NEW_LANGUAGE_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

VALIDATION_DATASET_PATH = Path("/mnt/vol1/ramy/calvin/dataset/task_A_D/validation")
ORIG_LANGUAGE_EMBEDDINGS_PATH_VAL = VALIDATION_DATASET_PATH / "lang_annotations" / "auto_lang_ann.npy"
NEW_LANGUAGE_EMBEDDINGS_PATH_VAL = VALIDATION_DATASET_PATH / "lang_one_hot" / "auto_lang_ann.npy"
NEW_LANGUAGE_EMBEDDINGS_PATH_VAL.parent.mkdir(parents=True, exist_ok=True)


VALIDATION_DATASET_PATH = Path("/mnt/vol1/ramy/calvin/dataset/task_A_D/validation")
ORIG_LANGUAGE_EMBEDDINGS_PATH_VALIDATION = VALIDATION_DATASET_PATH / "lang_annotations" / "embeddings.npy"
NEW_LANGUAGE_EMBEDDINGS_PATH_VALIDATION = VALIDATION_DATASET_PATH / "lang_one_hot" / "embeddings.npy"




# do the work for the training dataset

lang_data = np.load(ORIG_LANGUAGE_EMBEDDINGS_PATH, allow_pickle=True).item()

ann = lang_data['language']['ann']
emb = lang_data['language']['emb']
task = lang_data['language']['task']

task_embs_dict = {}
for i, t in enumerate(task):
    if t not in task_embs_dict.keys():
        task_embs_dict[t] = [emb[i][0]]
    else:
        task_embs_dict[t].append(emb[i][0])
        
        
# create one hot vector for each key in task_embs_dict
one_hot_task_embs = {}
for k, v in task_embs_dict.items():
    one_hot_task_embs[k] = np.eye(len(task_embs_dict.keys()))[list(task_embs_dict.keys()).index(k)]

# update lang_data with the new embeddings
# The shape should be (1, one_hot_vector_size)
new_emb = []
for i, emb in enumerate(lang_data['language']['emb']):
    new_emb.append([one_hot_task_embs[task[i]]])

lang_data['language']['emb'] = np.array(new_emb)

# save the new embeddings
np.save(NEW_LANGUAGE_EMBEDDINGS_PATH, lang_data)


# do the work for the validation dataset - part 1

lang_data = np.load(ORIG_LANGUAGE_EMBEDDINGS_PATH_VAL, allow_pickle=True).item()

ann = lang_data['language']['ann']
emb = lang_data['language']['emb']
task = lang_data['language']['task']

# update lang_data with the new embeddings
# The shape should be (1, one_hot_vector_size)
new_emb = []
for i, emb in enumerate(lang_data['language']['emb']):
    new_emb.append([one_hot_task_embs[task[i]]])

lang_data['language']['emb'] = np.array(new_emb)

# save the new embeddings
np.save(NEW_LANGUAGE_EMBEDDINGS_PATH_VAL, lang_data)




# do the work for the validation dataset - part 2
lang_data_val = np.load(ORIG_LANGUAGE_EMBEDDINGS_PATH_VALIDATION, allow_pickle=True).item()
for k, v in lang_data_val.items():
    # print(k, v['emb'].shape)    
    lang_data_val[k]['emb'] = np.array([[one_hot_task_embs[k]]])

# save the new embeddings
np.save(NEW_LANGUAGE_EMBEDDINGS_PATH_VALIDATION, lang_data_val)



