import os
import numpy as np
import pickle

# مسار الملفات
tokenizer_dir = os.path.join("C:/Mennah/semster 8/deep learning/assigments/video captioning/main_project", "tokenizer")

# تحميل tokenizer
with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

# تحميل video_caption_pairs
video_caption_pairs = np.load(os.path.join(tokenizer_dir, "video_caption_pairs.npy"), allow_pickle=True)

# تحميل max_length
with open(os.path.join(tokenizer_dir, "max_length.txt"), "r") as f:
    max_length = int(f.read())

print(f"✅ عدد الأزواج: {len(video_caption_pairs)}")
print(f"✅ max_length: {max_length}")
print(f"✅ عدد كلمات tokenizer: {len(tokenizer.word_index)}")
