import os
import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

# مسارات الملفات
train_videos_path = r"C:\Mennah\semster 8\deep learning\assigments\video captioning\main_project\dataset\train"
json_path = r"C:\Mennah\semster 8\deep learning\assigments\video captioning\main_project\captions_train.json"
features_path = r"C:\Mennah\semster 8\deep learning\assigments\video captioning\main_project\features"

# قراءة JSON
with open(json_path, 'r', encoding='utf-8') as f:
    captions_data = json.load(f)

print("✅ عدد عناصر JSON:", len(captions_data))
print("🔸 نوع البيانات:", type(captions_data))
print("🔸 مثال أول عنصر:", captions_data[0])

# إعداد البيانات: {'video_id': [captions...]}
captions_dict = {}
for item in captions_data:
    video_id = item['id']
    captions = [cap.strip() for cap in item['caption']]
    captions_dict[video_id] = captions

# إعداد الكابشنات
all_captions = []
video_caption_pairs = []

missing_files = 0
for video_id, caps in captions_dict.items():
    video_id_clean = video_id.replace(".avi", "")  # شيل .avi
    feature_file = os.path.join(features_path, video_id_clean + ".npy")
    if os.path.exists(feature_file):
        for cap in caps:
            caption = "<start> " + cap.lower() + " <end>"
            all_captions.append(caption)
            video_caption_pairs.append((feature_file, caption))
    else:
        missing_files += 1
        print(f"❌ ملف غير موجود: {feature_file}")

print(f"✅ عدد الكابشنز بعد الفلترة: {len(all_captions)}")
if len(all_captions) == 0:
    print("❌ لم يتم العثور على كابشنات صالحة بعد الفلترة! تأكدي من وجود ملفات .npy في المسار الصحيح.")
    exit()

# إعداد التوكنيزر
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(all_captions)

print(f"✅ حجم المفردات: {len(tokenizer.word_index)}")
print(f"🔸 أول 10 كلمات: {list(tokenizer.word_index.items())[:10]}")

# حفظ التوكنيزر
tokenizer_dir = os.path.join("C:/Mennah/semster 8/deep learning/assigments/video captioning/main_project", "tokenizer")
os.makedirs(tokenizer_dir, exist_ok=True)

with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

# تحويل الكابشنات لأرقام
sequences = tokenizer.texts_to_sequences([cap for _, cap in video_caption_pairs])
max_length = max(len(seq) for seq in sequences)
print(f"✅ الطول الأقصى للكابشن: {max_length}")

# حفظ الطول الأقصى
with open(os.path.join(tokenizer_dir, "max_length.txt"), "w") as f:
    f.write(str(max_length))

# حفظ أزواج الفيديو والكابشن
video_caption_pairs_array = np.array(video_caption_pairs, dtype=object)
np.save(os.path.join(tokenizer_dir, "video_caption_pairs.npy"), video_caption_pairs_array)

print(f"✅ تم حفظ tokenizer و max_length و video_caption_pairs ✅")
print(f"🔹 عدد ملفات غير موجودة: {missing_files}")
 