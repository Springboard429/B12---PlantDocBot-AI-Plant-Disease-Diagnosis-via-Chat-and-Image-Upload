import pandas as pd
import random

# ✅ EXACT 15 CLASSES (MATCH CNN)
classes = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato_Early_blight",
    "Potato_healthy",
    "Potato_Late_blight",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_Tomato_YellowLeaf_Curl_Virus"
]

# ✅ symptom templates (safe NLP augmentation)
templates = [
    "yellow spots on leaves",
    "brown lesions spreading on leaves",
    "leaf curling with visible veins",
    "white mold growth on leaf surface",
    "dark patches on plant leaves",
    "wilting plant symptoms",
    "fungal infection on leaves",
    "leaf decay and drying",
    "small spots with yellow halo",
    "stunted plant growth",
    "leaf discoloration and damage",
    "rapid leaf damage spreading",
    "tiny mites under leaves",
    "bacterial infection signs on leaf"
]

data = []

# ✅ generate ~25 samples per class (15 x 25 = 375 rows)
for disease in classes:
    for _ in range(25):
        text = random.choice(templates) + " related to " + disease.replace("_", " ").lower()
        data.append([text, disease])

# create dataframe
df = pd.DataFrame(data, columns=["text", "disease"])

# shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

# save CSV
df.to_csv("symptoms_dataset.csv", index=False)

print("Dataset created successfully!")
print(df.shape)
print(df["disease"].nunique())
print(df["disease"].value_counts())