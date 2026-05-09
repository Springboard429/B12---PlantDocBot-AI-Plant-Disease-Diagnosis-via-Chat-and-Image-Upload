import pandas as pd

data = [

    {
        "caption": "Tomato_Late_blight",
        "captions": [
            "dark brown lesions on tomato leaf",
            "water soaked spots on tomato leaves",
            "fungal infection causing late blight",
            "tomato leaf with black patches",
            "infected tomato leaf turning brown"
        ]
    },

    {
        "caption": "Tomato_Early_blight",
        "captions": [
            "target shaped brown spots on tomato leaf",
            "alternaria infection on tomato plant",
            "yellowing tomato leaf with concentric rings",
            "dry brown lesions on tomato leaf",
            "fungal early blight symptoms"
        ]
    },

    {
        "caption": "Tomato_Leaf_Mold",
        "captions": [
            "yellow spots and mold under tomato leaf",
            "leaf mold disease in tomato plant",
            "fungal mold growth on tomato leaves",
            "tomato leaf turning yellow with mold",
            "olive green mold under leaf"
        ]
    },

    {
        "caption": "Tomato_Tomato_mosaic_virus",
        "captions": [
            "mosaic patterns on tomato leaf",
            "distorted tomato leaves with virus",
            "green yellow mosaic symptoms",
            "tomato mosaic virus infection",
            "curling and mosaic discoloration"
        ]
    },

    {
        "caption": "Tomato_Tomato_YellowLeaf_Curl_Virus",
        "captions": [
            "yellow curling tomato leaves",
            "leaf curl virus symptoms",
            "tomato leaves curling upward",
            "yellow leaf curl infection",
            "virus infected curled tomato leaf"
        ]
    },

    {
        "caption": "Tomato_Bacterial_spot",
        "captions": [
            "small black bacterial spots",
            "bacterial infection on tomato leaf",
            "dark tiny lesions on tomato leaves",
            "yellow halo around spots",
            "xanthomonas bacterial spot disease"
        ]
    },

    {
        "caption": "Tomato_Septoria_leaf_spot",
        "captions": [
            "small circular brown spots",
            "septoria infection on tomato leaf",
            "yellowing around tiny lesions",
            "septoria leaf spot symptoms",
            "fungal spotting disease"
        ]
    },

    {
        "caption": "Tomato_Spider_mites_Two_spotted_spider_mite",
        "captions": [
            "tiny insects under tomato leaf",
            "yellow stippling on leaves",
            "spider mites infestation",
            "webbing under tomato leaves",
            "two spotted spider mite damage"
        ]
    },

    {
        "caption": "Tomato_Target_Spot",
        "captions": [
            "dark circular target spots",
            "target spot fungal disease",
            "concentric lesions on tomato leaves",
            "brown target lesions",
            "fungal target spot symptoms"
        ]
    },

    {
        "caption": "Tomato_healthy",
        "captions": [
            "healthy green tomato leaf",
            "fresh tomato plant leaves",
            "disease free tomato leaf",
            "bright green healthy tomato plant",
            "normal tomato foliage"
        ]
    },

    {
        "caption": "Potato_Early_blight",
        "captions": [
            "brown concentric lesions on potato leaf",
            "early blight on potato plant",
            "alternaria solani infection",
            "yellowing potato leaf with spots",
            "dry brown patches on potato leaves"
        ]
    },

    {
        "caption": "Potato_Late_blight",
        "captions": [
            "water soaked potato lesions",
            "late blight infection on potato",
            "dark wet patches on potato leaves",
            "fungal potato blight symptoms",
            "rotting potato foliage"
        ]
    },

    {
        "caption": "Potato_healthy",
        "captions": [
            "healthy potato leaf",
            "green potato foliage",
            "fresh disease free potato plant",
            "normal potato leaves",
            "healthy potato crop"
        ]
    },

    {
        "caption": "Pepper__bell___Bacterial_spot",
        "captions": [
            "black bacterial lesions on pepper leaf",
            "pepper leaf bacterial infection",
            "tiny dark spots on bell pepper",
            "yellow halo bacterial spots",
            "infected pepper leaf"
        ]
    },

    {
        "caption": "Pepper__bell___healthy",
        "captions": [
            "healthy bell pepper leaf",
            "green healthy pepper plant",
            "fresh pepper foliage",
            "disease free pepper leaf",
            "normal bell pepper plant"
        ]
    }

]

df = pd.DataFrame(data)

df.to_parquet("plantvillage.parquet")

print("✅ plantvillage.parquet created")
print(df.head())