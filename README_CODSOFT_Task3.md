# 🖼️ Task 03 — Image Captioning AI

![CodSoft](https://img.shields.io/badge/CodSoft-AI%20Internship-blue?style=for-the-badge)
![Task](https://img.shields.io/badge/Task-03-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.x-yellow?style=for-the-badge&logo=python)
![CV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP-Caption%20Generation-red?style=for-the-badge)

---

## 📌 Internship Details
| Field | Details |
|-------|---------|
| **Company** | CodSoft |
| **Intern** | Jaspreet Singh |
| **Track** | Artificial Intelligence |
| **Task** | 03 — Image Captioning |

---

## 📝 Task Description
Combine Computer Vision and NLP to build an image captioning AI using pre-trained image recognition models to extract features and generate captions.

---

## ✅ What I Built
A complete CV + NLP pipeline that analyzes images and generates natural language captions describing the scene.

### 🔄 Pipeline
```
Input Image
    ↓
Feature Extraction (OpenCV)
├── Color Analysis (K-Means Clustering)
├── Brightness Detection
├── Edge Density (Canny)
├── Texture Analysis
└── Scene Classification
    ↓
NLG Caption Generator
├── Template Selection by Scene Type
├── Feature-to-Language Mapping
└── Natural Language Caption
    ↓
Output Captions (3 per image)
```

### 🎯 Scene Types Detected
| Scene | Example | Caption Style |
|-------|---------|--------------|
| Nature/Outdoor | Parks, forests | Natural tones |
| Sky/Water | Beaches, clouds | Cool, serene |
| Urban/City | Buildings, roads | Detailed, complex |
| Minimalist | Simple objects | Clean, minimal |

### 📝 Sample Captions Generated
- *"A bright outdoor scene featuring green and yellow tones with complex texture."*
- *"A natural landscape with predominantly green hues and intricate details composition."*
- *"This bright image (400x300px) displays green and yellow as dominant colors with a natural color mood."*

---

## 🛠️ Technologies Used
- Python 3.x | OpenCV | NumPy | PIL | K-Means Clustering | NLG

---

## ▶️ How to Run
```bash
# Run demo with synthetic test images
python CODSOFT_Task3_ImageCaptioning.py

# Caption your own image
python -c "
from CODSOFT_Task3_ImageCaptioning import ImageCaptioningSystem
system = ImageCaptioningSystem()
features, captions = system.caption_image('your_image.jpg')
for cap in captions: print(cap)
"
```

---

## 🔗 Connect
- **LinkedIn:** [Jaspreet Singh](https://linkedin.com/in/jaspreet-singh-877748275)
- **GitHub:** [Jaspreet-Singh7](https://github.com/Jaspreet-Singh7)

---
*CodSoft Artificial Intelligence Internship — Task 03*
