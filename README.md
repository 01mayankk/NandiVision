# 🐂 NandiVision – AI-Powered Indian Cattle & Buffalo Breed Classification

> **Smart Vision for Smarter Livestock** – Empowering Farmers with AI

---

## 📘 Overview

NandiVision is an AI-driven system designed to classify Indian cow and buffalo breeds using deep learning. India's livestock diversity is vast, but identifying breeds manually is difficult and requires expert knowledge.

**Key Components:**
- **Stage 1 Model:** Species classification → EfficientNet-B0 (ONNX)
- **Stage 2 Model:** Breed identification → EfficientNet-B3 (ONNX)
- **Frontend:** Next.js with modern UI
- **Backend:** FastAPI with ONNX inference
- **Authentication:** Supabase with role-based dashboard
---

## 🎯 Objectives

- Classify 08 cow breeds and 07 buffalo breeds accurately
- Build a two-stage inference pipeline to improve classification reliability
- Provide a modern, responsive, animated frontend UI
- Enable admin control panel for sending notifications
- Enable users to raise queries directly from dashboard
- Build a complete production-ready system using ONNX models for fast inference

---

## 🐄 Supported Breeds

### Cow Breeds (8)
- Alambadi
- Amritmahal
- Banni
- Bargur
- Deoni
- Kasargod
- Kangayam
- Nagori

### Buffalo Breeds (7)
- Bhadawari
- Jaffrabadi
- Mehsana
- Murrah
- Nagpuri
- Nili Ravi
- Surti

---

## 🧠 Model Architecture

| Stage | Purpose | Model | Reason |
|-------|---------|-------|--------|
| Stage-1 | Cow vs Buffalo vs None | EfficientNet-B0 | Lightweight, fast, accurate for coarse classification |
| Stage-2 | Breed Prediction | EfficientNet-B3 | Higher depth → best for fine-grain features distinguishing breeds |

**Note:** Both models are optimized & exported to ONNX to reduce inference latency.

---

## 🏗️ System Architecture

```
                  ┌────────────────────────────┐
                  │        User Browser        │
                  │   (Next.js Frontend UI)    │
                  └──────────────┬─────────────┘
                                 │
                                 ▼
                   HTTP Request (Image Upload)
                                 │
                                 ▼
                ┌────────────────────────────────┐
                │        FastAPI Backend         │
                │ - Handles API routes           │
                │ - Preprocess image             │
                │ - Coordinates model inference  │
                └──────────────┬─────────────────┘
                               │
                               ▼
       ┌──────────────────────────────────────────────────────────┐
       │                  ONNX Inference Engine                   │
       │  ┌────────────────────────────────────────────────────┐  │
       │  │            Stage 1 Model (EffNet-B0)              │  │
       │  │    Determines: Cow / Buffalo / None               │  │
       │  └────────────────────────────────────────────────────┘  │
       │                             │                            │
       │                             ▼                            │
       │  ┌────────────────────────────────────────────────────┐  │
       │  │            Stage 2 Model (EffNet-B3)              │  │
       │  │       Predicts Breed (if Cow/Buffalo)             │  │
       │  └────────────────────────────────────────────────────┘  │
       └──────────────────────────────────────────────────────────┘
                               │
                               ▼
                   JSON Prediction Response
                               │
                               ▼
                  Rendered on Next.js UI Dashboard


                  ┌────────────────────────────────┐
                  │          Supabase DB           │
                  │  - Authentication              │
                  │  - Role-based Profiles         │
                  │  - Notifications               │
                  │  - User Queries                │
                  └────────────────────────────────┘
```

---

## 📚 Literature Review

| # | Title | Year | Link |
|---|-------|------|------|
| 1 | Identification of Cattle Breed using CNN | 2021 | [ResearchGate](https://www.researchgate.net/publication/352417552) |
| 2 | Computer Vision-Based Detection of Dairy Cow Breed | 2022 | [MDPI Electronics](https://www.mdpi.com/2079-9292/11/22/3791) |
| 3 | Cattle Breed Classification Techniques | 2024 | [Propulsion Tech Journal](https://www.propulsiontechjournal.com/) |
| 4 | Ensemble Learning for Cattle Breed Identification | 2023 | [EAI](https://eudl.eu/pdf/10.4108/eai.23-11-2023.2343338) |
| 5 | Animal Breed Classification using Deep Learning | 2021 | [IJARSCT](https://ijarsct.co.in/Paper10386.pdf) |
| 6 | Attention-based Transfer Learning | 2024 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/38954103) |

### Key Research Findings
- Transfer Learning significantly improves performance on limited datasets
- EfficientNet outperforms traditional CNN models
- Data augmentation is crucial for breed variability
- Real-world lighting/pose variance demands robust models (EffNet-B3 excels here)

---

## 📦 Dataset

**Dataset Source:** [Indian Bovine Breeds - Kaggle](https://www.kaggle.com/datasets/lukex9442/indian-bovine-breeds)

### Folder Structure
```
dataset/
├── cows/
│   ├── Banni/
│   ├── Amritmahal/
│   └── ...
└── buffaloes/
    ├── Murrah/
    ├── Surti/
    └── ...
```

### Data Augmentation Techniques
- Rotate
- Flip
- Brightness/Contrast
- Zoom
- Crop
- CLAHE (Contrast Limited Adaptive Histogram Equalization)

---

## 🧩 Detailed System Features

### 🔐 Authentication (Supabase)
- Email + Password login/signup
- Profiles table with role column (admin, user)
- Auto-redirect dashboard based on role

### 🛠 Admin Dashboard Features
- ✅ Send notifications (text, image, video, audio, links)
- ✅ View all user queries
- ✅ Respond to user queries

### 👤 User Dashboard Features
- ✅ Receive notifications
- ✅ Raise queries to admin
- ✅ See AI classification history (optional future upgrade)

### 🎨 UI/UX Features (Next.js Frontend)
- Drag & Drop upload
- Live preview of image
- Animated transitions (Framer Motion)
- Modern blue-shaded theme
- Responsive layout for mobile, tablet, desktop
- Loading animation while inferencing
- Role-protected routes

---

## 🚀 Future Scope

- Real-time breed detection via camera (mobile/web)
- Disease classification model
- Farm management dashboard
- Offline mode via TF Lite
- Large-scale Indian cattle dataset creation

---

## 👨‍💻 Developed By

**Mayank Kumar**  
AI/ML Engineer & Full-Stack Developer

- 📧 **Email:** 02mayankk@gmail.com
- 🌐 **GitHub:** https://github.com/02mayankk

---

## 📄 License

This project is licensed under the **MIT License**.

---

**Last Updated:** December 2025