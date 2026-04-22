# Salient Object Detection 🚀

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust implementation of Salient Object Detection using graph-based manifold ranking and superpixel segmentation. This project aims to identify and highlight the most visually significant regions (objects) in an image.

---

## 🌟 Key Features

- **Manifold Ranking Algorithm**: Utilizes graph-based ranking for precise saliency estimation.
- **Multi-Scale Processing**: Computes saliency at multiple superpixel scales (300, 500, 700) and fuses them for optimal results.
- **Superpixel Segmentation**: Efficient SLIC-based segmentation for structured image analysis.
- **Bilateral Filtering**: Smooths saliency maps while preserving sharp object boundaries.
- **Interactive Visualization**: Outputs a composite image where salient objects remain in color while the background is converted to grayscale.

---

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**:
  - `OpenCV`: Image processing and I/O.
  - `NumPy`: Numerical computations and matrix operations.
  - `scikit-image`: SLIC superpixel segmentation.
  - `scipy`: Graph construction and linear algebra.

---

## 📁 Project Structure

```text
Sailient-Object-Detection/
├── backend/
│   └── saliency_core/
│       ├── main.py            # Entry point for processing images
│       ├── saliency.py         # Saliency computation logic
│       ├── graph.py            # Graph construction and adjacency matrix
│       ├── superpixels.py      # SLIC segmentation wrapper
│       ├── ranking.py          # Manifold ranking implementation
│       ├── visualization.py    # Utilities for rendering results
│       └── requirements.txt    # Python dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/AviBapatu/Sailient-Object-Detection.git
cd Sailient-Object-Detection
```

### 2. Install Dependencies
It is recommended to use a virtual environment:
```bash
cd backend/saliency_core
pip install -r requirements.txt
```

### 3. Run the Detection
By default, the script processes an image from the `Test Images` folder:
```bash
python main.py
```

Check `output.png` and `saliency.png` in the directory for results!

---

## 🔮 Future Roadmap

We are continuously working to improve the accuracy and accessibility of this project.

### 💻 Phase 1: Interactive Frontend
- **Web Dashboard**: A modern React-based interface for drag-and-drop image uploads.
- **Live Preview**: Real-time adjustment of saliency parameters (scales, filtering strength).
- **API Integration**: A FastAPI or Flask backend to serve the model as a microservice.

### 🧠 Phase 2: Advanced AI Models
- **Deep Learning Upgrade**: Transitioning from traditional graph-based methods to state-of-the-art Deep Learning models (e.g., **U^2-Net**, **BASNet**, or **PFSNet**).
- **Object Detection Integration**: Incorporating **YOLOv11** for combined object detection and segmentation.
- **Performance Optimization**: GPU acceleration via CUDA/TensorRT for faster inference.

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements or find bugs, feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
