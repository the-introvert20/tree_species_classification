# 🌳 Tree Species Classification
<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-blue?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![GitHub Stars](https://img.shields.io/github/stars/SatyamPandey-07/TREE_SPECIES_CLASSIFICATION?style=social)

*An advanced AI-powered web application for tree species identification, location-based recommendations, and intelligent forestry insights using machine learning and computer vision.*

[🚀 Quick Start](#-quick-start) • [📖 Features](#-features--capabilities)  • [📋 Setup Guide](#-complete-setup--usage-guide) • [🤝 Contributing](#-contributing)

</div>


## 🎯 Overview

The **Tree Species Classification & Intelligence Assistant** is a comprehensive machine learning solution that combines:
- **🌍 Location Intelligence**: K-NN based tree species recommendations
- **🔍 Species Discovery**: Geographic distribution analysis
- **📸 Image Classification**: CNN-powered visual tree identification
- **📊 Data Analytics**: Insights from 1.38M+ tree records

Built with modern ML frameworks and deployed as an interactive web application.

---
## DEMO Screen Shots
### Example Screenshots


![Screenshot 6](img_used/Screenshot%202025-07-30%20202958.png)
![Screenshot 7](img_used/Screenshot%202025-07-30%20203119.png)


---
## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- pip package manager
- 4GB+ RAM (for CNN model loading)

### Installation & Setup

```bash
# Clone the repository
git clone https://github.com/the-introvert20/tree_species_classification.git
cd TREE_SPECIES_CLASSIFICATION

# Install dependencies
pip install -r requirements.txt

# Download the CNN model (255MB)
# Note: The CNN model is not included in the repository due to size limitations
# You can train your own using the tree_CNN.ipynb notebook or contact the author

# Run the application
streamlit run streamlit_integrated.py
```

🌐 **Access the app**: Open your browser and navigate to `http://localhost:8501`

---

## ✨ Features & Capabilities

### 🌲 1. Smart Location-Based Recommendations
- **Input**: GPS coordinates, tree diameter, native status, city/state
- **Output**: Top 5 most likely tree species for the location
- **Algorithm**: K-Nearest Neighbors with geospatial clustering
- **Use Case**: Urban planning, forestry management, biodiversity studies

### 📍 2. Species Distribution Mapping  
- **Input**: Select any tree species from dropdown
- **Output**: Geographic distribution and common locations
- **Features**: City-wise prevalence analysis
- **Use Case**: Conservation planning, habitat studies

### 📷 3. AI-Powered Image Classification
- **Input**: Upload tree images (leaves, bark, full tree)
- **Output**: Species prediction with confidence scores
- **Technology**: Custom CNN trained on 30+ species (255MB model)
- **Accuracy**: ~26% on validation set (challenging real-world dataset)
- **Note**: CNN model file not included in repo due to size - train using `tree_CNN.ipynb`

---

## 🗄️ Dataset & Data Sources

### 📊 Tree Metadata Repository
| **Attribute** | **Details** |
|---------------|-------------|
| **Source** | Municipal tree surveys from 50+ U.S. cities |
| **Total Records** | ~1.38 million georeferenced trees |
| **Coverage** | Louisville, Chicago, NYC, LA, and more |
| **Key Fields** | Species names, GPS coordinates, diameter, native status |
| **Time Period** | 2018-2022 survey data |

**Key Data Columns:**
- `common_name`: Tree species (e.g., Bur Oak)
- `scientific_name`: Botanical name (e.g., Quercus macrocarpa)  
- `latitude_coordinate`, `longitude_coordinate`: GPS location
- `city`, `state`, `address`: Geographic identifiers
- `native`: Whether the tree is native to the area
- `diameter_breast_height_CM`: Tree measurement standard

### 🖼️ Image Classification Dataset
| **Attribute** | **Details** |
|---------------|-------------|
| **Species Count** | 30 common North American species |
| **Total Images** | 1,454 curated samples |
| **Resolution** | Standardized to 224×224 pixels |
| **Augmentation** | Rotation, zoom, flip transformations |
| **Quality** | Real-world conditions (varying lighting, angles) |

**Dataset Structure:** Folder-based organization with each folder named after tree species for supervised learning.

---



### 🔍 Location-Based Recommender System
```
Input: [Latitude, Longitude, Diameter, Native_Status, City, State]
    ↓
Preprocessing: StandardScaler + LabelEncoder
    ↓
K-Nearest Neighbors (k=5)
    ↓
Output: Top 5 Recommended Species
```

**Technical Details:**
- **Algorithm**: scikit-learn `NearestNeighbors`
- **Distance Metric**: Euclidean distance in scaled feature space
- **Features**: Geographic + environmental + biological attributes
- **Performance**: Sub-second response time for 1.38M records


```

**Model Specifications:**
- **Framework**: TensorFlow/Keras
- **Architecture**: Sequential CNN with dropout regularization
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam (learning_rate=0.001)
- **Training**: 50 epochs with validation monitoring
- **Model Size**: 255MB (`basic_cnn_tree_species.h5`)

### 📊 Data Pipeline & Preprocessing
- **Encoding**: LabelEncoder for categorical variables
- **Scaling**: StandardScaler for numerical features
- **Image Processing**: Normalization to [0,1] range
- **Data Augmentation**: ImageDataGenerator with geometric transforms
- **Train/Validation Split**: 80/20 stratified sampling

---

## 🛠️ Technical Implementation

### 📁 Project Structure
```
TREE_SPECIES_CLASSIFICATION/
├── 📊 Data Processing
│   ├── 5M_trees.ipynb          # Train recommender system
│   └── tree_CNN.ipynb          # Train CNN classifier
├── 🚀 Production Application  
│   ├── streamlit_integrated.py # Main web application
│   └── requirements.txt        # Dependencies
├── 🤖 Trained Models
│   ├── tree_data.pkl          # Processed dataset (1.9MB)
│   ├── scaler.joblib          # Feature scaler (<1MB)
│   ├── nn_model.joblib        # KNN model (1MB)
│   └── basic_cnn_tree_species.h5  # CNN model (255MB)
└── 📚 Documentation
    ├── README.md              # This file
    └── PRODUCTION_READY.md    # Deployment guide
```

### ⚙️ System Requirements
| **Component** | **Requirement** |
|---------------|-----------------|
| **Python** | 3.13+ (tf-nightly compatible) |
| **Memory** | 4GB+ RAM for model loading |
| **Storage** | 2GB+ for models and data |
| **GPU** | Optional (CPU inference supported) |

### 🔧 Dependencies
```python
streamlit>=1.28.0      # Web application framework
tensorflow>=2.15.0     # Deep learning (use tf-nightly for Python 3.13)
scikit-learn>=1.3.0    # Machine learning algorithms
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical computing
pillow>=9.5.0          # Image processing
joblib>=1.3.0          # Model serialization
```

---

## 📋 Complete Setup & Usage Guide

### Step 1: Environment Setup
```bash
# Clone repository
git clone https://github.com/the-introvert20/tree_species_classification.git
cd TREE_SPECIES_CLASSIFICATION

# Create virtual environment (recommended)
python -m venv tree_env
tree_env\Scripts\activate  # Windows
# source tree_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Model Training (Optional - Models Included)
```bash
# Train recommender system (generates tree_data.pkl, scaler.joblib, nn_model.joblib)
jupyter notebook 5M_trees.ipynb

# Train CNN classifier (generates basic_cnn_tree_species.h5)
jupyter notebook tree_CNN.ipynb
```

### Step 3: Launch Application
```bash
# Start the web application
streamlit run streamlit_integrated.py

# Application will be available at: http://localhost:8501
```

---

## 🎯 Usage Examples

### 1. Location-Based Tree Recommendations
```
📍 Input Example:
- Latitude: 38.2527  
- Longitude: -85.7585
- Diameter: 25.4 cm
- Native Status: Yes
- City: Louisville
- State: Kentucky

🌳 Expected Output:
1. American Elm (Confidence: 85%)
2. Red Oak (Confidence: 78%)
3. Sugar Maple (Confidence: 72%)
4. Tulip Tree (Confidence: 69%)
5. Black Walnut (Confidence: 65%)
```

### 2. Species Distribution Query
```
🔍 Input: "Red Oak"
📊 Output: Geographic distribution map showing prevalence in:
- Chicago, IL (15,432 trees)
- Louisville, KY (8,921 trees)  
- Atlanta, GA (6,543 trees)
- [Additional cities...]
```

### 3. Image Classification
```
📸 Input: Upload tree image (JPG/PNG)
🤖 AI Analysis: 
- Primary Prediction: "Sugar Maple" (34.2%)
- Secondary: "Red Maple" (28.7%)
- Tertiary: "Norway Maple" (22.1%)
- Confidence Threshold: >25% for reliable prediction
```

---

## 📈 Performance Metrics & Limitations

![Screenshot 1](img_used/Screenshot%202025-07-30%20142901.png)
![Screenshot 2](img_used/Screenshot%202025-07-30%20142910.png)
![Screenshot 3](img_used/Screenshot%202025-07-30%20142920.png)
![Screenshot 4](img_used/Screenshot%202025-07-30%20142928.png)
![Screenshot 5](img_used/Screenshot%202025-07-30%20142941.png)

### Known Limitations
- **CNN Accuracy**: Limited by small training dataset (1,454 images for 30 classes)
- **Geographic Bias**: Dataset primarily covers U.S. cities
- **Image Quality**: Performance varies with lighting, angle, and image clarity
- **Species Coverage**: Limited to 30 common North American species

### Future Improvements
- [ ] Expand image dataset with data augmentation techniques
- [ ] Include international tree species and locations
- [ ] Implement ensemble methods for improved accuracy
- [ ] Add leaf shape and bark texture analysis
- [ ] Mobile application development

---

## 🚀 Deployment Options

### Local Development
```bash
streamlit run streamlit_integrated.py
```

### Cloud Platforms
- **Streamlit Cloud**: Direct GitHub integration




---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Areas for Contribution
- 🖼️ **Dataset Expansion**: Add more tree species images
- 🌍 **Geographic Coverage**: Include international tree data
- 🧠 **Model Improvements**: Enhance CNN architecture
- 🎨 **UI/UX**: Improve web interface design
- 📱 **Mobile Support**: Responsive design enhancements

### Development Workflow
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards
- Follow PEP 8 Python style guidelines
- Add docstrings for new functions
- Include unit tests for new features
- Update documentation for API changes

---


---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ for urban forestry and environmental conservation

</div>
