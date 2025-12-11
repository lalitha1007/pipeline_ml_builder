# 🚀 ML Pipeline Builder

A web-based, no-code Machine Learning pipeline builder that allows users to create and run simple ML workflows without writing any code. Built with Flask backend and a clean, intuitive web interface.

![ML Pipeline Builder](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)

## ✨ Features

### 📁 Dataset Upload

- Upload CSV or Excel (.xlsx, .xls) files
- Automatic format validation
- Display dataset information (rows, columns, column names)
- Preview first 5 rows of data
- Graceful error handling for invalid formats

### ⚙️ Data Preprocessing

- **Standardization**: StandardScaler (mean=0, std=1)
- **Normalization**: MinMaxScaler (range 0-1)
- **No Preprocessing**: Use raw data as-is
- Visual selection interface

### ✂️ Train-Test Split

- Interactive slider to select split ratio
- Options: 60-40, 65-35, 70-30, 75-25, 80-20, 85-15, 90-10
- Clear display of training and testing set sizes

### 🤖 Model Selection

Choose from two popular classification models:

- **Logistic Regression**: Linear model for classification
- **Decision Tree Classifier**: Tree-based model for non-linear patterns

### 📊 Model Results & Visualization

- Model execution status
- **Accuracy score** displayed prominently
- **Confusion Matrix** visualization
- Clear, visual presentation of results

### 🎨 User Experience

- **Step-by-step pipeline flow** with visual progress tracking
- **Drag-and-drop inspired** interface
- Clean, modern UI design
- No coding required from users
- Easy to understand for beginners
- Visual pipeline flow similar to Orange Data Mining

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone or download the project**

   ```bash
   cd ML_pipeline
   ```
2. **Install required packages**

   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**

   ```bash
   python app.py
   ```
4. **Open your browser**
   Navigate to: `http://localhost:5000`

## 📖 Usage Guide

### Step 1: Upload Your Dataset

1. Click the upload area or drag and drop your CSV/Excel file
2. Wait for the file to be processed
3. Review the dataset overview (rows, columns, data preview)

### Step 2: Select Target Column

1. Choose which column you want to predict (target variable)
2. Click "Next" to proceed

### Step 3: Data Preprocessing

1. Select a preprocessing method:
   - **No Preprocessing**: Use data as-is
   - **Standardization**: Scale features to mean=0, std=1
   - **Normalization**: Scale features to range 0-1
2. Click "Apply & Continue"

### Step 4: Split Your Data

1. Use the slider to select training/testing split ratio
2. Default is 80% training, 20% testing
3. Click "Split & Continue"

### Step 5: Choose & Train Model

1. Select a machine learning model:
   - **Logistic Regression**: Good for linear relationships
   - **Decision Tree**: Handles non-linear patterns
2. Click "🚀 Train Model"
3. Wait for training to complete

### Step 6: View Results

1. See your model's accuracy score
2. Examine the confusion matrix visualization
3. Start a new pipeline or go back to adjust settings

## 📁 Project Structure

```
ML_pipeline/
│
├── app.py                 # Flask backend with all API endpoints
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── static/
│   └── index.html        # Frontend UI (HTML/CSS/JavaScript)
│
└── uploads/              # Temporary folder for uploaded files (auto-created)
```

## 🔧 Technical Details

### Backend (Flask)

- **Framework**: Flask 3.0.0
- **ML Library**: scikit-learn 1.3.2
- **Data Processing**: pandas 2.1.4, numpy 1.26.2
- **Visualization**: matplotlib 3.8.2, seaborn 0.13.0
- **File Support**: openpyxl 3.1.2 (for Excel files)

### API Endpoints

- `POST /api/upload` - Upload and validate dataset
- `POST /api/select-target` - Select target column
- `POST /api/preprocess` - Apply preprocessing
- `POST /api/split` - Perform train-test split
- `POST /api/train` - Train selected model
- `GET /api/pipeline-status` - Get current pipeline status
- `POST /api/reset` - Reset pipeline state

### Frontend

- Pure HTML/CSS/JavaScript (no framework dependencies)
- Responsive design
- Modern, gradient-based UI
- Step-by-step wizard interface
- Real-time feedback and validation

## 🎯 Supported File Formats

### CSV Files (.csv)

- Standard comma-separated values
- UTF-8 encoding recommended

### Excel Files (.xlsx, .xls)

- Microsoft Excel format
- Reads first sheet by default

## 📊 Example Datasets

You can test the application with popular datasets:

- **Iris Dataset**: 3-class classification (flower species)
- **Titanic Dataset**: Binary classification (survival prediction)
- **Wine Quality Dataset**: Multi-class classification
- Any custom CSV/Excel with numerical features and a target column

## ⚠️ Requirements & Limitations

### Requirements

- Dataset must contain at least one target column
- Features should be numeric for best results
- Minimum 2 rows of data required

### Current Limitations

- Classification models only (not regression)
- Binary and multi-class classification supported
- Non-numeric features are automatically excluded
- Pipeline state stored in memory (resets on server restart)
- Single user at a time (demo version)

## 🚀 Future Enhancements

Potential improvements for future versions:

- [ ] Support for regression models
- [ ] More preprocessing options (PCA, feature selection)
- [ ] Additional model types (Random Forest, SVM, Neural Networks)
- [ ] Model comparison functionality
- [ ] Export trained models
- [ ] Save and load pipelines
- [ ] Multi-user support with database storage
- [ ] Advanced visualizations (ROC curves, feature importance)
- [ ] Hyperparameter tuning interface

## 👨‍💻 Development

To modify or extend the application:

1. **Backend modifications**: Edit `app.py`
2. **Frontend changes**: Edit `static/index.html`
3. **Add new dependencies**: Update `requirements.txt`

## 📄 License

This project is created for educational and demonstration purposes.

## 🤝 Contributing

This is an assignment/demonstration project. Suggestions and improvements are welcome!

## 📧 Developer
Made with 🎯 by Lalitha K

For issues or questions, please refer to the troubleshooting section above.

## 👨‍💻 Developer
Made with 🎯 by Lalitha

