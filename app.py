import os
from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Get absolute paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Static directory: {STATIC_DIR}")
logger.info(f"Upload directory: {UPLOAD_FOLDER}")

# Configuration
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create directories
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Pipeline state
pipeline_state = {
    'data': None,
    'original_data': None,
    'X_train': None,
    'X_test': None,
    'y_train': None,
    'y_test': None,
    'model': None,
    'scaler': None,
    'target_column': None,
    'feature_columns': None,
    'preprocessing_method': None,
    'split_ratio': 0.8
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    logger.info("=" * 50)
    logger.info("ROOT ROUTE ACCESSED")
    logger.info(f"Remote address: {request.remote_addr}")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request path: {request.path}")
    
    try:
        index_path = os.path.join(STATIC_DIR, 'index.html')
        logger.info(f"STATIC_DIR: {STATIC_DIR}")
        logger.info(f"STATIC_DIR exists: {os.path.exists(STATIC_DIR)}")
        logger.info(f"index_path: {index_path}")
        logger.info(f"index_path exists: {os.path.exists(index_path)}")
        
        if not os.path.exists(index_path):
            logger.error("index.html not found!")
            return make_response("index.html not found", 404)
        
        logger.info("About to send file...")
        with open(index_path, 'r', encoding='utf-8') as f:
            content = f.read()
        response = make_response(content)
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        logger.info("File sent successfully!")
        return response
    except Exception as e:
        logger.error(f"Error serving index: {str(e)}", exc_info=True)
        return make_response(f"Error: {str(e)}", 500)

@app.route('/test')
def test():
    return make_response("Test endpoint works!", 200)

@app.route('/api/health')
@app.route('/health')
def health():
    health_info = {
        'status': 'healthy',
        'service': 'ML Pipeline Builder',
        'static_dir': STATIC_DIR,
        'static_exists': os.path.exists(STATIC_DIR),
        'index_exists': os.path.exists(os.path.join(STATIC_DIR, 'index.html'))
    }
    logger.info(f"Health check: {health_info}")
    return jsonify(health_info), 200

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        pipeline_state['original_data'] = df.copy()
        pipeline_state['data'] = df.copy()
        
        info = {
            'filename': filename,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'column_types': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'sample_data': df.head(5).to_dict('records')
        }
        
        os.remove(filepath)
        
        return jsonify({'status': 'success', 'info': info}), 200
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/select-target', methods=['POST'])
def select_target():
    try:
        data = request.json
        target_column = data.get('target_column')
        
        if not target_column:
            return jsonify({'error': 'Target column is required'}), 400
        
        pipeline_state['target_column'] = target_column
        
        return jsonify({'status': 'success', 'target_column': target_column}), 200
        
    except Exception as e:
        logger.error(f"Select target error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/preprocess', methods=['POST'])
def preprocess_data():
    try:
        data = request.json
        preprocess_type = data.get('type', 'none')
        
        if pipeline_state['data'] is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        target_column = pipeline_state.get('target_column')
        if not target_column:
            return jsonify({'error': 'Target column not set'}), 400
        
        df = pipeline_state['data'].copy()
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) == 0:
            return jsonify({'error': 'No numeric columns found'}), 400
        
        X_numeric = X[numeric_columns]
        pipeline_state['feature_columns'] = numeric_columns
        
        if preprocess_type == 'standardization':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_numeric)
            pipeline_state['scaler'] = scaler
            pipeline_state['preprocessing_method'] = 'standardization'
        elif preprocess_type == 'normalization':
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_numeric)
            pipeline_state['scaler'] = scaler
            pipeline_state['preprocessing_method'] = 'normalization'
        else:
            X_scaled = X_numeric.values
            pipeline_state['preprocessing_method'] = 'none'
        
        pipeline_state['data'] = pd.DataFrame(X_scaled, columns=numeric_columns)
        pipeline_state['data'][target_column] = y.values
        
        return jsonify({
            'status': 'success',
            'preprocessing': preprocess_type,
            'features_used': numeric_columns
        }), 200
        
    except Exception as e:
        logger.error(f"Preprocess error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/split', methods=['POST'])
def split_data():
    try:
        data = request.json
        split_ratio = data.get('split_ratio', 0.8)
        test_size = 1 - split_ratio
        
        if pipeline_state['data'] is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        df = pipeline_state['data']
        target_column = pipeline_state['target_column']
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if len(y.unique()) < 20 else None
        )
        
        pipeline_state['X_train'] = X_train
        pipeline_state['X_test'] = X_test
        pipeline_state['y_train'] = y_train
        pipeline_state['y_test'] = y_test
        pipeline_state['split_ratio'] = split_ratio
        
        return jsonify({
            'status': 'success',
            'train_size': len(X_train),
            'test_size': len(X_test)
        }), 200
        
    except Exception as e:
        logger.error(f"Split error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        data = request.json
        model_type = data.get('model_type')
        
        if pipeline_state['X_train'] is None:
            return jsonify({'error': 'Data not split'}), 400
        
        X_train = pipeline_state['X_train']
        y_train = pipeline_state['y_train']
        X_test = pipeline_state['X_test']
        y_test = pipeline_state['y_test']
        
        if model_type == 'logistic_regression':
            model = LogisticRegression(max_iter=1000, random_state=42)
            model_name = "Logistic Regression"
        elif model_type == 'decision_tree':
            model = DecisionTreeClassifier(random_state=42)
            model_name = "Decision Tree Classifier"
        else:
            return jsonify({'error': 'Invalid model type'}), 400
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        pipeline_state['model'] = model
        
        return jsonify({
            'status': 'Training completed successfully',
            'model_name': model_name,
            'accuracy': float(accuracy),
            'confusion_matrix_plot': img_base64
        }), 200
        
    except Exception as e:
        logger.error(f"Train error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_pipeline():
    try:
        for key in pipeline_state:
            pipeline_state[key] = None if key != 'split_ratio' else 0.8
        return jsonify({'status': 'Pipeline reset successfully'}), 200
    except Exception as e:
        logger.error(f"Reset error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    logger.error(f"404 Error: {request.url}")
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"500 Error: {str(e)}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Get port from environment variable (Render/Railway/Heroku compatible)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)