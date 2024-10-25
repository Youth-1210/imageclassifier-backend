# app.py
from flask import Flask, request, jsonify, render_template
from inference import ImageClassifier

app = Flask(__name__)

# 配置参数
MODEL_PATH = 'best_model.pth'  # 使用最佳模型
NUM_CLASSES = 6  # 根据实际类别数修改
CLASS_LABELS = ['Fresh_0H', 'Fresh_5H', 'Fresh_10H', 'Fresh_15H', 'Fresh_20H', 'Fresh_25H']  # 根据实际类别标签修改

# 初始化模型分类器
classifier = ImageClassifier(
    model_path=MODEL_PATH,
    num_classes=NUM_CLASSES,
    class_labels=CLASS_LABELS,
    device='cpu'  # 如果有GPU，可设置为 'cuda'
)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    try:
        # 读取图像文件并进行预测
        image_bytes = file.read()
        app.logger.info('Received an image file for prediction.')
        result = classifier.predict(image_bytes)
        app.logger.info(f'Prediction result: {result}')
        return jsonify(result)  # 返回预测结果
    except Exception as e:
        app.logger.error(f'Error during prediction: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # 绑定到所有网络接口
