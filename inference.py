# inference.py
import torch
from torchvision import transforms
from PIL import Image
from model3 import DualBranchImageModel  # 假设模型定义在 model.py
import io

class ImageClassifier:
    def __init__(self, model_path, num_classes, class_labels, device='cpu'):
        self.device = device
        self.class_labels = class_labels
        # 直接加载整个模型对象
        self.model = torch.load(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_bytes):
        # 读取图像并进行预处理
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)

        # 模型推理
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_idx = predicted.item()
            predicted_label = self.class_labels[predicted_idx]

        # 返回字典格式的预测结果
        return {'predicted_label': predicted_label}
