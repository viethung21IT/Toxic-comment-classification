import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin  # Giúp cho gridsearchcv hiểu tham số của mô hình

class ManualLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, lambda_param=0.01, class_weight=None):
        """
        :param learning_rate: Tốc độ học (alpha)
        :param n_iters: Số vòng lặp
        :param lambda_param: Hệ số L2 Regularization
        :param class_weight: 'balanced' hoặc None. Nếu 'balanced', tự động tăng trọng số cho lớp thiểu số.
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.lambda_param = lambda_param # L2 penalty
        self.class_weight = class_weight
        self.weights = None
        self.bias = None
        self.loss_history = []

    def decision_function(self, X):
        """Trả về giá trị thô (z = w*x + b) chưa qua Sigmoid"""
        return X.dot(self.weights) + self.bias
        
    def _sigmoid(self, z):
        # Clip z để tránh tràn số (overflow) với exp
        z = np.clip(z, -250, 250) 
        return 1 / (1 + np.exp(-z))

    def _compute_sample_weights(self, y):
        """Tính trọng số cho từng mẫu dựa trên sự mất cân bằng dữ liệu"""
        if self.class_weight != 'balanced':
            return np.ones(len(y))
        
        classes = np.unique(y)
        n_samples = len(y)
        n_classes = len(classes)
        
        # Đếm số lượng từng class
        counts = np.bincount(y.astype(int))
        
        # Tính weight cho từng class (0 và 1)
        class_weights_dict = {cls: n_samples / (n_classes * counts[cls]) for cls in classes}
        
        # Gán weight tương ứng cho từng mẫu trong y
        sample_weights = np.array([class_weights_dict[label] for label in y])
        return sample_weights

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Tính toán trọng số mẫu (Sample Weights) để xử lý Imbalance
        sample_weights = self._compute_sample_weights(y)

        for i in range(self.n_iters):
            # 1. Forward pass: z = w*x + b
            linear_model = X.dot(self.weights) + self.bias
            
            # 2. Activation: y_pred = sigmoid(z)
            y_predicted = self._sigmoid(linear_model)

            # 3. Tính Gradient (Đạo hàm)
            # Error thông thường: y_pred - y
            # Error có trọng số: (y_pred - y) * sample_weight
            error = (y_predicted - y) * sample_weights

            # Gradient cho weights (có thêm L2 Regularization term: lambda * w)
            # X.T.dot(error) rất nhanh với ma trận thưa
            dw = (1 / n_samples) * (X.T.dot(error) + (self.lambda_param * self.weights))
            
            # Gradient cho bias (Bias thường không áp dụng L2)
            db = (1 / n_samples) * np.sum(error)

            # 4. Update parameters (Gradient Descent)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # (Optional) Tính Loss để theo dõi
            if i % 100 == 0:
                # Binary Cross Entropy Loss + L2 Term
                # Clip y_predicted để tránh log(0)
                epsilon = 1e-15
                y_pred_safe = np.clip(y_predicted, epsilon, 1 - epsilon)
                base_loss = -np.mean(sample_weights * (y * np.log(y_pred_safe) + (1 - y) * np.log(1 - y_pred_safe)))
                l2_loss = (self.lambda_param / (2 * n_samples)) * np.sum(self.weights ** 2)
                self.loss_history.append(base_loss + l2_loss)

    def predict_proba(self, X):
        linear_model = X.dot(self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        y_predicted_cls = [1 if i > threshold else 0 for i in self.predict_proba(X)]
        return np.array(y_predicted_cls)
    
    def print_weights(self, feature_names=None, top_n=10):
        """
        In ra Bias và Weights.
        Nếu có feature_names, sẽ in ra top_n đặc trưng đóng góp tích cực/tiêu cực nhất.
        """
        print(f"  Bias (Intercept): {self.bias:.4f}")
        
        if feature_names is None:
            # Nếu không có tên đặc trưng, in dạng array rút gọn
            print(f"  Weights (shape {self.weights.shape}):")
            print(f"  {self.weights}")
        else:
            # Nếu có tên đặc trưng, in top features
            if len(feature_names) != len(self.weights):
                print("  [Cảnh báo] Độ dài feature_names không khớp với số lượng weights.")
                return

            # Sắp xếp index của weights
            sorted_indices = np.argsort(self.weights)
            
            print(f"  Top {top_n} từ khóa quan trọng nhất (Positive - Cho lớp 1):")
            for idx in sorted_indices[-top_n:][::-1]:
                print(f"    {feature_names[idx]}: {self.weights[idx]:.4f}")
                
            print(f"  Top {top_n} từ khóa tiêu cực nhất (Negative - Cho lớp 0):")
            for idx in sorted_indices[:top_n]:
                print(f"    {feature_names[idx]}: {self.weights[idx]:.4f}")

# --- Class One Vs Rest để xử lý Multi-label ---
class ManualLogisticRegressionOneVsRest(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.01, n_iters=1000, lambda_param=0.01):
        """
        Khai báo tham số tường minh để GridSearchCV có thể nhận diện.
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        
        self.models = [] 
        self.labels = []

    def fit(self, X, Y):
        # Chuyển đổi Y thành numpy array nếu là DataFrame
        if hasattr(Y, 'columns'):
            self.labels = Y.columns
            Y_values = Y.values
        else:
            self.labels = range(Y.shape[1])
            Y_values = Y

        n_labels = Y_values.shape[1]
        self.models = []

        # Gom nhóm tham số để truyền vào model con
        # Tại đây ta dùng các tham số self.learning_rate v.v...
        base_params = {
            'learning_rate': self.learning_rate,
            'n_iters': self.n_iters,
            'lambda_param': self.lambda_param
        }

        print(f"Bắt đầu huấn luyện {n_labels} mô hình Binary với lr={self.learning_rate}, lambda={self.lambda_param}...")
        
        for i in range(n_labels):
            y_col = Y_values[:, i]
            # Tạo model con với tham số hiện tại
            model = ManualLogisticRegression(class_weight='balanced', **base_params)
            model.fit(X, y_col)
            self.models.append(model)
            
        return self

    def predict_proba(self, X):
        """Trả về ma trận xác suất (n_samples, n_labels)"""
        preds = []
        for model in self.models:
            # Lấy xác suất của lớp Positive (1)
            preds.append(model.predict_proba(X))
        
        # Transpose để có dạng (n_samples, n_labels)
        return np.array(preds).T

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)
    
    def inspect_all_models(self, feature_names=None, top_n=5):
        """
        In thông tin trọng số cho từng nhãn.
        :param feature_names: List tên các đặc trưng (từ TF-IDF hoặc CountVectorizer)
        :param top_n: Số lượng đặc trưng quan trọng muốn hiển thị cho mỗi nhãn
        """
        print("\n" + "="*40)
        print("CHI TIẾT TRỌNG SỐ CÁC MÔ HÌNH CON")
        print("="*40)
        
        for i, model in enumerate(self.models):
            label_name = self.labels[i]
            print(f"\n NHÃN: {label_name}")
            print("-" * 20)
            model.print_weights(feature_names=feature_names, top_n=top_n)

    def decision_function(self, X):
        """
        Trả về ma trận Logits (n_samples, n_labels)
        Mỗi cột tương ứng với điểm số thô của một nhãn.
        """
        logits = []
        for model in self.models:
            # Gọi hàm decision_function của từng model con
            logits.append(model.decision_function(X))
        
        # Transpose để có dạng (n_samples, n_labels)
        return np.array(logits).T