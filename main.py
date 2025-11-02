import numpy as np
import os
from PIL import Image
import torch
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# --------------------------- #
#   Load PlantVillage Dataset #
# --------------------------- #
def load_plantvillage_data(data_dir, n_samples=500, use_full_dataset=False, binary=False):
    """
    Load images from PlantVillage dataset.
    Args:
        data_dir (str): Path to dataset directory.
        n_samples (int): Number of samples to use.
        use_full_dataset (bool): If True, use all images.
        binary (bool): If True, use binary classification (healthy vs diseased).
    Returns:
        train_images, train_labels, test_images, test_labels
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory {data_dir} does not exist.")

    image_paths = []
    labels = []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    print(f"Found {len(class_names)} classes: {class_names}")

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Class {class_name}: {len(img_files)} images")
        for img_name in img_files:
            image_paths.append(os.path.join(class_path, img_name))
            if binary:
                label = 0 if class_name == 'Tomato___healthy' else 1
            else:
                label = class_to_idx[class_name]
            labels.append(label)

    image_paths = np.array(image_paths)
    labels = np.array(labels)
    print(f"Collected {len(image_paths)} image paths")

    if not use_full_dataset and len(image_paths) > n_samples:
        indices = np.random.choice(len(image_paths), n_samples, replace=False)
        image_paths = image_paths[indices]
        labels = labels[indices]
        print(f"Sampled {n_samples} image paths")

    batch_size = 100
    images = []
    valid_labels = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        for img_path, label in zip(batch_paths, batch_labels):
            try:
                img = Image.open(img_path).convert('RGB')
                img = transform(img)
                images.append(img.numpy().flatten().astype(np.float32))
                valid_labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        print(f"Processed batch {i//batch_size + 1}/{len(image_paths)//batch_size + 1}")

    images = np.array(images)
    labels = np.array(valid_labels)
    print(f"Loaded {len(images)} images with {len(labels)} labels")

    if len(images) == 0:
        raise ValueError("No images loaded. Check dataset path and image files.")

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    scaler = StandardScaler()
    train_images = scaler.fit_transform(train_images).astype(np.float32)
    test_images = scaler.transform(test_images).astype(np.float32)

    print(f"Train set: {len(train_images)} images, Test set: {len(test_images)} images")
    return train_images, train_labels, test_images, test_labels


# --------------------------- #
#     Feature Reduction (PCA) #
# --------------------------- #
def reduce_features(data, n_features=6):
    pca = PCA(n_components=n_features)
    return pca.fit_transform(data).astype(np.float32)


# --------------------------- #
#       Load and Process Data #
# --------------------------- #
data_dir = 'C:/Users/amitc/Hybrid_QCNN_Project/plantvillage'
n_samples = 500
use_full_dataset = False
n_features = 6
binary_classification = True  # Binary classification for high accuracy

try:
    train_images, train_labels, test_images, test_labels = load_plantvillage_data(
        data_dir, n_samples, use_full_dataset, binary=binary_classification
    )
    train_images = reduce_features(train_images, n_features)
    test_images = reduce_features(test_images, n_features)
except Exception as e:
    print(f"Failed to load dataset: {e}")
    exit(1)


# --------------------------- #
#     Quantum Circuit Setup   #
# --------------------------- #
num_qubits = n_features
backend = AerSimulator()
feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2)
ansatz = EfficientSU2(num_qubits=num_qubits, reps=2)
estimator = Estimator()
pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)


# --------------------------- #
#      Create QNN & QCNN      #
# --------------------------- #
def create_qnn(num_classes=2):
    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    qc_transpiled = transpile(qc, backend=backend, optimization_level=1)
    observables = SparsePauliOp(["Z" * num_qubits])
    gradient = ParamShiftEstimatorGradient(estimator=estimator)
    qnn = EstimatorQNN(
        circuit=qc_transpiled,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        observables=observables,
        estimator=estimator,
        gradient=gradient
    )
    return qnn


def create_qcnn(num_classes=2):
    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    for i in range(0, num_qubits, 2):
        if i + 1 < num_qubits:
            qc.cx(i, i + 1)
    qc_transpiled = transpile(qc, backend=backend, optimization_level=1)
    observables = SparsePauliOp(["Z" * num_qubits])
    gradient = ParamShiftEstimatorGradient(estimator=estimator)
    qnn = EstimatorQNN(
        circuit=qc_transpiled,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        observables=observables,
        estimator=estimator,
        gradient=gradient
    )
    return qnn


# --------------------------- #
#       QCHCNN Class          #
# --------------------------- #
class QCHCNN(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(QCHCNN, self).__init__()
        self.qnn = create_qnn(num_classes=num_classes)
        self.classical_layer = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, num_classes)
        )
        for layer in self.classical_layer:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x, weights=None):
        if weights is None:
            weights = np.random.rand(len(self.qnn.weight_params))
        quantum_output = self.qnn.forward(x.numpy(), weights)
        quantum_output = torch.tensor(quantum_output, dtype=torch.float32)
        classical_output = self.classical_layer(quantum_output)
        return classical_output

    def train_model(self, x, y, optimizer, epochs=50, batch_size=32, patience=5):
        criterion = torch.nn.CrossEntropyLoss()
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        weights = np.random.rand(len(self.qnn.weight_params))
        print("Starting QCHCNN training...")
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(0, len(x), batch_size):
                batch_x = x_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                optimizer.zero_grad()
                output = self.forward(batch_x, weights)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / (len(x) // batch_size + 1)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print("Completed QCHCNN training")
        with torch.no_grad():
            train_output = self.forward(x_tensor, weights)
            train_pred = torch.argmax(train_output, dim=1).numpy()
            train_acc = accuracy_score(y, train_pred)
            train_precision = precision_score(y, train_pred, average='macro', zero_division=0)
            train_recall = recall_score(y, train_pred, average='macro', zero_division=0)
            train_f1 = f1_score(y, train_pred, average='macro', zero_division=0)

        print(f"QCHCNN Train Accuracy: {train_acc}")
        print(f"QCHCNN Train Precision: {train_precision}")
        print(f"QCHCNN Train Recall: {train_recall}")
        print(f"QCHCNN Train F1 Score: {train_f1}")

        return train_acc, train_precision, train_recall, train_f1


# --------------------------- #
#     Training & Evaluation   #
# --------------------------- #
def train_and_evaluate(classifier, optimizer, train_data, train_labels, test_data, test_labels):
    print("Starting classifier.fit...")
    classifier.fit(train_data, train_labels)
    print("Completed classifier.fit")

    train_pred = classifier.predict(train_data)
    test_pred = classifier.predict(test_data)

    train_acc = accuracy_score(train_labels, train_pred)
    test_acc = accuracy_score(test_labels, test_pred)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    return train_acc, test_acc


# --------------------------- #
#         Main Loop           #
# --------------------------- #
optimizers = {
    "COBYLA": COBYLA(maxiter=400),
    "SPSA": SPSA(maxiter=400)
}

results = {}
for model_name in ["QNN", "QCNN"]:
    for opt_name, optimizer in optimizers.items():
        print(f"Training {model_name} with {opt_name}")
        num_classes = 2 if binary_classification else 10
        try:
            qnn = create_qnn(num_classes) if model_name == "QNN" else create_qcnn(num_classes)
            classifier = NeuralNetworkClassifier(qnn, optimizer=optimizer, loss="cross_entropy")
            train_and_evaluate(classifier, optimizer, train_images, train_labels, test_images, test_labels)
        except Exception as e:
            print(f"Error training {model_name} with {opt_name}: {e}")

# Train hybrid QCHCNN
try:
    print("Training QCHCNN...")
    qchcnn = QCHCNN(num_classes=2 if binary_classification else 10)
    optimizer = torch.optim.Adam(qchcnn.parameters(), lr=0.0005)
    qchcnn.train_model(train_images, train_labels, optimizer)
except Exception as e:
    print(f"Error training/evaluating QCHCNN: {e}")

