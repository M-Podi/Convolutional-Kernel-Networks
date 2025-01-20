# Convolutional Kernel Networks

## **Project Title:**
Convolutional Kernel Networks

## **Authors:**
- Matei-Alexandru Podeanu
- Robert Eduard Schmidt

---

## **Overview**

This project investigates the performance of **Convolutional Kernel Networks (CKNs)** compared to traditional **Convolutional Neural Networks (CNNs)** on three standard datasets: MNIST, Fashion-MNIST, and CIFAR-10. The focus is on evaluating accuracy, robustness, and computational efficiency when replacing ReLU activations in CNNs with Radial Basis Function (RBF)-based activations in CKNs.

---

## **Repository Structure**

- **`CNN.ipynb`:** Main Jupyter Notebook containing data preparation, model training, evaluation, and visualization.  
- **`requirements.txt`:** Lists all software dependencies for easy environment setup.

---

## **Key Contributions**

1. **Models Implemented:**  
   - **CNN:** Standard convolutional layers with ReLU activation.  
   - **CKN:** Similar architecture but with RBF activation instead of ReLU for feature transformation.

2. **Datasets:**  
   - **MNIST:** Handwritten digits (28x28 grayscale).  
   - **Fashion-MNIST:** Grayscale images of clothing (28x28).  
   - **CIFAR-10:** Color images of 10 classes (32x32).

3. **Performance Metrics:**  
   - Accuracy and loss across epochs.  
   - Confusion matrices for detailed classification analysis.  

4. **Technical Insights:**  
   - Simple yet efficient implementation to reproduce CKN results from academic research.  
   - Comparative analysis of runtime and accuracy.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/M-Podi/Convolutional-Kernel-Networks.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Convolutional-Kernel-Networks
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook CNN.ipynb
   ```
2. Follow the steps in the notebook to:
   - Load datasets.
   - Train and evaluate both CNN and CKN models.
   - Visualize performance metrics.

---

## **Results**

| Dataset       | CNN Accuracy (%) | CKN Accuracy (%) |
|---------------|------------------|------------------|
| **MNIST**     | 99.27           | 98.71           |
| **Fashion-MNIST** | 92.27           | 90.61           |
| **CIFAR-10**  | 72.00           | 66.93           |