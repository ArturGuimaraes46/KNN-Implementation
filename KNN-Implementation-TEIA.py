from sklearn.datasets import make_classification, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



def prepare_data(X, y, test_size=0.2, random_state=42, scale=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# 1. Dados Sintéticos 2D
X_synthetic, y_synthetic = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
X_train_synthetic, X_test_synthetic, y_train_synthetic, y_test_synthetic = prepare_data(X_synthetic, y_synthetic)

# 2. Base Iris
iris = load_iris()
X_train_iris, X_test_iris, y_train_iris, y_test_iris = prepare_data(iris.data, iris.target)

# 3. Base Wine
wine = load_wine()
X_train_wine, X_test_wine, y_train_wine, y_test_wine = prepare_data(wine.data, wine.target)

# Exibir os tamanhos das divisões de treino e teste para confirmação
print("Synthetic dataset train-test sizes:", X_train_synthetic.shape, X_test_synthetic.shape)
print("Iris dataset train-test sizes:", X_train_iris.shape, X_test_iris.shape)
print("Wine dataset train-test sizes:", X_train_wine.shape, X_test_wine.shape)

def prepare_data(X, y, test_size=0.2, random_state=42, scale=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Gerar dados sintéticos 2D
X_synthetic, y_synthetic = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
X_train_synthetic, X_test_synthetic, y_train_synthetic, y_test_synthetic = prepare_data(X_synthetic, y_synthetic)

# Carregar dados da Iris
iris = load_iris()
X_train_iris, X_test_iris, y_train_iris, y_test_iris = prepare_data(iris.data, iris.target)

# Carregar dados do Wine
wine = load_wine()
X_train_wine, X_test_wine, y_train_wine, y_test_wine = prepare_data(wine.data, wine.target)

print("Synthetic dataset train-test sizes:", X_train_synthetic.shape, X_test_synthetic.shape)
print("Iris dataset train-test sizes:", X_train_iris.shape, X_test_iris.shape)
print("Wine dataset train-test sizes:", X_train_wine.shape, X_test_wine.shape)

def run_experiments(X_train, X_test, y_train, y_test):
    k_values = range(1, 10)  # Valores de k de 1 a 9
    training_sizes = [0.1, 0.2, 0.3, 0.4]  # Proporções para o conjunto de treinamento
    results = {}

    for k in k_values:
        results[k] = {}
        for size in training_sizes:
            # Calcular o tamanho do novo conjunto de treinamento
            limit = int(size * len(X_train))
            X_train_subset = X_train[:limit]
            y_train_subset = y_train[:limit]

            # Criar e treinar o modelo KNN
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_subset, y_train_subset)

            # Avaliar o modelo
            predictions = knn.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            results[k][size] = accuracy

            print(f"k={k}, train_size={size*100}%, accuracy={accuracy:.4f}")

    return results

# Exemplo de uso com o conjunto de dados Iris
# A função prepare_data já deve ter sido definida e executada para carregar e preparar os dados
results_iris = run_experiments(X_train_iris, X_test_iris, y_train_iris, y_test_iris)


def plot_results(results):
    plt.figure(figsize=(10, 8))
    for k in results:
        sizes = list(results[k].keys())
        accuracies = [results[k][size] for size in sizes]
        plt.plot(sizes, accuracies, marker='o', label=f'k={k}')

    plt.xlabel('Fraction of Training Data')
    plt.ylabel('Accuracy')
    plt.title('KNN Performance Variation')
    plt.legend()
    plt.grid(True)
    plt.show()

# Chamando a função de plotagem para o conjunto de dados Iris
plot_results(results_iris)


def run_evaluations(X_train, X_test, y_train, y_test):
    k_values = range(1, 10)  # Valores de k de 1 a 9
    training_sizes = [0.1, 0.2, 0.3, 0.4]  # Proporções para o conjunto de treinamento
    results = {}

    for k in k_values:
        results[k] = {}
        for size in training_sizes:
            # Calcular o tamanho do novo conjunto de treinamento
            limit = int(size * len(X_train))
            X_train_subset = X_train[:limit]
            y_train_subset = y_train[:limit]

            # Criar e treinar o modelo KNN
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_subset, y_train_subset)

            # Avaliar o modelo
            predictions = knn.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
            recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
            conf_matrix = confusion_matrix(y_test, predictions)

            results[k][size] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': conf_matrix
            }

            print(f"k={k}, train_size={size*100}%:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Confusion Matrix:\n{conf_matrix}\n")

    return results

# Exemplo de uso com o conjunto de dados Iris
# A função prepare_data já deve ter sido definida e executada para carregar e preparar os dados
results_iris = run_evaluations(X_train_iris, X_test_iris, y_train_iris, y_test_iris)


def plot_f1_scores(results, dataset_name):
    """ Plota o F1-Score em função de k e do percentual de treinamento. """
    plt.figure(figsize=(10, 8))
    for k in results:
        sizes = [size for size in results[k]]
        f1_scores = [results[k][size]['f1_score'] for size in results[k]]
        plt.plot(sizes, f1_scores, marker='o', label=f'k={k}')

    plt.title(f'F1-Score Variation for {dataset_name}')
    plt.xlabel('Fraction of Training Data')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True)
    plt.show()

# Exemplo de uso
plot_f1_scores(results_iris, "Iris Dataset")


def compare_with_sklearn(X_train, X_test, y_train, y_test):
    k_values = range(1, 10)
    training_sizes = [0.1, 0.2, 0.3, 0.4]
    own_results = {}  # Supondo que results já foram calculados
    sklearn_results = {}

    for k in k_values:
        sklearn_results[k] = {}
        for size in training_sizes:
            limit = int(size * len(X_train))
            X_train_subset = X_train[:limit]
            y_train_subset = y_train[:limit]

            # KNN do scikit-learn
            knn_sklearn = KNeighborsClassifier(n_neighbors=k)
            knn_sklearn.fit(X_train_subset, y_train_subset)
            predictions = knn_sklearn.predict(X_test)
            f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
            sklearn_results[k][size] = f1

    return own_results, sklearn_results

# Você pode agora plotar os resultados da mesma forma que plotou para sua implementação

def plot_comparison(own_results, sklearn_results, dataset_name):
    plt.figure(figsize=(12, 10))
    markers = ['o', 's']
    for i, (results, label) in enumerate(zip([own_results, sklearn_results], ['Own Implementation', 'scikit-learn'])):
        for k in results:
            sizes = [size for size in results[k]]
            f1_scores = [results[k][size] for size in results[k]]
            plt.plot(sizes, f1_scores, marker=markers[i], label=f'{label} k={k}')

    plt.title(f'Comparison of F1-Score for {dataset_name}')
    plt.xlabel('Fraction of Training Data')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True)
    plt.show()

# Use esta função para fazer a comparação visual.
