
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import ttest_ind
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Global Visualization Styling

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)
plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.titlesize': 14,
    'axes.labelsize': 13
})

emotion_colors = ['grey', 'red', 'green']

# Load and Prepare Data

data = pd.read_csv('C:/emotions.csv')
label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
data['label'] = data['label'].map(label_mapping)

# Pie Chart: Emotion Distribution
emotion_counts = data['label'].value_counts()
emotional_labels = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
emotion_labels = [emotional_labels[label] for label in emotion_counts.index]

plt.figure(figsize=(7, 7))
plt.pie(
    emotion_counts,
    labels=emotion_labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=emotion_colors,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    textprops={'fontsize': 13, 'fontweight': 'bold'}
)
plt.title("Distribution of Emotions", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('emotion_distribution_pie.png', bbox_inches='tight')
plt.show()

# EEG Time-Series Visualization

sample = data.iloc[0, data.columns.str.startswith('fft_')]
sample_array = sample.values.astype(float)

plt.figure(figsize=(14, 6))
plt.plot(sample_array, color='teal', linewidth=2.5) 
plt.title("EEG Time-Series Signal", fontsize=16, fontweight='bold')
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('eeg_time_series.png', bbox_inches='tight')
plt.show()

# Power Spectral Density (PSD)

frequencies, power_density = signal.welch(sample_array, fs=sampling_rate, nperseg=256)
plt.figure(figsize=(10, 6))
plt.semilogy(frequencies, power_density, color='purple', linewidth=2)
plt.fill_between(frequencies, power_density, alpha=0.3, color='purple')
plt.title("Power Spectral Density", fontsize=16, fontweight='bold')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power/Frequency")
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('power_spectral_density.png', bbox_inches='tight')
plt.show()

# Correlation Heatmap

correlation_matrix = data.drop('label', axis=1).corr()
plt.figure(figsize=(14, 10))
sns.heatmap(
    correlation_matrix,
    cmap='coolwarm',
    vmin=-1, vmax=1,
    center=0,
    annot=False,
    cbar_kws={'shrink': 0.8, 'aspect': 15}
)
plt.title("Feature Correlation Heatmap", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', bbox_inches='tight')
plt.show()


# t-SNE Visualization (with regplot optional)
emotion_label_map = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(data.drop('label', axis=1))
tsne_df = pd.DataFrame(tsne_results, columns=['Dimension 1', 'Dimension 2'])
tsne_df['label'] = data['label']

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='Dimension 1', 
    y='Dimension 2',
    data=tsne_df, 
    hue=tsne_df['label'].map(emotion_label_map), 
    palette={
        'NEGATIVE': 'red', 
        'NEUTRAL': 'gray',  
        'POSITIVE': 'green'
    },
    s=80, alpha=0.8, edgecolor='white'
)
plt.title("t-SNE Visualization of EEG Feature Space", fontsize=16, fontweight='bold')
plt.legend(title='Emotion', loc='best')
plt.tight_layout()
plt.savefig('tsne_visualization.png', bbox_inches='tight')
plt.show()

# Significance Analysis (t-tests)

emotions = data['label'].unique()
num_features = {emotion: {'significant': 0, 'non_significant': 0} for emotion in emotions}

for emotion in emotions:
    subset = data[data['label'] == emotion]
    for feature in data.columns[:-1]:
        _, p_value = ttest_ind(subset[feature], data[feature])
        if p_value < 0.05:
            num_features[emotion]['significant'] += 1
        else:
            num_features[emotion]['non_significant'] += 1

emotion_labels = list(num_features.keys())
significant_counts = [num_features[e]['significant'] for e in emotion_labels]
non_significant_counts = [num_features[e]['non_significant'] for e in emotion_labels]

plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(emotion_labels))
plt.bar(index, significant_counts, bar_width, label='Significant', color='green', alpha=0.9)
plt.bar(index + bar_width, non_significant_counts, bar_width, label='Non-Significant', color='red', alpha=0.8)
plt.xlabel('Emotion (0: Negative, 1: Neutral, 2: Positive)')
plt.ylabel('Number of Features')
plt.title('Significant vs Non-Significant Features by Emotion', fontsize=16, fontweight='bold')
plt.xticks(index + bar_width / 2, ['Negative', 'Neutral', 'Positive'])
plt.legend()

for i, (s, n) in enumerate(zip(significant_counts, non_significant_counts)):
    plt.text(i + 0.17, max(s, n) + 5, f"S:{s}\nNS:{n}", ha='center', fontsize=12)

plt.tight_layout()
plt.savefig('significant_features_by_emotion.png', bbox_inches='tight')
plt.show()

# Data Normalization and Splitting

scaler = StandardScaler()
data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Neural Network Model

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_split=0.2, epochs=70, batch_size=32, verbose=2)

# Model Evaluation

model_acc = model.evaluate(X_test, y_test, verbose=0)[1]
print("Test Accuracy: {:.3f}%".format(model_acc * 100))

y_pred = np.argmax(model.predict(X_test), axis=-1)

cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred, target_names=label_mapping.keys())

# Confusion Matrix Visualization

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    xticklabels=list(label_mapping.keys()),
    yticklabels=list(label_mapping.keys()),
    linewidths=1,
    linecolor='white'
)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.show()

print("Classification Report:\n", clr)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import itertools

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

# MODEL COMPARISON AND EVALUATION

print("\n--- Model Comparison and Evaluation ---")

# Define classical models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True, random_state=42)
}

results = []

# Train and evaluate each model
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred_model = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred_model)
    f1 = f1_score(y_test, y_pred_model, average='weighted')
    results.append({'Model': name, 'Accuracy': acc, 'F1-Score': f1})

# Add Neural Network results
results.append({
    'Model': 'Neural Network',
    'Accuracy': model_acc,
    'F1-Score': f1_score(y_test, y_pred, average='weighted')
})

# Convert results to DataFrame
results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)

print("\nModel Performance Summary:\n")
print(results_df.to_string(index=False))

# Identify best performing model
best_model = results_df.iloc[0]
print(f"\n Best Performing Model: {best_model['Model']} "
      f"with Accuracy = {best_model['Accuracy']*100:.2f}% "
      f"and F1-Score = {best_model['F1-Score']*100:.2f}%")
