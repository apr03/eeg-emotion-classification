import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import signal
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import random
from scipy.stats import ttest_ind


data = pd.read_csv('C:/emotions.csv')
data.head(3)
# Convert labels to numerical values
label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
data['label'] = data['label'].map(label_mapping)


# Visualization using a Pie Chart
# Count the occurrences of each emotion
emotion_counts = data['label'].value_counts()
# Define emotional labels
emotional_labels = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
# Map numerical labels to emotional labels
emotion_labels = [emotional_labels[label] for label in emotion_counts.index]
# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(emotion_counts, labels=emotion_labels, autopct='%1.1f%%', startangle=140, colors=['red', 'yellow', 'green'])
plt.title("Distribution of Emotions (0: NEGATIVE, 1: NEUTRAL, 2: POSITIVE)")
plt.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.
plt.savefig('emotion_distribution_pie.png')  # Save the plot
plt.show()

# Time-Series Visualization and Spectral Analysis
sample = data.iloc[0, data.columns.str.startswith('fft_')]  # Get all FFT columns for first row
sample_array = sample.values.astype(float)  # Convert to numpy array
plt.figure(figsize=(16, 10))
plt.plot(range(len(sample_array)), sample_array)
plt.title("EEG Time-Series Data")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.savefig('eeg_time_series.png')
plt.show()
# Spectral Analysis
sampling_rate = 256
frequencies, power_density = signal.welch(sample_array, fs=sampling_rate, nperseg=256)
plt.figure(figsize=(10, 6))
plt.semilogy(frequencies, power_density)
plt.title("Power Spectral Density")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power/Frequency")
plt.grid()
plt.savefig('power_spectral_density.png')
plt.show()

# Correlation Heatmap
correlation_matrix = data.drop('label', axis=1).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.savefig('correlation_heatmap.png')  # Save the plot
plt.show()

# t-SNE Visualization
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(data.drop('label', axis=1))
tsne_df = pd.DataFrame(tsne_results, columns=['Dimension 1', 'Dimension 2'])
tsne_df['label'] = data['label']
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='label', data=tsne_df, palette='viridis')
plt.title("t-SNE Visualization")
plt.savefig('tsne_visualization.png')  # Save the plot
plt.show()

# Define emotions based on unique labels in the dataset
emotions = data['label'].unique()


# Define colors for significant and non-significant features
significant_color = 'green'
non_significant_color = 'red'
# Create a dictionary to store the number of significant and non-significant features for each emotion
num_features = {emotion: {'significant': 0, 'non_significant': 0} for emotion in emotions}
# Perform t-tests and count significant features for each emotion
for emotion in emotions:
    subset = data[data['label'] == emotion]
    for feature in data.columns[:-1]:
        _, p_value = ttest_ind(subset[feature], data[feature])
        if p_value < 0.05:
            num_features[emotion]['significant'] += 1
        else:
            num_features[emotion]['non_significant'] += 1
# Extract emotion labels and corresponding feature counts
emotion_labels = list(num_features.keys())
significant_counts = [num_features[emotion]['significant'] for emotion in emotion_labels]
non_significant_counts = [num_features[emotion]['non_significant'] for emotion in emotion_labels]
# Create a bar chart to visualize the number of significant and non-significant features for each emotion
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(emotion_labels))
plt.bar(index, significant_counts, bar_width, label='Significant', color=significant_color)
plt.bar(index + bar_width, non_significant_counts, bar_width, label='Non-Significant', color=non_significant_color)
# Add labels and title
plt.xlabel('Emotion (0: Negative, 1: Neutral, 2: Positive)')
plt.ylabel('Number of Features')
plt.title('Significant and Non-Significant Features by Emotion')
plt.xticks(index + bar_width / 2, emotion_labels)
plt.legend()
# Display the counts above the bars
for i, (significant_count, non_significant_count) in enumerate(zip(significant_counts, non_significant_counts)):
    plt.text(i, significant_count + non_significant_count + 1, f'S: {significant_count}\nNS: {non_significant_count}', ha='center', va='bottom', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('significant_features_by_emotion.png')  # Save the plot
# Explanation for significant features
plt.figure(figsize=(8, 4))
plt.text(0.5, 0.5, 'Significant Features: These features show\nstatistically significant differences\nacross different emotions.',
         ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightgray', edgecolor='gray', boxstyle='round,pad=0.5'))
plt.axis('off')
plt.savefig('significant_features_explanation.png')  # Save the plot
plt.show()


# Advanced Preprocessing

# Normalization (e.g., z-score normalization)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

# Split the data into training and testing sets
X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Build the advanced neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=70, batch_size=32, verbose=2)

# Evaluate the model
model_acc = model.evaluate(X_test, y_test, verbose=0)[1]
print("Test Accuracy: {:.3f}%".format(model_acc * 100))

# Make predictions
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Confusion Matrix and Classification Report
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred, target_names=label_mapping.keys())

# Visualization of Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(label_mapping.keys()),
            yticklabels=list(label_mapping.keys()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# Save the confusion matrix plot
plt.savefig('confusion_matrix.png')
plt.tight_layout()
plt.show()

# Print Classification Report
print("Classification Report:\n", clr)