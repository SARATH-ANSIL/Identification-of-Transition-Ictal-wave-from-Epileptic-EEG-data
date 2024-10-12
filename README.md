# Identification of Transition Ictal Wave from Epileptic EEG Data

This project focuses on identifying transition ictal waves in epileptic EEG data using machine learning techniques. Various classification models are implemented to find the best-suited model for seizure onset detection.

## Features
- **Data Processing:** Utilizes EEG datasets to extract relevant features for classification.
- **Multiple Models:** Implements various machine learning models, including SVM, Logistic Regression, KNN, Naive Bayes,PCA, Linear Regression and a Deep Learning ANN.
- **Visualization:** Provides visualizations of the EEG signals and model accuracy across different kernel types.

## Prerequisites
- Python 3.x
- The following Python libraries must be installed:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `keras`
  - `pyeeg`
  - `nolds`
  - `entropy`

You can install the required libraries using pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn keras pyeeg nolds entropy
```
## Project Structure
```bash
.
├── datasets/              # Folder containing EEG datasets
├── models/                # Contains model architecture and training scripts
│   ├── model.py           # Script defining the model and training process
│   └── ...                # Other necessary model files and dependencies
├── output/                # Folder to save output images and results
├── analysis.ipynb         # Main Jupyter Notebook for data processing and model training
└── README.md              # Project README file
```
## How It Works
- **Data Loading**: The EEG dataset is loaded and prepared for analysis.
- **Feature Extraction**: Relevant features are extracted from the EEG signals for model training.
- **Model Training**: Multiple machine learning models are trained on the extracted features.
- **Model Evaluation**: The models are evaluated based on their accuracy in detecting seizures.
- **Visualization**: The results are visualized to provide insights into model performance and EEG signal characteristics.
  
## Running the Project
Open the Jupyter Notebook:
`jupyter notebook Identification-of-transition-ictal-wave-from-Epileptic-EEG-data.ipynb`

Follow the instructions in the notebook to load your dataset and initialize the models.

Run the provided scripts to train and evaluate the models.

Visualize the results to analyze the model performance and EEG signal patterns.

## Example Output
After running the analysis notebook, the output folder will contain visualizations of the EEG signals and the model accuracies across different kernel types.

## Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page for new ideas.

## License
This project is licensed under the MIT License.

## Acknowledgments
Scikit-learn for machine learning functionalities.
Keras for deep learning capabilities.
Pyeeg for EEG data processing tools.
Matplotlib and Seaborn for data visualization.
