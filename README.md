# Image_Classification_CNN

# **Problem Statement:** 

Image classification is a fundamental task in computer vision with applications in automation and visual recognition systems. The objective of this project is to design and train a Convolutional Neural Network (CNN) to accurately classify images into predefined categories.

# **Dataset:**

The dataset is taken from Kaggle, which consists of labeled images belonging to multiple categories. Images are preprocessed and resized before being used to train and evaluate the Convolutional Neural Network.

- Total Images: 2000.

- Classes: Labels (Cat/ Dog)

- Image Shape: 100*100, RGB 

# **Tools:**

Pandas

Numpy 

Matplotlib

Tensorflow

Keras

Scikit Learn

# **Approach and Methodology:**

## **a) Missing/ Null values:**

- Removed all the null values. 

- For missing values, binary/ fixed output valued columns like gender, credit history, self employed and dependents are replaced with their respective modes. For continuous outputs like loan amount and loan amount term are replaced with their means. 

## **b) Feature Engineering:** 

All the data is converted to numerical (for easy calculations).
  
## **c) Train Test split:**

- The data is split into training and testing data using the Scikitlearn function, train_test_split().

- Training the data using the random forest classifier model, a test data accuracy of up to 76% is obtained.
  
## **d) Feature Importance:**

Creating a new table, with features and their importances (calculated using feature_importances_ attribute of random forest), a bar graph visualised their importances.

# **Models used:**

## **Random Forest Classifier:**
Ensemble model, it combines several decision tree models to get stable predictions. Instead of relying on a single decision tree, a ‘forest’ of multiple trees are built, each trained on a random subset of the data and the features reducing overfitting.

# **Evaluation Metrics:**

## **Confusion Matrix:** 
It compares the model predicted values and the actual known values in a tabular way. 
Confusion Matrix for our model:

[ [12,29],

  [0, 82]  ]
  
## **Accuracy:**

It calculates the proportion of correct predictions. For example, if a model correctly predicts 90 out of 100 instances, the accuracy is 0.9. Our model’s accuracy is 76.42%.

# **Insights:**

- The CNN model achieved good accuracy on both training and test data.

- Data augmentation helped reduce overfitting and improved generalization.

- The model learned to identify visual patterns specific to each image class.
  
# **How to run the project:**

- Download or clone the repository.

- Open the project folder.

- Open the Jupyter Notebook file (.ipynb).

- Run all the cells from top to bottom.

# Project Structure:
├── input.csv, labels.csv, input_test.csv, labels_test.csv       # Datasets used in the project

├── image.classification.CNN.ipynb      # Jupyter Notebook with full code

├── README.md          # Project explanation
