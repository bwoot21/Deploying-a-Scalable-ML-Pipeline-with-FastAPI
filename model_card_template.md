# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Blaine Wooten created this model. The model is Random Forest from scikit learn.

## Intended Use
This model should be used to predict those whose income is greater than 50k.

## Training Data
The data used for training is from the UC Irvine Machine Learning Repository: https://archive.ics.uci.edu/dataset/20/census+income. This dataset has 48842 rows and 14 columns.
The training data trains on 75% of the data. I trained on the "salary" label/column. I used one-hot encoding and a label binarizer for binary conversion.

## Evaluation Data
The evaluation data uses, of course, the same dataset as the training data, but it comprises the other 25% of the dataset. It tests on the "salary" label/column.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
The metrics I used to evaluate the model are precision, recall, and F1 score. The values for each were the following: Precision was 0.7359, Recall was 0.6248, and F1 was 0.6758.

## Ethical Considerations
Salary could be considered personal and confidential information between employee and employer. This is not legally protected, but the information should be used responsibly.

## Caveats and Recommendations
Data could include bias, and it is unknown where the machine learning repository got their information, so more research could be done there to determine the sources of this data and their reliability.
