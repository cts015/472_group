import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
import shap
import pickle

# Import filters to remove unnecessary warnings
from warnings import simplefilter
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import ConvergenceWarning

# Load the data
df = pd.read_csv("CHEG472_Hooplas.csv")
print(df.head)

# check if the data contains null values
print(df.isna().sum())

#Check datatypes
print(df.dtypes)

# Check for duplicates in the entire dataset
duplicates = df.duplicated()
# If there are any duplicates, the 'duplicates' variable will contain True for those rows
if duplicates.any():
    # Get the rows with duplicates
    duplicate_rows = df[duplicates]
    print("Duplicate rows:")
    print(duplicate_rows)
else:
    print("No duplicates found in the dataset.")

#Remove unnecessary data (round no., player names)
df = df.drop(['Round','Player_Name'],axis=1)

df.describe() #Summary stats

# One-hot encode the 'Speed' column
df_encoded = pd.get_dummies(df['Speed'])

# Combine the original DataFrame with the encoded column, removing the original speed column
df_final = pd.concat([df.drop('Speed', axis= 1), df_encoded], axis=1)

df_final.head()

# Calculate outliers for each column
outliers_dict = {}

# Only check the columns that are numerical and would contain outliers
columns_to_check = df_final.select_dtypes(include=['int64', 'float64']).columns

# Create boxplots for specified columns
df_final[columns_to_check].boxplot()
plt.title("Boxplots for Numerical Columns")
plt.ylabel("Values")
plt.xticks(rotation=45)
plt.show()

# Check for outliers and plot the boxplot
for col in columns_to_check:
   q1 = np.quantile(df_final[col], 0.25)
   q3 = np.quantile(df_final[col], 0.75)
   iqr = q3 - q1
   lower_bound = q1 - 1.5 * iqr
   upper_bound = q3 + 1.5 * iqr
   outliers = df_final[col][(df_final[col] < lower_bound) | (df_final[col] > upper_bound)]
   outliers_dict[col] = outliers.tolist()

# Print outliers for each column
for col, outliers in outliers_dict.items():
   if outliers:
       print(f"Outliers in column '{col}': {outliers}")
   else:
       print(f"No outliers found in column '{col}'")

# Normalize the dataframe
scaler = MinMaxScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df_final), columns=df_final.columns)

# Display the normalized dataframe head
df_norm.head()

#Create an overall correlation matrix
corr_matrix = df_norm.corr()
#Create a heatmap based on the correlation matrix
sns.set(rc={'figure.figsize': (10, 4)}) #Set figure size
sns.heatmap(corr_matrix, annot=True, cmap="viridis")
plt.title("Correlation Heatmap")
plt.show()

#Throw angle and pins knocked have a significant negative correlation. Decreasing throw angle will help to increase pins knocked.

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Split the dataset into features and target
X = df_norm.drop('Pins_Knocked ', axis=1)
Y = df_norm['Pins_Knocked ']

# Define the number of folds for K-Fold cross-validation
n_folds = 6

# Initialize empty lists to store evaluation metrics
rmse_scores = []
mae_scores = []
r2_test_scores = []
r2_train_scores = []

# Define the models dictionary
models = {
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'MLP': MLPRegressor(),
    'SVR': SVR(),
    'Linear Regression': LinearRegression(),
}
# K-Fold cross-validation loop
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
for name, model in models.items():
    # Loop through each fold
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

        # Train the model on the training data for this fold
        model.fit(X_train, y_train)

        # Predict on the testing data for this fold
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        rmse = mean_squared_error(y_test, y_pred, squared=False)  # Calculate RMSE directly
        mae = mean_absolute_error(y_test, y_pred)
        r2_test = r2_score(y_test, y_pred)

        # Additionally, calculate R-squared on the training data for each fold (optional)
        y_train_pred = model.predict(X_train)
        r2_train = r2_score(y_train, y_train_pred)

        # Calculate R^2 difference
        r2_diff = r2_train - r2_test

        # Append the scores to the lists
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_test_scores.append(r2_test)
        r2_train_scores.append(r2_train)

    # Print average scores after all folds for each model
    print(f"{name}:")
    print(f"  Average RMSE: {np.mean(rmse_scores):.3f}")
    print(f"  Average MAE: {np.mean(mae_scores):.3f}")
    print(f"  Average R² Test Score: {np.mean(r2_test_scores):.3f}")
    print(f"  Average R² Train Score: {np.mean(r2_train_scores):.3f}")
    print(' ')

# Create a DataFrame to store the results
results_df_norm = pd.DataFrame({
    'Model': ['Random Forest', 'Random Forest', 'Random Forest', 'Random Forest', 'Random Forest', 'Random Forest',
              'Gradient Boosting', 'Gradient Boosting', 'Gradient Boosting', 'Gradient Boosting', 'Gradient Boosting', 'Gradient Boosting',
              'MLP', 'MLP', 'MLP', 'MLP', 'MLP', 'MLP',
              'SVR', 'SVR', 'SVR', 'SVR', 'SVR', 'SVR',
              'Linear Regression', 'Linear Regression', 'Linear Regression', 'Linear Regression', 'Linear Regression', 'Linear Regression'],
    'R^2 Test': r2_test_scores,
    'R^2 Train': r2_train_scores,
})

# Select the best model based on Test R², or lowest RMSE if R² is identical
best_model_row = results_df_norm.loc[results_df_norm['R^2 Test'].idxmax()]

print("\nBest model based on Test R² performance:")
print(best_model_row)

# Assuming 'results_df_norm' is the DataFrame containing the model performance metrics
models = results_df_norm['Model']
train_r2 = results_df_norm['R^2 Train']
test_r2 = results_df_norm['R^2 Test']

# Melt the DataFrame for easier plotting
melted_df_norm = results_df_norm.melt(id_vars='Model', var_name='Metric', value_name='R^2')

# Create a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Model', y='R^2', hue='Metric', data=melted_df_norm)
plt.title('R^2 Scores for Different Models')
plt.xlabel('Model')
plt.ylabel('R^2')
plt.legend(title='Metric')
plt.show()

!pip install pickle-mixin -q
!pip install numpy -q
!pip install langchain -q
!pip install openai -q
!pip install langchain_community -q
!pip install langchain_openai -q
!pip install tiktoken -q
!pip install python-dotenv -q
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
!source $HOME/.cargo/env
!pip install langchain_core -q

from langchain.prompts import PromptTemplate
template = (
    """
    You are a knowledgeable bowling expert. A user will provide you with a description of their bowling attempt. Your task is to give them a specific, actionable tip to improve their performance.

    **Example Input:** "I keep hitting the gutter on the right side."

    **Example Output:** "Try moving slightly to the left and focusing on a target closer to the center of the lane."

    Please provide your tip in a clear and concise manner.
    """
)
prompt = PromptTemplate.from_template(template)

!pip install langchain-core -q
from langchain_core.functional import chain
from langchain_core.io import Input, Print

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


def create_chatbot(llm, prompt):
    """Creates a chatbot using the provided LLM.

    Args:
        llm: The language model to use.
        prompt: The prompt template.

    Returns:
        A LangChain chain representing the chatbot.
    """

    chatbot = chain(
        Input(),  # User input
        prompt,
        llm
    ) | Print()  # Print the chatbot's response

    return chatbot


# Replace 'YOUR_OPENAI_API_KEY' with your actual OpenAI API key
llm = OpenAI(openai_api_key="Enter your API key here")
prompt = PromptTemplate(...)  # Define your prompt template here

chatbot = create_chatbot(llm, prompt)

while True:
    user_input = input("Enter your bowling attempt: ")
    if user_input.lower() == "quit":
        break
    print(f"User Input: {user_input}")  # Debug print statement
    response = chatbot.invoke(user_input)
    print(response)
