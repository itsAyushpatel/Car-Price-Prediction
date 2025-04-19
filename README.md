# Used Car Price Prediction System

This project implements a machine learning-based system for predicting used car prices using various features like brand, model, year, kilometers driven, and other specifications. Built with Streamlit for an interactive web interface.

![Screenshot 2025-04-19 114440](https://github.com/user-attachments/assets/ef4e59ea-8da7-4222-9515-35f8bdedb29a)

## 👨‍💻 Author
**Ayush Patel**

## 🚗 Project Overview
A comprehensive machine learning application that predicts the price of used cars based on their specifications. The system uses multiple regression models including:

- Linear Regression
- Random Forest
- Gradient Boosting

The best performing model is automatically selected based on RMSE (Root Mean Square Error).

## 🌟 Features
- **Data Preprocessing**: Handles missing values, outliers, and data formatting
- **Multiple ML Models**: Compares performance of different algorithms
- **Interactive UI**: User-friendly interface built with Streamlit
- **Real-time Predictions**: Get instant price predictions for car specifications
- **Visualization**: Actual vs Predicted price comparison plots
- **Data Validation**: Ensures data quality and handles edge cases

## 🛠️ Technologies Used
- Python 3.x
- Pandas & NumPy for data manipulation
- Scikit-learn for machine learning models
- Streamlit for web interface
- Matplotlib & Seaborn for visualizations

## 📁 Project Structure
```
car-price-prediction/
│
├── car_price_prediction_2.py    # Main application code
├── used_car_dataset.csv        # Training dataset
├── actual_vs_predicted.png     # Visualization output
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ayushpatel/car-price-prediction.git
   cd car-price-prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run car_price_prediction_2.py
   ```

## 📊 Dataset
The system requires a CSV file named `used_car_dataset.csv` with the following columns:
- Brand
- model
- Year
- Age
- kmDriven
- Transmission
- Owner
- FuelType
- AskPrice

## 🔧 How It Works

1. **Data Loading**: The application loads and preprocesses the car dataset
2. **Feature Engineering**: Combines brand and model to create a more specific feature
3. **Data Preprocessing**: Handles outliers, converts data types, and scales features
4. **Model Training**: Trains three different models and selects the best performer
5. **Prediction**: Uses the best model to predict car prices based on user input

## 📈 Model Performance
The application displays:
- Root Mean Square Error (RMSE)
- R-squared (R²) score
- Mean Absolute Error (MAE)

## 💻 Usage

1. Run the application using Streamlit
2. View model performance metrics in the dashboard
3. Enter car specifications in the input form:
   - Select brand, transmission type, owner type, and fuel type
   - Adjust year and kilometers driven
   - Choose the specific model of the selected brand
4. Click "Predict Price" to get an estimated price
5. View the predicted price and car age

## 🔍 Key Functions

- `load_and_preprocess_data()`: Handles data loading and initial preprocessing
- `prepare_data_for_modeling()`: Prepares features and target for ML models
- `train_and_evaluate_models()`: Trains multiple models and evaluates performance
- `visualize_predictions()`: Creates actual vs predicted price plots

## ⚙️ Configuration
The application includes reasonable default values for:
- Price range: ₹50,000 to ₹10,000,000
- Kilometers driven: 1,000 to 500,000
- Train-test split ratio: 80-20

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
This project is open source and available under the MIT License.


## 🙏 Acknowledgments
- Thanks to all contributors and users of this project
- Inspired by the need for transparent used car pricing
