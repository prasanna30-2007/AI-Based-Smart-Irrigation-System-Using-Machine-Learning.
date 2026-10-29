
import joblib
from irrigation_model import train_model

def main():
    model = train_model()
    joblib.dump(model, "irrigation_model.pkl")
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    main()
  
