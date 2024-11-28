import numpy as np
from modules.ml_option_3 import DeepMAgePredictor
from modules.ml_common import DeepMAgeBase


def main():
    input_dim = 1000  # Number of CpG sites
    model_path = DeepMAgeBase.model_path

    # Load trained model
    predictor = DeepMAgePredictor.load_model(model_path, input_dim)

    # Example methylation beta values for a sample
    example_sample = np.random.rand(1, input_dim)  # Replace with real data
    prediction = predictor.predict_batch(example_sample)
    print(f"Predicted Age: {prediction[0]}")


if __name__ == "__main__":
    main()
