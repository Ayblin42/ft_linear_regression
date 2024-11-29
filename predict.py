import json

import json

def predict_price(mileage):
    with open("model.json", "r") as f:
        model = json.load(f)
    
    theta0 = model["theta0"]
    theta1 = model["theta1"]
    mileage_mean = model["mileage_mean"]
    mileage_std = model["mileage_std"]
    price_mean = model["price_mean"]
    price_std = model["price_std"]
    
    # Normalisation du kilométrage
    normalized_mileage = (mileage - mileage_mean) / mileage_std
    
    # Prédiction normalisée
    normalized_price = theta0 + theta1 * normalized_mileage
    
    # Dénormalisation du prix
    estimated_price = normalized_price * price_std + price_mean
    return estimated_price

if __name__ == "__main__":
    mileage = float(input("Enter mileage of the car: "))
    price = predict_price(mileage)
    print(f"Estimated price: {price:.2f}")