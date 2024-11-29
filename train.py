import numpy as np
import json
import matplotlib.pyplot as plt

def display(mileage, price, predictions, theta0, theta1, mileage_mean, mileage_std, price_mean, price_std, epoch, is_final=False):
    """Affiche le graphique des points et de la régression."""
    plt.clf()
    plt.scatter(mileage * mileage_std + mileage_mean, price * price_std + price_mean, color="blue", label="Data points")
    line_x = np.linspace(min(mileage), max(mileage), 100)
    line_y = theta0 + theta1 * line_x
    plt.plot(line_x * mileage_std + mileage_mean, line_y * price_std + price_mean, color="red", label=f"Epoch {epoch}")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.title("Linear Regression Training")
    plt.legend()
    if is_final:
        plt.show()
    else:
        plt.pause(0.05)

def train_model(data_file, learning_rate, epochs, update_frequency=50):
    # Charger le dataset
    data = np.genfromtxt(data_file, delimiter=",", skip_header=1)
    mileage = data[:, 0]
    price = data[:, 1]
    
    #calcul des ecart-types et moyennes
    mileage_mean = np.mean(mileage)
    mileage_std = np.std(mileage)
    price_mean = np.mean(price)
    price_std = np.std(price) 

    # Normalisation des valeurs
    mileage = (mileage - mileage_mean) / mileage_std
    price = (price - price_mean) / price_std
    
    m = len(mileage)  # Nombre d'exemples
    theta0, theta1 = 0, 0  # Initialiser les paramètres
    
    # Initialiser le graphique
    plt.figure(figsize=(10, 6))
    
    # Descente de gradient
    for epoch in range(epochs):
        # Prédictions
        predictions = theta0 + theta1 * mileage
        
        # Calcul des gradients
        error = predictions - price
        tmp_theta0 = learning_rate * (1 / m) * np.sum(error)
        tmp_theta1 = learning_rate * (1 / m) * np.sum(error * mileage)
        
        # Mise à jour des paramètres
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1
        
        # Calcul des métriques
        #MSE (Mean Squared Error) : Moyenne des carrés des erreurs. Indique à quel point les 
        #   prédictions sont proches des vraies valeurs (plus petit = mieux).
        #RMSE (Root Mean Squared Error) : Racine carrée du MSE. Indique l'erreur moyenne en unités de la variable cible.
        #R² (Coefficient de détermination) : Mesure la proportion de variance expliquée par le modèle (1 = parfait, > 0.7 = bon).

        mse = np.mean(error ** 2)
        rmse = np.sqrt(mse)
        r_squared = 1 - (np.sum((price - predictions) ** 2) / np.sum((price - np.mean(price)) ** 2))
        
        # Affichage des métriques et mise à jour du graphique toutes les `update_frequency`
        if epoch % update_frequency == 0:
            print(f"Epoch {epoch}: MSE = {mse:.4f}, RMSE = {rmse:.4f}, R² = {r_squared:.4f}")
            display(mileage, price, predictions, theta0, theta1, mileage_mean, mileage_std, price_mean, price_std, epoch)
    
    # Affichage du graphique final
    display(mileage, price, predictions, theta0, theta1, mileage_mean, mileage_std, price_mean, price_std, "Final", is_final=True)
    
    # Sauvegarde du modèle
    with open("model.json", "w") as f:
        json.dump({
            "theta0": theta0,
            "theta1": theta1,
            "mileage_mean": mileage_mean,
            "mileage_std": mileage_std,
            "price_mean": price_mean,
            "price_std": price_std
        }, f)
    
    print("Training complete. Model saved!")
    print(f"Final MSE: {mse:.4f}, RMSE: {rmse:.4f}, Final R²: {r_squared:.4f}")

if __name__ == "__main__":
    train_model("data.csv", learning_rate=0.005, epochs=2000, update_frequency=50)
