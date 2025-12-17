import os  # Importez le module os pour gérer les chemins
import pandas as pd 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib  # Importez joblib pour sauvegarder le modèle
from sklearn.preprocessing import LabelEncoder

def load_data():
    """Charge les données à partir d'un fichier CSV et retourne X et y."""
    # Chemin vers le fichier CSV
    csv_file_path = 'c:/Users/wiwi/OneDrive/Bureau/facteur/backend/data/StudentPerformanceFactors.csv'
    
    # Chargement des données
    try:
        df = pd.read_csv(csv_file_path)
        print("Données chargées avec succès.")
    except FileNotFoundError:
        print(f"Erreur : Le fichier {csv_file_path} n'a pas été trouvé.")
        return None, None

    # Vérification des colonnes dans le DataFrame
    print("Colonnes dans le DataFrame :", df.columns.tolist())

    # Encodage des colonnes catégorielles
    label_encoders = {}
    categorical_columns = ['Attendance', 'Parental_Involvement', 'Access_to_Resources', 
                           'Extracurricular_Activities', 'Motivation_Level', 
                           'Internet_Access', 'Family_Income', 'Teacher_Quality', 
                           'School_Type', 'Peer_Influence', 'Learning_Disabilities', 
                           'Parental_Education_Level', 'Gender', 'Distance_from_Home']

    for column in categorical_columns:
        if column in df.columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))  # Convertir en chaîne pour gérer les NaN
            label_encoders[column] = le  # Gardez une référence si vous avez besoin de décoder plus tard
        else:
            print(f"Avertissement : La colonne '{column}' n'existe pas dans le DataFrame.")

    # Gestion des valeurs NaN
    df.fillna(-1, inplace=True)  # Remplacez les NaN par une valeur, ici -1

    # Assurez-vous que 'Exam_Score' est bien le nom de la colonne cible
    if 'Exam_Score' not in df.columns:
        print("Erreur : La colonne 'Exam_Score' n'existe pas dans le DataFrame.")
        return None, None
    
    X = df.drop(columns=['Exam_Score'])  # Remplacez 'Exam_Score' par le nom réel de la colonne à prédire
    y = df['Exam_Score']  # Remplacez par le nom réel de la colonne
    return X, y

def select_best_model(X, y):
    """Sélectionne le meilleur modèle parmi plusieurs et retourne son nom et son score."""
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "RandomForestRegressor": RandomForestRegressor(),
        "GradientBoostingRegressor": GradientBoostingRegressor(),
        "KNeighborsRegressor": KNeighborsRegressor(),
        "SVR": SVR()
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vérification des types de données
    print("Types de données dans X_train:")
    print(X_train.dtypes)
    print("\nValeurs uniques dans chaque colonne:")
    for col in X_train.columns:
        print(f"{col}: {X_train[col].unique()}")
    
    best_model_name = None
    best_score = float('inf')
    best_model = None  # Variable pour stocker le meilleur modèle
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = mean_squared_error(y_test, y_pred)
        
        print(f"{model_name} - Score (MSE): {score}")
        
        if score < best_score:
            best_score = score
            best_model_name = model_name
            best_model = model  # Gardez une référence au meilleur modèle
    
    # Sauvegarder le meilleur modèle
    model_path = os.path.join(os.path.dirname(__file__), 'best_model.pkl')  # Chemin vers le fichier
    joblib.dump(best_model, model_path)  # Enregistrez le modèle dans un fichier
    print(f"Meilleur modèle sauvegardé : {best_model_name} dans {model_path}")

    return best_model_name, best_score

# Pour charger les données et sélectionner le meilleur modèle
if __name__ == "__main__":
    X, y = load_data()
    if X is not None and y is not None:  # Vérifiez que les données sont chargées correctement
        best_model_name, best_score = select_best_model(X, y)
        print(f"Meilleur modèle : {best_model_name}, Meilleur score : {best_score}")
