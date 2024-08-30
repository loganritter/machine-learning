import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def parse_configs(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    
    configs = []
    i = 0
    while i < len(data):
        num_atoms = int(data[i].strip())
        i += 1
        energy_and_box = data[i].split()
        i += 1
        
        # Handle "XXX" energy values
        try:
            energy = float(energy_and_box[0])
        except ValueError:
            energy = None  # Placeholder for energy to be predicted
        
        box_dims = np.array(list(map(float, energy_and_box[1:])))
        atoms = []
        for _ in range(num_atoms):
            atom_data = data[i].split()
            coords = list(map(float, atom_data[1:4]))
            atoms.append(coords)
            i += 1
        
        configs.append({'energy': energy, 'box_dims': box_dims, 'atoms': np.array(atoms)})
    return configs

def calculate_distances(atoms, box_dims):
    cutoff = min(box_dims[:3]) / 2  # half the smallest box length
    noble_gas = atoms[-1]
    mof_atoms = atoms[:-1]
    distances = []
    for atom in mof_atoms:
        distance = np.linalg.norm(atom - noble_gas)
        if distance <= cutoff:
            distances.append(distance)
    return distances

def prepare_data(configs, max_distances=10):
    X = []
    y = []
    for config in configs:
        distances = calculate_distances(config['atoms'], config['box_dims'])
        
        # Limit distances to `max_distances` length
        if len(distances) < max_distances:
            distances.extend([0] * (max_distances - len(distances)))  # Padding with zeros
        else:
            distances = distances[:max_distances]  # Truncate if too many distances
        
        features = distances + list(config['box_dims'])
        X.append(features)
        if config['energy'] is not None:
            y.append(config['energy'])
    return np.array(X), np.array(y)

def main():
    # Load and parse the data
    configs_fit = parse_configs('configs.fit')
    
    # Prepare features and labels
    X, y = prepare_data(configs_fit)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train an XGBoost Regressor
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')
    
    # Read and predict new data
    configs_train = parse_configs('configs.train')
    X_new, _ = prepare_data(configs_train)
    y_new_pred = model.predict(X_new)
    
    # Write predictions to the new file
    with open('configs.train', 'r') as f:
        train_data = f.readlines()
    
    with open('configs.predicted', 'w') as f:
        i = 0
        for line in train_data:
            if 'XXX' in line:
                f.write(f'{y_new_pred[i]:.6f}\n')
                i += 1
            else:
                f.write(line)

if __name__ == '__main__':
    main()