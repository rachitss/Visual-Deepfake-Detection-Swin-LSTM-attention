import numpy as np
np.random.seed(42) # seed for reproducibility

data_path = '<path-to-train-test-split>'
bal_data_path = '<output-path-for-balanced-split>'

# load data
data = np.load(data_path, allow_pickle=True)
x_train = data['x_train']
y_train = data['y_train']
x_val = data['x_val']
y_val = data['y_val']

# find train label indices
train_real_idx = np.where(y_train == 0)[0]
train_fake_idx = np.where(y_train == 1)[0]
train_n = len(train_real_idx)

# randomly undersample fake class
train_keep = np.concatenate([
	train_real_idx,
	np.random.choice(train_fake_idx, train_n, replace=False)
])

# shuffle for randomness
np.random.shuffle(train_keep)

# balanced train splits
x_train_bal = x_train[train_keep]
y_train_bal = y_train[train_keep]

# find val label indices
val_real_idx = np.where(y_val == 0)[0]
val_fake_idx = np.where(y_val == 1)[0]
val_n = len(val_real_idx)

# randomly undersample fake class
val_keep = np.concatenate([
	val_real_idx,
	np.random.choice(val_fake_idx, val_n, replace=False)
])

# shuffle for randomness
np.random.shuffle(val_keep)

# balanced val splits
x_val_bal = x_val[val_keep]
y_val_bal = y_val[val_keep]

# save balanced splits
np.savez(
	bal_data_path,
	x_train=x_train_bal,
	y_train=y_train_bal,
	x_val=x_val_bal,
	y_val=y_val_bal
)

print(f"Balanced dataset saved to {bal_data_path}")
