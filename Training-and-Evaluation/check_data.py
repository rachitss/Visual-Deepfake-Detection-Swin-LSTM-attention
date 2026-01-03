import numpy as np

data_path = '<path-to-train-test-split>'
data = np.load(data_path, allow_pickle=True)

x_train = data['x_train']
y_train = data['y_train']
x_val = data['x_val']
y_val = data['y_val']

print(f"Training set size: {len(x_train)}")
print(f"Number of training real samples: {np.sum(y_train == 0)}")
print(f"Number of training fake samples: {np.sum(y_train == 1)}")

print(f"Validation set size: {len(x_val)}")
print(f"Number of validation real samples: {np.sum(y_val == 0)}")
print(f"Number of validation fake samples: {np.sum(y_val == 1)}")

print("First 5 train x and y:")
for i in range(5):
    print(f"X: {x_train[i]}, Y: {y_train[i]}")

print("First 5 val x and y:")
for i in range(5):
    print(f"X: {x_val[i]}, Y: {y_val[i]}")
