import os
package_dir = os.path.dirname(os.path.abspath(__file__))

test_identity_dir = os.path.join(
    package_dir,
    "testing_identity.csv")
test_transaction_dir = os.path.join(
    package_dir,
    "testing_transaction.csv")

train_data_dir = os.path.join(package_dir, "train_data.bin")
val_data_dir = os.path.join(package_dir, "val_data.bin")
test_data_x_dir = os.path.join(package_dir, "test_data_x.csv")
test_data_y_dir = os.path.join(package_dir, "test_data_y.csv")
