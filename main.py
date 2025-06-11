import tensorflow as tf


if __name__ == "__main__":
    # Print TensorFlow version
    print("TensorFlow version:", tf.__version__)

    # List available physical devices
    print("Available physical devices:")
    for device in tf.config.list_physical_devices():
        print(device)

    # Check if GPU is available
    print("Is GPU available:", tf.config.list_physical_devices('GPU'))

