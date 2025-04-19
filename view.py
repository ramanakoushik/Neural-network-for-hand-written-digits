import matplotlib.pyplot as plt
from data import get_mnist


images, labels = get_mnist()


while True:
    try:
        
        index = int(input(f"Enter an index (0 - {len(images) - 1}, or -1 to exit): "))
        if index == -1:
            print("Exiting image viewer.")
            break
        if index < 0 or index >= len(images):
            print("Invalid index. Please enter a number between 0 and 59999.")
            continue
        
        
        plt.imshow(images[index].reshape(28, 28), cmap="Greys")
        plt.title(f"Label: {labels[index].argmax()}")
        plt.axis("off")
        plt.show()
    except ValueError:
        print("Please enter a valid integer.")
