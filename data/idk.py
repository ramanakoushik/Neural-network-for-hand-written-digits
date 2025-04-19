import pygame
import numpy as np
import sys
import matplotlib.pyplot as plt
from data import get_mnist  # Assuming get_mnist function is defined as before

# Initialize Pygame and set up the drawing window
pygame.init()

# Set up the display window
WIDTH, HEIGHT = 280, 280  # Canvas size (28x28)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw a Digit")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Initialize the canvas
canvas = pygame.Surface((WIDTH, HEIGHT))
canvas.fill(WHITE)

# Initialize the trained neural network parameters
# Assuming the weights and biases are already trained and loaded
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))  # Weights: Input to hidden
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))   # Weights: Hidden to output
b_i_h = np.zeros((20, 1))  # Biases: Hidden layer
b_h_o = np.zeros((10, 1))  # Biases: Output layer

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtracting max for numerical stability
    return exp_x / np.sum(exp_x, axis=0)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def cross_entropy_loss(o, label):
    return -np.sum(label * np.log(o + 1e-10))  # Add a small constant for numerical stability

def preprocess_image(image_surface):
    """Convert the drawn image to the format expected by the model"""
    # Convert surface to a numpy array
    image_array = pygame.surfarray.array3d(image_surface)  # 3D (R, G, B)
    image_array = np.mean(image_array, axis=2)  # Convert to grayscale (average R, G, B)
    
    # Resize image to 28x28
    image_resized = pygame.transform.scale(image_surface, (28, 28))
    image_resized_array = pygame.surfarray.array3d(image_resized)  # Convert resized surface to array

    # Convert the 3D array (R, G, B) to grayscale (mean of the channels)
    grayscale_image = np.mean(image_resized_array, axis=2)  # Now we have a 2D grayscale image
    
    # Invert the image (white background, black text)
    image_normalized = 255 - grayscale_image  # Invert pixel values
    image_normalized = image_normalized / 255.0  # Normalize pixel values to 0-1

    return image_normalized.reshape(784, 1)  # Flatten to 784x1


def predict_digit(image):
    """Use the trained network to predict the digit"""
    # Forward pass (hidden layer)
    h_pre = b_i_h + w_i_h @ image
    h = relu(h_pre)

    # Forward pass (output layer)
    o_pre = b_h_o + w_h_o @ h
    o = softmax(o_pre)

    # Get the predicted label
    return np.argmax(o)

def reset_canvas():
    """Clear the canvas to start drawing a new digit"""
    canvas.fill(WHITE)

def draw_digit():
    """Main function to handle drawing and prediction"""
    drawing = False
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True  # Start drawing
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False  # Stop drawing

            if drawing:
                # Draw on the canvas
                x, y = pygame.mouse.get_pos()
                pygame.draw.circle(canvas, BLACK, (x, y), 10)
                
        # Display the canvas
        screen.fill(WHITE)
        screen.blit(canvas, (0, 0))

        # If the 'Enter' key is pressed, predict the digit
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RETURN]:
            # Preprocess the image and predict
            preprocessed_image = preprocess_image(canvas)
            prediction = predict_digit(preprocessed_image)
            
            # Display prediction
            print(f"Predicted digit: {prediction}")
            plt.imshow(preprocessed_image.reshape(28, 28), cmap="gray")
            plt.title(f"Predicted: {prediction}")
            plt.show()

            # Reset canvas after prediction
            reset_canvas()

        pygame.display.update()

# Start drawing and prediction loop
draw_digit()



