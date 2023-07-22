import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


class Perceptron(tf.keras.Model):
    """Perceptron model class"""
    def __init__(self):
        """Constructor function."""
        super(Perceptron, self).__init__()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        """Forward pass."""
        return self.dense(inputs)


class Trainer():
    """Trainer class"""
    def __init__(self, model, loss, optimizer, train_examples):
        """Constructor function."""
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_examples = train_examples
        self.weight_history = []
        self.grand_example_dict = {}
        self.loss_example_dict = {}
        self.init_example_dicts()
        self.create_train_dataset()

    def init_example_dicts(self):
        """Function that initilizes the dictionary of values per example."""
        for i in range(len(self.train_examples)):
            self.grand_example_dict[f"example_{i+1}"]=[]
            self.loss_example_dict[f"example_{i+1}"]=[]    
     
    def create_train_dataset(self):
        """Convert training examples to TensorFlow tensors."""
        train_inputs = tf.constant([[xi1, xi2] for xi1, xi2, _ in self.train_examples], dtype=tf.float32)
        train_targets = tf.constant([[target] for _, _, target in self.train_examples], dtype=tf.float32)

        # Create a TensorFlow Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets))

        # Shuffle and batch the training dataset
        self.train_dataset = train_dataset.shuffle(len(self.train_examples)).batch(1)

    def train_step(self, inputs, targets):
        """Train step function."""
        with tf.GradientTape() as tape:
            # Forward pass
            logits = self.model(inputs)
            loss_value = self.loss(targets, logits)

            # Compute gradients
            gradients = tape.gradient(loss_value, self.model.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    
        return loss_value, gradients

    def train(self, num_epochs, print_step_freq=10):
        """Custom training function"""
        for epoch in range(num_epochs):
            if((epoch+1) % print_step_freq) == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}")
            for batch_inputs, batch_targets in self.train_dataset:
                for example_input, example_target in zip(batch_inputs, batch_targets):

                    # Expand training data dims
                    example_input = tf.expand_dims(example_input, axis=0)
                    example_target = tf.expand_dims(example_target, axis=0)

                    # Compute loss
                    loss, gradients = self.train_step(example_input, example_target)

                    # Compute GraNd scores
                    grand = tf.norm(gradients[0].numpy())

                    # Store the weights at each step
                    self.weight_history.append(self.model.get_weights())

                    # Store the loss and GraNd scores per training example
                    self.store_example_scores(example_input, loss, grand)

                    # Print stats
                    if((epoch+1) % print_step_freq) == 0:
                        print(f"\t\tx={example_input}, y={example_target}, loss: {loss:.4f}, GraNd: {grand:.4f}")

    def store_example_scores(self, x, loss_value, grand_value):
        """Naive implementation for storing GraNd and loss values per example."""
        # Convert x from tensor to numpy array
        x_np = x.numpy()
        # 2D points
        example_1_np = np.array([0.25, 0.25]).reshape(x_np.shape)
        example_2_np = np.array([-0.25, -0.5]).reshape(x_np.shape)
        example_3_np = np.array([-0.1, 0.5]).reshape(x_np.shape)
        example_4_np = np.array([0.2, 0.3]).reshape(x_np.shape)
        example_5_np = np.array([0.5, 0.2]).reshape(x_np.shape)
        example_6_np = np.array([0.35, 0.5]).reshape(x_np.shape)
        example_7_np = np.array([0.6, 0.4]).reshape(x_np.shape)
        example_8_np = np.array([0.9, 0.6]).reshape(x_np.shape)
        example_9_np = np.array([1.25, 0.4]).reshape(x_np.shape)
        example_10_np = np.array([0.1, 0.1]).reshape(x_np.shape)
        # Binary points
        example_b1_np = np.array([0.0, 0.0]).reshape(x_np.shape)
        example_b2_np = np.array([0.0, 1.0]).reshape(x_np.shape)
        example_b3_np = np.array([1.0, 0.0]).reshape(x_np.shape)
        example_b4_np = np.array([1.0, 1.0]).reshape(x_np.shape)
        # example_1
        if(np.allclose(x_np, example_1_np) or np.allclose(x_np, example_b1_np)):
            self.loss_example_dict["example_1"].append(loss_value)
            self.grand_example_dict["example_1"].append(grand_value.numpy())
        # example_2
        elif(np.allclose(x_np, example_2_np) or np.allclose(x_np, example_b2_np)):
            self.loss_example_dict["example_2"].append(loss_value)
            self.grand_example_dict["example_2"].append(grand_value.numpy())
        # example_3
        elif(np.allclose(x_np, example_3_np) or np.allclose(x_np, example_b3_np)):
            self.loss_example_dict["example_3"].append(loss_value)
            self.grand_example_dict["example_3"].append(grand_value.numpy())
        # example_4
        elif(np.allclose(x_np, example_4_np) or np.allclose(x_np, example_b4_np)):
            self.loss_example_dict["example_4"].append(loss_value)
            self.grand_example_dict["example_4"].append(grand_value.numpy())
        # example_5
        elif(np.allclose(x_np, example_5_np)):
            self.loss_example_dict["example_5"].append(loss_value)
            self.grand_example_dict["example_5"].append(grand_value.numpy())
        # example_6
        elif(np.allclose(x_np, example_6_np)):
            self.loss_example_dict["example_6"].append(loss_value)
            self.grand_example_dict["example_6"].append(grand_value.numpy())
        # example_7
        elif(np.allclose(x_np, example_7_np)):
            self.loss_example_dict["example_7"].append(loss_value)
            self.grand_example_dict["example_7"].append(grand_value.numpy())
        # example_8
        elif(np.allclose(x_np, example_8_np)):
            self.loss_example_dict["example_8"].append(loss_value)
            self.grand_example_dict["example_8"].append(grand_value.numpy())
        # example_9
        elif(np.allclose(x_np, example_9_np)):
            self.loss_example_dict["example_9"].append(loss_value)
            self.grand_example_dict["example_9"].append(grand_value.numpy())
        # example_10
        elif(np.allclose(x_np, example_10_np)):
            self.loss_example_dict["example_10"].append(loss_value)
            self.grand_example_dict["example_10"].append(grand_value.numpy())
        # Other
        else:
            print(f"Input example {x_np} not found!")

    def plot_decision_boundary(self):
        """Function that plots a linear decision boundary on the 2D plane"""
        # Extract the model parameters
        weights = self.model.get_weights()
        w1, w2 = weights[0]
        b = weights[1]

        # Print slope and intercept of decision boundary line
        print(f"Slope:     {-w1/w2}")
        print(f"Intercept: {-b/w2}")

        # Convert the train_examples to a NumPy array
        train_examples_np = np.array(self.train_examples)

        # Separate the input features and labels from train_examples
        input_features = train_examples_np[:, :-1]
        labels = train_examples_np[:, -1]
        names = [f"e{i+1}" for i in range(len(self.train_examples))]

        # Separate the input features based on the class labels
        class_0_points = input_features[labels == 0]
        class_1_points = input_features[labels == 1]

        # Plot the points
        plt.scatter(class_0_points[:, 0], class_0_points[:, 1], label='Class 0')
        plt.scatter(class_1_points[:, 0], class_1_points[:, 1], label='Class 1')

        # Add point names
        for x, y, name in zip(input_features[:,0], input_features[:,1], names):
            plt.text(x, y, name, verticalalignment='bottom', horizontalalignment='right')

        # Plot the decision boundary line
        x = np.linspace(-0.5, 1.5, 100)
        y = (-b - w1 * x) / w2
        plt.plot(x, y, color='red', label='Decision Boundary')

        # Add labels and legend
        plt.title("Perceptron 2D-Points")
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.ylim(-1.0, 1.5)
        plt.xlim(-0.5, 1.5)
        plt.legend()

        # Show the plot
        plt.show()
    

    def plot_parameter_evolution(self):
        """Plot parameter values vs iterations of training."""
        # Extract the model parameters
        w1_history = []
        w2_history = []
        b_history = []

        for w in self.weight_history:
            w1_history.append(w[0][0])
            w2_history.append(w[0][1])
            b_history.append(w[1])

        # Plot evolution of parameter values with time
        plt.figure(figsize=(8,6))
        plt.plot(list(range(1, len(w1_history)+1)), w1_history, label="w1")
        plt.plot(list(range(1, len(w2_history)+1)), w2_history, label="w2")
        plt.plot(list(range(1, len(b_history)+1)), b_history, label="b")
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    def plot_grand_evolution(self):
        """Plot evolution of GraNd values with time."""
        plt.figure(figsize=(8,6))
        for key, val in self.grand_example_dict.items():
            plt.plot(list(range(1, len(val)+1)), val, label=key)
        plt.xlabel('Epoch')
        plt.ylabel('GraNd')
        plt.legend()
        plt.show()

    def plot_loss_evolution(self):
        """Plot evolution of loss values with time."""
        plt.figure(figsize=(8,6))
        for key, val in self.loss_example_dict.items():
            plt.plot(list(range(1, len(val)+1)), val, label=key)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


def create_train_examples(type="AND"):
    """Function that creates 2D dataset points."""
    if type == "2D":
        return [
            (0.25, 0.25, 0.0),  # example_1
            (-0.25, -0.5, 0.0), # example_2
            (-0.1, 0.5, 0.0),   # example_3
            (0.2, 0.3, 0.0),    # example_4
            (0.5, 0.2, 0.0),    # example_5 (boundary)
            (0.35, 0.5, 1.0),   # example_6 (boundary)
            (0.6, 0.4, 1.0),    # example_7
            (0.9, 0.6, 1.0),    # example_8
            (1.25, 0.4, 1.0),   # example_9
            (0.1, 0.1, 1.0),    # example_10 (misclassified)
        ]
    elif type == "AND":
        return [
            (0.0, 0.0, 0.0), # example_1
            (0.0, 1.0, 0.0), # example_2
            (1.0, 0.0, 0.0), # example_3
            (1.0, 1.0, 1.0)  # example_4
        ]
    elif type == "OR":
        return [
            (0.0, 0.0, 0.0), # example_1
            (0.0, 1.0, 1.0), # example_2
            (1.0, 0.0, 1.0), # example_3
            (1.0, 1.0, 1.0)  # example_4   
        ]
    elif type == "NAND":
        return [
            (0.0, 0.0, 1.0), # example_1
            (0.0, 1.0, 1.0), # example_2
            (1.0, 0.0, 1.0), # example_3
            (1.0, 1.0, 0.0)  # example_4
        ]
    elif type == "NOR":
        return [
            (0.0, 0.0, 1.0), # example_1
            (0.0, 1.0, 0.0), # example_2
            (1.0, 0.0, 0.0), # example_3
            (1.0, 1.0, 0.0)  # example_4
        ]
    elif type == "XOR":
        return [
            (0.0, 0.0, 0.0), # example_1
            (0.0, 1.0, 1.0), # example_2
            (1.0, 0.0, 1.0), # example_3
            (1.0, 1.0, 0.0)  # example_4
        ]
    else:
        print("Wrong value provided. Defaulting to `AND`.")
        return [
            (0.0, 0.0, 0.0), # example_1
            (0.0, 1.0, 0.0), # example_2
            (1.0, 0.0, 0.0), # example_3
            (1.0, 1.0, 1.0)  # example_4
        ]
    

def plot_train_examples(train_examples):
    """Function that plots 2D points on the plane."""
    # Convert the train_examples to a NumPy array
    train_examples_np = np.array(train_examples)

    # Separate the input features and labels from train_examples
    input_features = train_examples_np[:, :-1]
    labels = train_examples_np[:, -1]
    names = [f"e{i+1}" for i in range(len(train_examples))]
    
    # Separate the input features based on the class labels
    class_0_points = input_features[labels == 0]
    class_1_points = input_features[labels == 1]

    # Plot the points
    plt.scatter(class_0_points[:, 0], class_0_points[:, 1], label='Class 0')
    plt.scatter(class_1_points[:, 0], class_1_points[:, 1], label='Class 1')

    # Add point names
    for x, y, name in zip(input_features[:,0], input_features[:,1], names):
        plt.text(x, y, name, verticalalignment='bottom', horizontalalignment='right')

    # Add labels and legend
    plt.title("Training Examples")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.ylim(-1.0, 1.5)
    plt.xlim(-0.5, 1.5)
    plt.legend()

    # Show the plot
    plt.show()