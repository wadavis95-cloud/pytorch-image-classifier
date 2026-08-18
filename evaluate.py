import torch


def evaluate_model(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Test Accuracy: {accuracy:.2f}%")

    return accuracy

def calculate_accuracy(model, data_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy

def inspect_predictions(model, test_loader, device, classes, num_images=10):
    model.eval()

    images, labels = next(iter(test_loader))

    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

    for i in range(num_images):
        actual = classes[labels[i].item()]
        prediction = classes[predicted[i].item()]

        if actual == prediction:
            result = "CORRECT"
        else:
            result = "WRONG"

        print(
            f"{i + 1}. Actual: {actual:10} "
            f"Predicted: {prediction:10} "
            f"{result}"
        )

def confusion_matrix_counts(model, test_loader, device, classes):
    model.eval()

    num_classes = len(classes)

    matrix = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.int32
    )

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            for actual, prediction in zip(labels, predicted):
                matrix[actual.item(), prediction.item()] += 1

    print("\nConfusion Matrix")

    print("Rows = Actual")
    print("Columns = Predicted\n")

    print(matrix)

    return matrix  

def most_common_confusions(matrix, classes, top_n=10):
    confusions = []

    for actual in range(len(classes)):
        for predicted in range(len(classes)):
            if actual != predicted:
                count = matrix[actual, predicted].item()

                confusions.append(
                    (
                        count,
                        classes[actual],
                        classes[predicted]
                    )
                )

    confusions.sort(reverse=True)

    print("\nMost Common Confusions")

    for count, actual, predicted in confusions[:top_n]:
        print(
            f"Actual: {actual:10} "
            f"Predicted: {predicted:10} "
            f"Count: {count}"
        )