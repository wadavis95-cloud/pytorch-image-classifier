def train_model(model, train_loader, criterion, optimizer, device, epochs=3):
    for epoch in range(epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")