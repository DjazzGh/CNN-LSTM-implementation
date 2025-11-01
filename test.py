def evaluate(model, loader):
    correct = 0
    for Xb, yb in loader:
        logits = model.forward(Xb)
        pred = np.argmax(logits, axis=1)
        correct += np.sum(pred == yb)
    return correct / len(loader.dataset)