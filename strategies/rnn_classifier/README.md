Notes
---
2.18.2018 - simple lstm classifier still overfitting. Tried smaller capacity
model, (e.g 1 layer, smaller hidden size), and reduced learning rate. Also found
that I was not setting 'self.training=True' for F.dropout(). Accuracy no better
then random 30-35%, and validation loss blows out while training loss steadily
declines. Trying to start with smaller capacity model (a simple feed forward
network) and see how that works out. Its quite possible the data set is a) just too
small, as working daily open to close returns, and b) not rich enough as all is
based off price.
