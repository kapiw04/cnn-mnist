# cnn-mnist

## Instalation

```sh
git clone https://github.com/kapiw04/cnn-mnist.git
pip install -r requirements.txt
```

## Usage

To use default hyperparameters for CNN, which I found to be the best ones for the highest accuracy run this command

```sh
python3 cnn.py
```

To configure a parameter add \<paremeter name\>=\<value\>

| Parameter       | Description                            | Default |
| --------------- | -------------------------------------- | ------- |
| `batch_size`    | Number of samples per gradient update  | 64      |
| `learning_rate` | Learning rate for the optimizer        | 0.01    |
| `num_epochs`    | Number of epochs to train the model    | 10      |
| `momentum`      | Momentum for the optimizer             | 0.9     |
| `kernel_size`   | The size of the convolutional kernel   | 3       |
| `dropout_rate`  | Dropout rate for regularization        | 0.5     |
| `l1_size`       | Size of the first linear layer         | 128     |
| `l2_size`       | Size of the second linear layer        | 64      |
| `conv_1_size`   | Size of the first convolutional layer  | 32      |
| `conv_2_size`   | Size of the second convolutional layer | 64      |

Example:

```sh
python3 cnn.py batch_size=32 dropout_rate=0.4
```
