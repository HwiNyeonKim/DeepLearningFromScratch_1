from multiply_layer import MultiplyLayer


apple = 100
apple_num = 2
tax = 1.1

multiply_apple_layer = MultiplyLayer()
multiply_tax_layer = MultiplyLayer()

# forward propagation
apple_price = multiply_apple_layer.forward(apple, apple_num)
total_price = multiply_tax_layer.forward(apple_price, tax)

print(f"forward propagation: total price is {total_price}")

# backward propagation
dprice = 1
dapple_price, dtax = multiply_tax_layer.backward(dprice)
dapple, dapple_num = multiply_apple_layer.backward(dapple_price)

print(f"backward propagation: dapple is {dapple}, dapple_num is {dapple_num}, dtax is {dtax}")  # noqa