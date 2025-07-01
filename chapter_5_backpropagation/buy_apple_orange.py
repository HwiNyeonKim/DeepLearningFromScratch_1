from add_layer import AddLayer
from multiply_layer import MultiplyLayer


apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

multiply_apple_layer = MultiplyLayer()
multiply_orange_layer = MultiplyLayer()

add_apple_orange_layer = AddLayer()
multiply_tax_layer = MultiplyLayer()

# forward propagation
apple_price = multiply_apple_layer.forward(apple, apple_num)
orange_price = multiply_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
total_price = multiply_tax_layer.forward(all_price, tax)

print(f"forward propagation: total price is {total_price}")

# backward propagation
dprice = 1
dall_price, dtax = multiply_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dapple, dapple_num = multiply_apple_layer.backward(dapple_price)
dorange, dorange_num = multiply_orange_layer.backward(dorange_price)

print(f"backward propagation: dapple is {dapple}, dapple_num is {dapple_num}, dorange is {dorange}, dorange_num is {dorange_num}, dtax is {dtax}")  # noqa