import deepxde as dde


def jacobian_matrix(x, y):
    return dde.grad.jacobian(ys=y, xs=x)


def dy_x(x, y, input_var_id: int, output_var_id: int):
    return dde.grad.jacobian(ys=y, xs=x, i=input_var_id, j=output_var_id)


def hessian_matrix(x, y):
    return dde.grad.hessian(ys=y, xs=x)


def dyy_x(x, y, input_var1_id: int, input_var2_id: int, output_var_id: int):
    return dde.grad.hessian(ys=y, xs=x, i=input_var1_id, j=input_var2_id, component=output_var_id)
