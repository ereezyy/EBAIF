import timeit
import random

def current_accumulate_and_average(total_gradients, gradients_list, n_samples):
    # Accumulate
    for gradients in gradients_list:
        for param_name in total_gradients:
            if isinstance(total_gradients[param_name][0], list):  # Weight matrix
                for i in range(len(total_gradients[param_name])):
                    for j in range(len(total_gradients[param_name][i])):
                        total_gradients[param_name][i][j] += gradients[param_name][i][j]
            else:  # Bias vector
                for i in range(len(total_gradients[param_name])):
                    total_gradients[param_name][i] += gradients[param_name][i]

    # Average
    for param_name in total_gradients:
        if isinstance(total_gradients[param_name][0], list):  # Weight matrix
            for i in range(len(total_gradients[param_name])):
                for j in range(len(total_gradients[param_name][i])):
                    total_gradients[param_name][i][j] /= n_samples
        else:  # Bias vector
            for i in range(len(total_gradients[param_name])):
                total_gradients[param_name][i] /= n_samples

def optimized_accumulate_and_average(total_gradients, gradients_list, n_samples):
    # Accumulate
    for gradients in gradients_list:
        for param_name, total_val in total_gradients.items():
            grad_val = gradients[param_name]
            if isinstance(total_val[0], list):
                for total_row, grad_row in zip(total_val, grad_val):
                    for i in range(len(total_row)):
                        total_row[i] += grad_row[i]
            else:
                for i in range(len(total_val)):
                    total_val[i] += grad_val[i]

    # Average
    for param_name, total_val in total_gradients.items():
        if isinstance(total_val[0], list):
            total_gradients[param_name] = [[cell / n_samples for cell in row] for row in total_val]
        else:
            total_gradients[param_name] = [cell / n_samples for cell in total_val]

def current_update_parameters(parameters, gradients, learning_rate):
    updated_params = {}
    for param_name, param_matrix in parameters.items():
        grad_matrix = gradients[param_name]
        if 'W' in param_name:  # Weight matrix
            updated_matrix = []
            for i in range(len(param_matrix)):
                updated_row = []
                for j in range(len(param_matrix[i])):
                    new_val = param_matrix[i][j] - learning_rate * grad_matrix[i][j]
                    updated_row.append(new_val)
                updated_matrix.append(updated_row)
        else:  # Bias vector
            updated_matrix = []
            for i in range(len(param_matrix)):
                new_val = param_matrix[i] - learning_rate * grad_matrix[i]
                updated_matrix.append(new_val)
        updated_params[param_name] = updated_matrix
    return updated_params

def optimized_update_parameters(parameters, gradients, learning_rate):
    updated_params = {}
    for param_name, param_matrix in parameters.items():
        grad_matrix = gradients[param_name]
        if 'W' in param_name:
            updated_params[param_name] = [
                [p - learning_rate * g for p, g in zip(p_row, g_row)]
                for p_row, g_row in zip(param_matrix, grad_matrix)
            ]
        else:
            updated_params[param_name] = [p - learning_rate * g for p, g in zip(param_matrix, grad_matrix)]
    return updated_params

# Setup data
rows, cols = 64, 64
n_samples = 10
params = {
    'W1': [[random.random() for _ in range(cols)] for _ in range(rows)],
    'b1': [random.random() for _ in range(rows)],
    'W2': [[random.random() for _ in range(cols)] for _ in range(rows)],
    'b2': [random.random() for _ in range(rows)]
}
gradients_list = [
    {
        'W1': [[random.random() for _ in range(cols)] for _ in range(rows)],
        'b1': [random.random() for _ in range(rows)],
        'W2': [[random.random() for _ in range(cols)] for _ in range(rows)],
        'b2': [random.random() for _ in range(rows)]
    } for _ in range(n_samples)
]

def get_tg():
    return {
        'W1': [[0.0 for _ in range(cols)] for _ in range(rows)],
        'b1': [0.0 for _ in range(rows)],
        'W2': [[0.0 for _ in range(cols)] for _ in range(rows)],
        'b2': [0.0 for _ in range(rows)]
    }

def benchmark():
    n = 200

    t_curr_acc = timeit.timeit(lambda: current_accumulate_and_average(get_tg(), gradients_list, n_samples), number=n)
    t_opt_acc = timeit.timeit(lambda: optimized_accumulate_and_average(get_tg(), gradients_list, n_samples), number=n)

    avg_grads = gradients_list[0]
    t_curr_upd = timeit.timeit(lambda: current_update_parameters(params, avg_grads, 0.01), number=n)
    t_opt_upd = timeit.timeit(lambda: optimized_update_parameters(params, avg_grads, 0.01), number=n)

    print(f"Accumulate & Average - Current: {t_curr_acc:.4f}s")
    print(f"Accumulate & Average - Optimized: {t_opt_acc:.4f}s")
    print(f"Improvement: {(t_curr_acc - t_opt_acc) / t_curr_acc * 100:.2f}%")

    print(f"\nUpdate Parameters - Current: {t_curr_upd:.4f}s")
    print(f"Update Parameters - Optimized: {t_opt_upd:.4f}s")
    print(f"Improvement: {(t_curr_upd - t_opt_upd) / t_curr_upd * 100:.2f}%")

if __name__ == "__main__":
    benchmark()
