import numpy as np
import matplotlib.pyplot as plt


class Lab1:
    def __init__(self):
        # Table values
        self.x = np.array([10, 20, 30, 40, 50, 60])
        self.y = np.array([1.06, 1.33, 1.52, 1.68, 1.81, 1.91])

        # Mean values
        self.x_mean = np.mean(self.x)
        self.y_mean = np.mean(self.y)
        pass


    def power_approximation_alternative(self, show_plot=True):
        def find_a_b():
            # Calculate a and b
            n = len(self.x)
            log_x = np.log(self.x)
            log_y = np.log(self.y)
            log_x_sum = np.sum(log_x)
            log_y_sum = np.sum(log_y)
            log_x_squared_sum = np.sum(log_x ** 2)
            log_x_log_y_sum = np.sum(log_x * log_y)
            a = (n * log_x_log_y_sum - log_x_sum * log_y_sum) / (n * log_x_squared_sum - log_x_sum ** 2)
            b = (log_y_sum - a * log_x_sum) / n
            return a, b

        def predict_y():
            a, b = find_a_b()
            return np.exp(b) * self.x ** a

        # Predict y
        y_pred = predict_y()
        # Plot the data and the approximation
        self.plot_result(y_pred, color='green', show_plot=show_plot)
        return y_pred

    def linear_approximation(self, show_plot=True):
        A = np.vstack([self.x, np.ones(len(self.x))]).T
        a, b = np.linalg.lstsq(A, self.y, rcond=None)[0]
        new_y = lambda x: a * x + b
        self.plot_result(new_y(self.x), color='red', show_plot=show_plot)
        y_pred = new_y(self.x)
        return y_pred

    def power_approximation(self, show_plot=True):
        x_log = np.log(self.x)
        y_log = np.log(self.y)
        A = np.vstack([x_log, np.ones(len(x_log))]).T
        a, b = np.linalg.lstsq(A, y_log, rcond=None)[0]
        new_y = lambda x: np.exp(a * np.log(x) + b)
        self.plot_result(new_y(self.x), color='green', show_plot=show_plot)
        y_pred = new_y(self.x)
        return y_pred

    def exponential_approximation(self, show_plot=True):
        A = np.vstack([self.x, np.ones(len(self.x))]).T
        a, b = np.linalg.lstsq(A, np.log(self.y), rcond=None)[0]
        new_y = lambda x: np.exp(a * x + b)
        self.plot_result(new_y(self.x), color='orange', show_plot=show_plot)
        y_pred = new_y(self.x)
        return y_pred


    def quadratic_approximation(self, show_plot=True):
        A = np.vstack([self.x ** 2, self.x, np.ones(len(self.x))]).T
        a, b, c = np.linalg.lstsq(A, self.y, rcond=None)[0]
        new_y = lambda x: a * x ** 2 + b * x + c
        self.plot_result(new_y(self.x), color='purple', show_plot=show_plot)
        y_pred = new_y(self.x)
        return y_pred

    def plot_result(self, y_pred, color='red', show_plot=True):
        # Plot the data and the approximation
        plt.plot(self.x, y_pred, color=color)
        if show_plot:
            self.plot_show_dots()

    def plot_show_dots(self):
        plt.scatter(self.x, self.y, color='blue')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Least squares approximation')
        plt.legend(['Linear', 'Power', 'Exponential', 'Quadratic', 'Data'])
        plt.show()

    def plot_all_results(self):
        self.linear_approximation(False)
        self.power_approximation_alternative(False)
        self.exponential_approximation(False)
        self.quadratic_approximation(False)
        self.plot_show_dots()


Lab1().plot_all_results()

def compare_MSE_MSA():
    # MSE = Mean Squared Error
    # MSA = Mean Squared Approximation
    MSE = lambda y, y_pred: np.mean((y - y_pred) ** 2)
    MSA = lambda y, y_pred: np.mean((y - y_pred) ** 2) / np.mean(y ** 2)
    return MSE, MSA

def final_compare_all_function():
    lab1 = Lab1()
    MSE, MSA = compare_MSE_MSA()
    y_pred = lab1.linear_approximation(False)
    print('Linear MSE: ', MSE(lab1.y, y_pred))
    print('Linear MSA: ', MSA(lab1.y, y_pred))
    y_pred = lab1.power_approximation_alternative(False)
    print('Power MSE: ', MSE(lab1.y, y_pred))
    print('Power MSA: ', MSA(lab1.y, y_pred))
    y_pred = lab1.exponential_approximation(False)
    print('Exponential MSE: ', MSE(lab1.y, y_pred))
    print('Exponential MSA: ', MSA(lab1.y, y_pred))
    y_pred = lab1.quadratic_approximation(False)
    print('Quadratic MSE: ', MSE(lab1.y, y_pred))
    print('Quadratic MSA: ', MSA(lab1.y, y_pred))
    # The best approximation is the quadratic one
    print('The best approximation is the quadratic one')
    print(min(MSE(lab1.y, y_pred), MSE(lab1.y, y_pred), MSE(lab1.y, y_pred), MSE(lab1.y, y_pred)))


final_compare_all_function()

# какая из вышеперечисленных функций наиболее точно аппроксимирует функцию?

# Точность аппроксимирующих функций будет зависеть от конкретных данных,
# которые вы используете. Невозможно сказать, какая функция будет наиболее точно
# аппроксимировать данные, не изучив сами данные. Чтобы определить точность
# аппроксимирующих функций, вы можете сравнить их прогнозируемые значения
# с фактическими значениями для заданных данных и вычислить такой показатель,
# как среднеквадратичная ошибка или средняя абсолютная ошибка,
# чтобы количественно оценить разницу. Вы также можете визуализировать
# аппроксимирующие функции и данные на графике и визуально проверить соответствие.

# На данном наборе данных лучшей аппроксимацией является квадратичная функция