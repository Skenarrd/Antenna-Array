import numpy as np
from numpy import pi as pi
import matplotlib.pyplot as plt


class F_n:
    def __init__(self, A, theta, psi, N, lmbd, dx, theta_0):
        self.A = A      # амплитуды (теперь одномерный массив N)
        self.theta = theta  # углы места
        self.psi = psi    # начальные фазы (теперь одномерный массив N)
        self.N = N        # количество элементов
        self.lmbd = lmbd  # длина волны
        self.k = 2 * pi / lmbd
        self.dx = dx      # расстояние между элементами
        self.theta_0 = theta_0  # угол места главного лепестка

    def calc(self):
        F_n = np.zeros_like(self.theta, dtype=np.complex128)
        
        for n in range(self.N):
            # Фазовый сдвиг для одномерной решетки
            phase = self.k * n * self.dx * (np.sin(self.theta) - np.sin(self.theta_0))
            F = np.abs(self.A[n]) * np.exp(1j * (phase - self.psi[n]))
            F_n += F
        return F_n

class F_n_magnitude: # модуль комплексной амплитуды
    def __init__(self, F_n):
        self.F_n = F_n

    def calc(self): 
        F_n_magnitude = np.abs(self.F_n) * np.abs(self.F_n) # **2
        F_n_magnitude /= np.max(F_n_magnitude)
        return F_n_magnitude 
    
class F_1: # нормированная ДН
    def __init__(self, F_n_magnitude, theta):
        self.F_n_magnitude = F_n_magnitude
        self.theta = theta

    def calc(self):
        return self.F_n_magnitude * np.cos(self.theta)
    
class F_n_dB: # ДН в дБ
    def __init__(self, F_1):
        self.F_1 = F_1
    def calc(self):
        return 10 * np.log10(self.F_1)
    
class PlotShow: # класс, отвечающий за отображение графиков
    def __init__(self, figure_size=(10, 10)):
        self.fig_size = figure_size
        

    def setup_plot(self, title, xlabel, ylabel, xlim, ylim, color='black', linewidth=1, grid=True):
        plt.figure(figsize=self.fig_size)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid(grid)
        
    def plot_diagram(self, x, y, color='black', linewidth=1):
        plt.plot(x, y, color=color, linewidth=linewidth)
        
    def show_plot(self):
        plt.show()
        
    def save_plot(self, filename):
        plt.savefig(filename)

f = 1e6
c = 3e8
N = 10  # элементов
lmbd = c / f
dx = lmbd / 2

# Создаем одномерные массивы для амплитуд и фаз
A = np.ones(N)
psi = np.zeros(N)

# Создаем массив углов
theta = np.linspace(-90, 90, 180)
theta_rad = np.radians(theta)

# Задаем направление главного лепестка
theta_0 = np.radians(0)

# Создаем объекты и строим диаграмму
f_n = F_n(A, theta_rad, psi, N, lmbd, dx, theta_0)
F_n = f_n.calc()

f_n_magnitude = F_n_magnitude(F_n)
F_n_magnitude = f_n_magnitude.calc()

f_1 = F_1(F_n_magnitude, theta_rad)
f_n_db = F_n_dB(f_1.calc())

# Построение одномерной диаграммы направленности
plot_show = PlotShow()
plot_show.setup_plot(
    title='Диаграмма направленности линейной АР',
    xlabel='θ, градусы',
    ylabel='F(θ), дБ',
    xlim=(-90, 90),
    ylim=(-40, 0)
)
plot_show.plot_diagram(theta, f_n_db.calc(), color='black', linewidth=2)
plot_show.show_plot()

print(f'Направление главного лепестка: {theta_0 * 180 / pi:.2f} градусов')

# Сохранить график
# plot_show.save_plot('diagram.png')
