import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist


def rect_regular_grid_2d(nx: int, ny: int, dx: float, dy: float) -> npt.NDArray:
    """Создает прямоугольную регулярную сетку.

    nx - количество элементов по оси X
    ny - количество элементов по оси Y
    dx - шаг по оси X
    dy - шаг по оси Y
    """
    x = np.arange(nx) * dx - (nx - 1) * dx / 2
    y = np.arange(ny) * dy - (ny - 1) * dy / 2
    return np.array([(x_i, y_i) for x_i in x for y_i in y])


def create_symmetry_grid_2d(q1: npt.NDArray) -> npt.NDArray:
    q2 = np.array([(-x_i, y_i) for x_i, y_i in q1])
    q3 = np.array([(-x_i, -y_i) for x_i, y_i in q1])
    q4 = np.array([(x_i, -y_i) for x_i, y_i in q1])
    return np.concatenate((q1, q2, q3, q4))


def rect_random_grid_2d(
        nx: int, ny: int, dx: float, dy: float, max_shift_x: float, max_shift_y: float
) -> npt.NDArray:
    array = rect_regular_grid_2d(nx, ny, dx, dy)
    # Shift - смещения элементов
    shift_x = np.random.uniform(-max_shift_x, max_shift_x, array.shape[0])
    shift_y = np.random.uniform(-max_shift_y, max_shift_y, array.shape[0])
    array[:, 0] += shift_x
    array[:, 1] += shift_y
    return array


# Внедрение гексагональной сетки координат
def hex_grid_q1(nx: int, ny: int, dx: float, dy: float) -> npt.NDArray:
    """Создает гексагональную сетку в первом квадранте."""
    hex_dy = dy * np.sqrt(3) / 2  # Вертикальное расстояние между рядами
    q1 = []
    for row in range(ny):
        x_shift = (row % 2) * dx / 2  # Смещение четных/нечетных рядов
        for col in range(nx):
            x = col * dx + x_shift
            y = row * hex_dy
            q1.append((x, y))
    return np.array(q1)


def rect_random_symmetry_grid_2d(
        nx: int, ny: int, dx: float, dy: float,
        max_shift_x: float, max_shift_y: float,
        min_distance: float = 0.0
) -> npt.NDArray:
    """Генерирует симметричную случайную сетку с проверкой минимального расстояния между элементами."""

    # Инициализация флага для проверки валидности сетки
    valid = False
    while not valid:
        # 1. Генерация гексагональной сетки в первом квадранте
        # nx//2 и ny//2 - количество элементов в половине сетки по осям X и Y
        q1 = hex_grid_q1(nx // 2, ny // 2, dx, dy)

        # 2. Случайные смещения элементов внутри первого квадранта
        # Генерация смещений в диапазоне [-max_shift_x, max_shift_x] для каждой координаты
        shift_x = np.random.uniform(-max_shift_x, max_shift_x, q1.shape[0])
        shift_y = np.random.uniform(-max_shift_y, max_shift_y, q1.shape[0])
        # Применение смещений к координатам элементов
        shifted = q1.copy()
        shifted[:, 0] += shift_x  # Смещение по X
        shifted[:, 1] += shift_y  # Смещение по Y

        # 3. Создание полной симметричной сетки из первого квадранта
        # Отражаем элементы в 4 квадранта: (x,y), (-x,y), (-x,-y), (x,-y)
        full_grid = create_symmetry_grid_2d(shifted)

        # 4. Проверка минимального расстояния между элементами
        if min_distance > 0:
            # Вычисление попарных расстояний между всеми элементами
            dist_matrix = cdist(full_grid, full_grid)
            # Игнорирование расстояния элемента до самого себя (замена диагонали на бесконечность)
            np.fill_diagonal(dist_matrix, np.inf)
            # Проверка: все расстояния >= min_distance
            valid = np.all(dist_matrix >= min_distance)
        else:
            # Если min_distance не задан, сетка всегда валидна
            valid = True

    # 5. Возврат сгенерированной сетки, удовлетворяющей условиям
    return full_grid


'''
1. nx, ny - количество элементов по осям X и Y в первом квадранте. 
Поскольку используется симметрия, общее количество элементов будет в 4 раза больше (4 квадранта). 
Например, если nx=8 и ny=6, то в первом квадранте 8//2=4 по X и 6//2=3 по Y, всего 4*3=12 элементов. Полная сетка: 12*4=48 элементов

2. dx, dy - шаг между элементами в регулярной сетке до смещения. 
Если шаг слишком мал, даже без смещения элементы могут быть слишком близко. 
При добавлении случайных смещений (max_shift_x, max_shift_y) расстояние может уменьшиться ещё больше

3. max_shift_x, max_shift_y - максимальное смещение элементов от регулярной позиции. 
Если эти значения слишком велики относительно dx и dy, 
элементы могут перекрываться даже при изначально достаточном шаге

4. min_distance - минимальное допустимое расстояние между любыми двумя элементами. 
Если min_distance слишком велико по сравнению с dx, dy и смещениями, условие может не выполняться

Примеры проблемных сценариев:

а) Слишком маленький шаг (dx, dy). 
Например, dx=0.5*wavelength, а min_distance=0.6*wavelength. 
Даже без смещений расстояние между соседними элементами будет 0.5*wavelength, что меньше min_distance

б) Большие смещения при маленьком шаге. 
Например, dx=0.7*wavelength, max_shift_x=0.4*wavelength. 
Тогда элементы могут сдвигаться на 0.7 - 0.4 = 0.3*wavelength друг к другу. 
Если min_distance=0.3*wavelength, это на грани, но если шаг меньше или смещение больше, возникнет ошибка
'''

def rect_random_symmetry_size_grid_2d(nx: int, ny: int, size_x: float, size_y: float, gap_x: float = 0, gap_y: float = 0) -> npt.NDArray:
    x_min = -size_x / 2
    x_max = -gap_x
    y_min = gap_y
    y_max = size_y / 2

    q_count = (nx // 2) * (ny // 2)

    x = np.random.uniform(x_min, x_max, q_count)
    y = np.random.uniform(y_min, y_max, q_count)
    q1 = np.column_stack((x, y))
    return create_symmetry_grid_2d(q1)
