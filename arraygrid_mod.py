'''
В гексагональной сетке элементы могут оказаться слишком 
близко друг к другу после смещений, с учетом того, 
что создаются симметричные копии
'''
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
def hex_grid_q1(nx: int, ny: int, dx: float, dy: float, min_distance: float) -> npt.NDArray:
    """Создает гексагональную сетку в первом квадранте с отступом от осей."""
    hex_dy = dy * np.sqrt(3) / 2
    
    # Добавляем отступ от осей равный половине минимального расстояния
    axis_offset = min_distance / 2
    
    q1 = []
    for row in range(ny):
        x_shift = (row % 2) * dx / 2
        for col in range(nx):
            x = col * dx + x_shift + axis_offset  # Добавляем отступ по X
            y = row * hex_dy + axis_offset        # Добавляем отступ по Y
            q1.append((x, y))
    return np.array(q1)


def rect_random_symmetry_grid_2d(
        nx: int, ny: int, dx: float, dy: float,
        max_shift_x: float, max_shift_y: float,
        min_distance: float = 0.0,
        max_try_count: int = 1500
) -> npt.NDArray:
    try_count = 0
    valid = False
    while not valid and try_count < max_try_count:
        try_count += 1
        # Передаем min_distance в функцию hex_grid_q1
        q1 = hex_grid_q1(nx // 2, ny // 2, dx, dy, min_distance)
        
        # Ограничиваем смещения, чтобы элементы не приближались к осям ближе min_distance/2
        shift_x = np.random.uniform(-max_shift_x, max_shift_x, q1.shape[0])
        shift_y = np.random.uniform(-max_shift_y, max_shift_y, q1.shape[0])
        
        shifted = q1.copy()
        shifted[:, 0] += shift_x
        shifted[:, 1] += shift_y
        
        # Дополнительная проверка: элементы не должны быть ближе min_distance/2 к осям
        if np.any(shifted[:, 0] < min_distance/2) or np.any(shifted[:, 1] < min_distance/2):
            continue
            
        full_grid = create_symmetry_grid_2d(shifted)
        
        if min_distance > 0:
            dist_matrix = cdist(full_grid, full_grid)
            np.fill_diagonal(dist_matrix, np.inf)
            valid = np.all(dist_matrix >= min_distance)
        else:
            valid = True

    if not valid:
        raise ValueError(
            f"Не удалось создать сетку с заданными параметрами после {max_try_count} попыток. "
            f"Попробуйте увеличить dx, dy или уменьшить min_distance"
        )
    
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
