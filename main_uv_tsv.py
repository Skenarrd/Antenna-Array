import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
import aspose.cells as cells

from tools_extended import (
    plot_array,
    plot_surf_uv,
    get_side_lobes,
    calc_F_theta_phi,
    get_directivity_uv,
)

from arraygrid_mod import (
    rect_random_grid_2d,
    rect_regular_grid_2d,
    rect_random_symmetry_grid_2d,
    rect_random_symmetry_size_grid_2d,
    create_hex_grid
)


def save_array_to_tsv(filename: str, coord: np.ndarray, freq: float):
    """Сохраняет координаты элементов массива в TSV файл."""
    workbook = cells.Workbook()
    worksheet = workbook.worksheets[0]
    
    # Записываем заголовки
    worksheet.cells.get(0, 0).put_value("# unit: meters")
    worksheet.cells.get(1, 0).put_value(f"# design frequency: {int(freq)} Hz")
    
    # worksheet.Cells.Get(2, 0).PutValue("# Element\tX\tY\tZ\tMagnitude\tPhase\tPhi\tTheta")
    # Записываем заголовки колонок без кавычек, используя отдельные ячейки
    # во избежание появления кавычек в файле с данными
    worksheet.cells.get(2, 0).put_value("# Element")
    worksheet.cells.get(2, 1).put_value("X")
    worksheet.cells.get(2, 2).put_value("Y")
    worksheet.cells.get(2, 3).put_value("Z")
    worksheet.cells.get(2, 4).put_value("Magnitude")
    worksheet.cells.get(2, 5).put_value("Phase")
    worksheet.cells.get(2, 6).put_value("Phi")
    worksheet.cells.get(2, 7).put_value("Theta")
    worksheet.cells.get(2, 8).put_value("Gamma")
    
    # Записываем данные
    for i, (x, y) in enumerate(coord, 1):
        row = i + 2  # Начинаем с 4-й строки (после заголовков)
        worksheet.cells.get(row, 0).put_value(str(i))  # Номер элемента
        worksheet.cells.get(row, 1).put_value(f"{x:.8f}")  # X coord
        worksheet.cells.get(row, 2).put_value(f"{y:.8f}")  # Y coord
        worksheet.cells.get(row, 3).put_value("0")  # Z coord
        worksheet.cells.get(row, 4).put_value("1")  # Амплитудное распределение
        worksheet.cells.get(row, 5).put_value("0")  # Фазовове распределение
        worksheet.cells.get(row, 6).put_value("0")  # Phi
        worksheet.cells.get(row, 7).put_value("0")  # Theta
        worksheet.cells.get(row, 8).put_value("0")  # Gamma

    # Сохраняем файл
    workbook.save(filename, cells.SaveFormat.TSV)

    # Удаляем последнюю строку с сообщением об оценочной версии
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Перезаписываем файл без последней строки
    with open(filename, 'w') as file:
        file.writelines(lines[:-1])


if __name__ == "__main__":
    # np.random.seed(100)
    freq = 10e9  # при такой записи тип данных - float
    wavelength = c / freq  # при freq = 10e9: 30 см
    # coord = rect_regular_grid_2d(20, 10, 0.7 * wavelength, 0.7 * wavelength)
    # coord = rect_random_grid_2d(20, 10, 0.7 * wavelength, 0.7 * wavelength, 0.3 * wavelength, 0.3 * wavelength)
    '''
    coord = rect_random_symmetry_grid_2d(
    nx=32, 
    ny=9, 
    dx=1.5 * wavelength,  
    dy=1.2 * wavelength,
    max_shift_x=0.5 * wavelength,
    max_shift_y=0.02 * wavelength,
    min_distance=0.15 * wavelength
    )
    '''
    coord = create_hex_grid(nx=8, ny=8, dx=0.7 * wavelength, dy=0.7 * wavelength)
    # Код заработал, когда были увеличены dx, dy и уменьшено min_distance
    # coord = rect_random_symmetry_size_grid_2d(10, 10, 0.3, 0.3, 0.015, 0.015)
    # coord = np.array([0.0, 0.0], ndmin=2)
    # print(coord.shape)

    # Сохраняем координаты в TSV файл
    save_array_to_tsv("array_coordinates.tsv", coord, freq)

    theta_0 = 0.0
    phi_0 = 0.0

    u_min = -1
    u_max = 1
    v_min = -1
    v_max = 1
    step_u = 0.01
    step_v = 0.02

    u = np.arange(u_min, u_max + step_u, step_u)
    v = np.arange(v_min, v_max + step_v, step_v)

    u_mesh, v_mesh = np.meshgrid(u, v)
    uv2 = u_mesh**2 + v_mesh**2
    uv2[uv2 > 1] = np.nan

    theta_mesh = np.rad2deg(np.arcsin(np.sqrt(uv2)))
    phi_mesh = np.rad2deg(np.arctan2(v_mesh, u_mesh))

    F = calc_F_theta_phi(coord, freq, theta_0, phi_0, theta_mesh, phi_mesh)
    F_dB = 10 * np.log10(F)

    directivity = get_directivity_uv(F, u_mesh, v_mesh, step_u, step_v)
    directivity_dB = 10 * np.log10(directivity)

    print(f"КНД: {directivity_dB} дБ")

    side_lobes = get_side_lobes(F_dB)
    main_lobe = side_lobes[0][2]
    for v_val, u_val, val in side_lobes[:10]:
        print(
            f"u={u_min + u_val * step_u}, v={v_min + v_val * step_v }, val={val - main_lobe} дБ"
        )

    plot_array(coord)
    F_dB[F_dB < -50] = -50

    fig = plt.figure()
    axes_uv = fig.add_subplot(1, 1, 1, projection="3d")
    plot_surf_uv(u_mesh, v_mesh, F_dB, axes_uv)
    plt.show()
