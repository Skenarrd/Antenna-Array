import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

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


if __name__ == "__main__":
    # np.random.seed(100)
    freq = 10e9
    wavelength = c / freq # при freq = 10e9: 30 см
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
    coord = create_hex_grid(nx=8, ny=8, dx=0.7, dy=0.7)
    # coord = rect_random_symmetry_size_grid_2d(10, 10, 0.3, 0.3, 0.015, 0.015)
    # coord = np.array([0.0, 0.0], ndmin=2)
    # print(coord.shape)

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
