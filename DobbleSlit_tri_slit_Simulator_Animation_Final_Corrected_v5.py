import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tkinter import *
import numpy as np
from matplotlib.pyplot import figure, plot, show, subplots
from numpy import array
from scipy.linalg import eigh_tridiagonal
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages
from tkinter import *
from tkinter import filedialog, messagebox
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import psutil
import random
from tkinter import Tk, Checkbutton, IntVar

from sfs.plot2d import particles

# Constants (for a single electron in a 1D box of length L=1m)

hbar = 1.0545718e-34
m = 9.10938356e-31
L = 1.0
N = 1000
dx = L / N


def V(x):
    return np.zeros_like(x)


# Generate grid
x = np.linspace(0, L, N)
potential = V(x)

# Construct Hamiltonian matrix using finite difference method
diagonal = hbar ** 2 / (m * dx ** 2) + potential
off_diagonal = -hbar ** 2 / (2 * m * dx ** 2) * np.ones(N - 1)
H = np.diag(diagonal) + np.diag(off_diagonal, k=1) + np.diag(off_diagonal, k=-1)

# Solve for eigenvalues (energies) and eigenvectors (wave-functions)
E, psi = np.linalg.eigh(H)

# Plot the first few eigenfunctions
for i in range(5):
    plot(x, psi[:, i])
show()


def particle_deflection(self, x, y, z, t, delta_x, delta_y, delta_z, delta_t_ctc, delta_t_1r, delta_t_neu, delta_t_rad,
                        delta_t_qf, delta_t_gw, delta_t_dm, delta_t_ns,
                        delta_v, delta_theta, delta_phi, delta_omega, l_pl, m_pl, g, G_N, c, h_bar, k_B, epsilon_0,
                        mu_0, alpha, e, m_e, m_p, n, l, Delta_E, Delta_p):
    l_pl = 1.616 * 10 ** (-35)  # Planck Length
    # Distance from galaxy 1 to critical radius
    delta_r1c = rc1 - (G * m1 / c ** 2)
    # Distance from galaxy 2 to critical radius
    delta_r2c = rc2 - (G * m2 / c ** 2)
    term1 = math.sqrt(1 + (l_pl * delta_x) ** 2) + math.sqrt(1 +
                                                             (l_pl * delta_y) ** 2) + math.sqrt(
        1 + (l_pl * delta_z) ** 2)
    term2 = math.sqrt(1 + ((G * m1 * delta_r1c) /
                           (delta_r1c ** 2 - (v1 ** 2 / c ** 2))) ** 2)
    term3 = math.sqrt(1 + ((G * m2 * delta_r2c) /
                           (delta_r2c ** 2 - (v2 ** 2 / c ** 2))) ** 2)
    term4 = math.sqrt(1 + (E / (m1 + m2) * c ** 2) ** 2)
    term5 = math.sqrt(1 + Phi) + Psi
    term6 = delta_t_grav + delta_t_lens + delta_t_bh + delta_t_ctc + delta_t_1r + delta_t_neu + delta_t_rad + \
            delta_t_qf + \
            delta_t_gw + delta_t_dm + delta_t_ns + delta_t_ps + delta_t_acc + \
            delta_t_qm + delta_t_ent + delta_t_tun + delta_t_mea
    return term1 + term2 + term3 + term4 + term5 + term6
    wavefront = self.wavefront(pos, source_pos, wavelength)
    if not one_r_concept:
        return wavefront * particles
    else:
        interference_pattern = np.copy(wavefront)

    # Loop through each particle and simulate its interaction with the wavefront
    for i, particle in enumerate(particles):
        deflection = particle * wavefront[i] * np.random.uniform(0.9, 1.1)

        # Apply the deflection to the interference pattern
        interference_pattern[i] += deflection

    # Normalize the interference pattern to keep the values between 0 and 1
    interference_pattern = interference_pattern / np.max(interference_pattern)

    return interference_pattern


def get_input(wavelength_entry, distance_entry, slit_width_entry, screen_width_entry, num_points_entry,
              slit1_pos_entry, slit2_pos_entry, slit3_pos_entry):
    wavelength = float(wavelength_entry.get())
    distance = float(distance_entry.get())
    slit_width = float(slit_width_entry.get())
    screen_width = float(screen_width_entry.get())
    num_points = int(num_points_entry.get())
    slit1_pos = float(slit1_pos_entry.get())
    slit2_pos = float(slit2_pos_entry.get())
    slit3_pos = float(slit3_pos_entry.get())
    return wavelength, distance, slit_width, screen_width, num_points, slit1_pos, slit2_pos, slit3_pos


class App:
    def __init__(self, master):
        self.master = master
        self.plot_fig = None
        master.title("Interference Pattern Generator")
        self.file_type_var = StringVar()
        self.file_type_var.set('txt')
        self.one_r_var = IntVar()
        self.create_widgets()

    def create_widgets(self):
        # Label and input for wavelength
        self.wavelength_label = Label(self.master, text="Enter wavelength of incident wave (in meters):")
        self.wavelength_label.grid(row=0, column=0)
        self.wavelength_entry = Entry(self.master)
        self.wavelength_entry.grid(row=0, column=1)

        # Label and input for distance
        self.distance_label = Label(self.master, text="Enter distance between slits and detector (in meters):")
        self.distance_label.grid(row=1, column=0)
        self.distance_entry = Entry(self.master)
        self.distance_entry.grid(row=1, column=1)

        # Label and input for slit width
        self.slit_width_label = Label(self.master, text="Enter width of each slit (in meters):")
        self.slit_width_label.grid(row=2, column=0)
        self.slit_width_entry = Entry(self.master)
        self.slit_width_entry.grid(row=2, column=1)

        # Label and input for screen width
        self.screen_width_label = Label(self.master, text="Enter width of detector screen (in meters):")
        self.screen_width_label.grid(row=3, column=0)
        self.screen_width_entry = Entry(self.master)
        self.screen_width_entry.grid(row=3, column=1)

        # Label and input for number of points
        self.num_points_label = Label(self.master, text="Enter number of points to sample on the screen:")
        self.num_points_label.grid(row=4, column=0)
        self.num_points_entry = Entry(self.master)
        self.num_points_entry.grid(row=4, column=1)

        # Label and input for slit 1 position
        self.slit1_pos_label = Label(self.master, text="Enter position of first slit (in meters):")
        self.slit1_pos_label.grid(row=5, column=0)
        self.slit1_pos_entry = Entry(self.master)
        self.slit1_pos_entry.grid(row=5, column=1)

        # Label and input for slit 2 position
        self.slit2_pos_label = Label(self.master, text="Enter position of second slit (in meters):")
        self.slit2_pos_label.grid(row=6, column=0)
        self.slit2_pos_entry = Entry(self.master)
        self.slit2_pos_entry.grid(row=6, column=1)

        # Label and input for slit 3 position
        self.slit3_pos_label = Label(self.master, text="Enter position of third slit (in meters):")
        self.slit3_pos_label.grid(row=7, column=0)
        self.slit3_pos_entry = Entry(self.master)
        self.slit3_pos_entry.grid(row=7, column=1)

        # Radio buttons for file type
        self.file_type_label = Label(self.master, text="Select file type:")
        self.file_type_label.grid(row=10, column=0)
        self.txt_button = Radiobutton(self.master, text=".txt", variable=self.file_type_var, value='txt')
        self.txt_button.grid(row=10, column=1)
        self.png_button = Radiobutton(self.master, text=".png", variable=self.file_type_var, value='png')
        self.png_button.grid(row=10, column=2)
        self.pdf_button = Radiobutton(self.master, text=".pdf", variable=self.file_type_var, value='pdf')
        self.pdf_button.grid(row=10, column=3)

        # Label and input for source-slit distance
        self.source_slit_distance_label = Label(self.master,
                                                text="Enter distance between source and slits (in meters):")
        self.source_slit_distance_label.grid(row=8, column=0)
        self.source_slit_distance_entry = Entry(self.master)
        self.source_slit_distance_entry.grid(row=8, column=1)

        # Create entry field for field_strength
        self.field_strength_label = Label(self.master, text="Field Strength")
        self.field_strength_label.grid(row=9, column=0)  # Replace x, y with appropriate row and column
        self.field_strength_entry = Entry(self.master)
        self.field_strength_entry.grid(row=9, column=1)  # Replace x, y with appropriate row and column

        # Start button
        self.start_button = Button(self.master, text="Start", command=self.generate_pattern)
        self.start_button.grid(row=11, column=0)

        # 3D button
        self.show_3d_button = Button(self.master, text="Show 3D", command=self.show_3d_pattern)
        self.show_3d_button.grid(row=11, column=1)

        # Save button
        self.save_button = Button(self.master, text="Save Interference Pattern", command=self.save_result)
        self.save_button.grid(row=11, column=2)

        # Create the checkbox
        self.one_r_checkbox = Checkbutton(self.master, text='Apply 1R concept', variable=self.one_r_var)
        self.one_r_checkbox.grid(row=9, column=3)

    def generate_pattern(self):
        # Get the current CPU usage as a percentage
        cpu_usage = psutil.cpu_percent()

        # Adjust the intensity of the light source based on CPU usage
        base_intensity = 1.0
        intensity = base_intensity * (1 + cpu_usage / 100)

        # Introduce a random element by slightly varying the positions of the slits
        slit1_pos = self.safe_float(self.slit1_pos_entry.get()) + random.uniform(-0.1, 0.1)
        slit2_pos = self.safe_float(self.slit2_pos_entry.get()) + random.uniform(-0.1, 0.1)
        slit3_pos = self.safe_float(self.slit3_pos_entry.get()) + random.uniform(-0.1, 0.1)

        wavelength = self.safe_float(self.wavelength_entry.get())
        distance = self.safe_float(self.distance_entry.get())
        slit_width = self.safe_float(self.slit_width_entry.get())
        screen_width = self.safe_float(self.screen_width_entry.get())
        num_points = self.safe_int(self.num_points_entry.get())
        slit1_pos = self.safe_float(self.slit1_pos_entry.get())
        slit2_pos = self.safe_float(self.slit2_pos_entry.get())
        slit3_pos = self.safe_float(self.slit3_pos_entry.get())

        source_slit_distance = self.safe_float(self.source_slit_distance_entry.get())

        x = np.linspace(-screen_width / 2, screen_width / 2, num_points)
        y = np.vectorize(self.intensity)(x, wavelength, distance, slit_width, slit1_pos, slit2_pos, slit3_pos)

        fig, ax = subplots()
        ax.plot(x, y)
        ax.set_xlabel('Position on screen (m)')
        ax.set_ylabel('Intensity')
        ax.set_title('Theoretical Interference Pattern with Three Slits')

        canvas = FigureCanvasTkAgg(fig, self.master)
        canvas.get_tk_widget().grid(row=9, columnspan=2)

        global plot_fig
        self.plot_fig = fig

    def safe_float(self, val):
        try:
            return float(val)
        except ValueError:
            return 0.0  # or any default value

    def safe_int(self, val):
        try:
            return int(val)
        except ValueError:
            return 0  # or any default value

    def save_result(self):
        file_type = self.file_type_var.get()  # Use self.file_type_var
        file_path = filedialog.asksaveasfilename(defaultextension=file_type,
                                                 filetypes=[(f"{file_type.upper()} Files", f"*.{file_type}")])
        if not file_path:  # Handle case where user cancels file dialog
            return
        save_func = {
            'txt': self.save_txt,
            'png': self.save_png,
            'pdf': self.save_pdf
        }
        if file_type not in save_func:  # Handle case where file type is not supported
            messagebox.showerror("Error", f"File type {file_type} is not supported")
            return
        save_func[file_type](file_path, plot_fig)
        if self.plot_fig is None:
            messagebox.showinfo("No plot", "Please generate a plot first.")
            return
        save_func[file_type](file_path, self.plot_fig)

    def intensity(self, x, wavelength, distance, slit_width, slit1_pos, slit2_pos, slit3_pos):
        # Calculate the distance from each slit
        d1 = np.sqrt((x - slit1_pos) ** 2 + distance ** 2)
        d2 = np.sqrt((x - slit2_pos) ** 2 + distance ** 2)
        d3 = np.sqrt((x - slit3_pos) ** 2 + distance ** 2)

        # Calculate the phase difference for each slit
        phase1 = 2 * np.pi * d1 / wavelength
        phase2 = 2 * np.pi * d2 / wavelength
        phase3 = 2 * np.pi * d3 / wavelength

        # Calculate the diffraction pattern for each slit
        diffraction1 = np.sinc(slit_width * np.sin(np.arctan((x - slit1_pos) / distance)))
        diffraction2 = np.sinc(slit_width * np.sin(np.arctan((x - slit2_pos) / distance)))
        diffraction3 = np.sinc(slit_width * np.sin(np.arctan((x - slit3_pos) / distance)))

        # Calculate the total intensity, considering both interference and diffraction
        total_intensity = (diffraction1 * np.cos(phase1) + diffraction2 * np.cos(phase2) + diffraction3 * np.cos(
            phase3)) ** 2 \
                          + (diffraction1 * np.sin(phase1) + diffraction2 * np.sin(phase2) + diffraction3 * np.sin(
            phase3)) ** 2

        return total_intensity

    def propagate_particle(self, position, wavelength, distance, slit_width, slit1_pos, slit2_pos, slit3_pos):
        field_strength = self.field_strength
        one_r_concept = self.one_r_concept

        # Create a wavefront for the particle
        wavefront = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        wavefront[tuple(position.astype(int))] = 1

        # Initialize the direction
        direction = np.array([1, 0, 0])  # Adjust this according to your requirements

        # Propagate the particle until it hits the screen (at x = distance)
        while position[0] < distance:
            # Determine the source position based on the current position and the slit positions
            source_pos = np.array([position[0], position[1], position[2]])  # Adjust this to match your requirements

            # Calculate the deflection due to the 1R field
            deflection = self.particle_deflection(position, source_pos, field_strength, one_r_concept, wavelength,
                                                  wavefront)

            # Update the direction and position
            direction += deflection
            direction /= np.linalg.norm(direction)  # Normalize to keep the speed constant
            position += direction

            # Check if the particle is out of bounds
            if np.any(position < 0) or np.any(position >= self.grid_size):
                return

            # Propagate the wavefront
            wavefront = self.propagate_wavefront(wavefront, direction, wavelength)

            # Modify the phase if one_r_concept is active
            if one_r_concept:
                phase_shift = self.calculate_one_r_phase_shift(position, wavelength, distance)
                wavefront *= np.exp(1j * phase_shift)

                # Determine which slit the particle came from
                y_pos = position[1]
                if slit1_pos <= y_pos < slit1_pos + slit_width:
                    pass
                elif slit2_pos <= y_pos < slit2_pos + slit_width:
                    pass
                elif slit3_pos <= y_pos < slit3_pos + slit_width:
                    pass
                else:
                    return  # The particle didn't come from any of the slits

                # Update the grid with the new particle
                if wavefront.shape == self.grid.shape:  # check if the shapes match
                    self.grid += wavefront
                else:
                    raise ValueError("Shapes of wavefront and grid do not match")

    def calculate_one_r_phase_shift(self, position, wavelength, distance):
        # Planck length in meters
        l_pl = 1.616229e-35

        # Calculate the position of the particle within the current Planck cell
        x_in_cell = position[0] % l_pl
        y_in_cell = position[1] % l_pl
        z_in_cell = position[2] % l_pl

        # Calculate the phase shift based on the position within the cell and the wavelength of the particle
        phase_shift = 2 * np.pi * ((x_in_cell + y_in_cell + z_in_cell) / l_pl) * (distance / wavelength)

        return phase_shift

    def propagate_wavefront(self, wavefront, direction, wavelength):
        new_wavefront = np.zeros_like(wavefront)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    shift = np.array([dx, dy, dz])
                    new_wavefront += np.roll(wavefront, shift, axis=(0, 1, 2))
        new_wavefront /= 27  # Normalize the wavefront
        new_wavefront *= np.exp(
            -1j * 2 * np.pi * np.linalg.norm(direction) / wavelength)  # Apply the phase shift due to propagation
        return new_wavefront

    def save_txt(self, file_path, plot_fig):
        np.savetxt(file_path, plot_fig.axes[0].lines[0].get_xydata())

    def save_png(self, file_path, plot_fig):
        plot_fig.savefig(file_path)

    def save_pdf(self, file_path, plot_fig):
        with PdfPages(file_path) as pdf:
            pdf.savefig(plot_fig)
            plot_fig_data = plot_fig.axes[0].lines[0].get_xydata()
            plot_fig_arr = np.column_stack((plot_fig_data[:, 0], plot_fig_data[:, 1]))
            plot_fig_image = Image.fromarray(plot_fig_arr)
            plot_fig_image.save(f"{file_path[:-4]}.png")
            pdf.attach_note("Data saved as text file")

    def show_3d_pattern(self):
        new_window = Toplevel(self.master)
        new_window.title("3D Interference Pattern")

        # Obtain the necessary data and values
        wavelength = self.safe_float(self.wavelength_entry.get())
        distance = self.safe_float(self.distance_entry.get())
        slit_width = self.safe_float(self.slit_width_entry.get())
        screen_width = self.safe_float(self.screen_width_entry.get())
        num_points = self.safe_int(self.num_points_entry.get())
        slit1_pos = self.safe_float(self.slit1_pos_entry.get())
        slit2_pos = self.safe_float(self.slit2_pos_entry.get())
        slit3_pos = self.safe_float(self.slit3_pos_entry.get())

        # Get the field strength from the entry field
        field_strength = self.safe_float(self.field_strength_entry.get())

        # Check system memory and adjust num_points accordingly
        available_memory = psutil.virtual_memory().available  # in bytes
        # Calculate maximum affordable num_points (leave some memory for other processes, say 50%)
        max_num_points = int(np.sqrt(available_memory * 0.5 / (8 * 3)))
        num_points = min(num_points, max_num_points)

        x = np.linspace(-screen_width / 2, screen_width / 2, num_points)
        source_slit_distances = np.linspace(0, 10, num_points)  # Change this range as needed
        X, Y = np.meshgrid(x, source_slit_distances)

        intensity_func = np.vectorize(self.intensity)
        Z = intensity_func(X, wavelength, distance, slit_width, slit1_pos, slit2_pos, slit3_pos)

        # Set up the 3D plot
        fig = figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')

        # Create a 3D grid to represent the environment
        self.grid_size = num_points
        self.grid = np.zeros((self.grid_size, self.grid_size, self.grid_size))

        one_r_concept = bool(self.one_r_var.get())

        # Emit particles from the slits and propagate them through the 3D environment
        for y in range(self.grid_size):
            for z in range(self.grid_size):
                self.propagate_particle(np.array([0, y, z]), wavelength, distance, slit_width,
                                        slit1_pos, slit2_pos, slit3_pos)

        # Update the 3D plot
        ax.clear()
        ax.scatter(*np.nonzero(self.grid), c=self.grid[self.grid > 0], alpha=0.5)

        canvas = FigureCanvasTkAgg(fig, new_window)

        canvas.get_tk_widget().grid(row=0, column=0)

        show()


def move_third_slit(distance, speed, duration, simulator_instance):
    """
    Moves the third slit back and forth for a specified duration at a specified speed.

    Parameters:
    - distance: The distance to move the slit in each direction from its starting position.
    - speed: The speed at which to move the slit.
    - duration: The total duration for which the slit should move.
    - simulator_instance: An instance of the simulator class.
    """

    import time

    # Calculate the time required to move the slit to one end
    time_to_move_one_end = distance / speed

    # Calculate the number of full back-and-forth cycles within the duration
    cycles = int(duration / (2 * time_to_move_one_end))


def main():
    root = Tk()
    app = App(root)
    root.mainloop()


if __name__ == '__main__':
    main()


