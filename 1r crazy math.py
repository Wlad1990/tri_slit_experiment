import math
import tkinter as tk


class Delta_T_GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Delta T Calculator")

        # Define constants
        self.G = 6.6743e-11  # Gravitational constant
        self.c = 299792458  # Speed of light
        self.l_pl = 1.616 * 10 ** (-35)  # Planck Length

        # Create input frame
        self.input_frame = tk.Frame(self.master)
        self.input_frame.pack(side=tk.TOP, padx=10, pady=10)

        # Define input variables
        self.delta_x_var = tk.DoubleVar()
        self.delta_y_var = tk.DoubleVar()
        self.delta_z_var = tk.DoubleVar()
        self.m1_var = tk.DoubleVar()
        self.m2_var = tk.DoubleVar()
        self.rc1_var = tk.DoubleVar()
        self.rc2_var = tk.DoubleVar()
        self.v1_var = tk.DoubleVar()
        self.v2_var = tk.DoubleVar()
        self.E_var = tk.DoubleVar()
        self.Phi_var = tk.DoubleVar()
        self.Psi_var = tk.DoubleVar()
        self.delta_t_grav_var = tk.DoubleVar()
        self.delta_t_lens_var = tk.DoubleVar()
        self.delta_t_bh_var = tk.DoubleVar()
        self.delta_t_ctc_var = tk.DoubleVar()
        self.delta_t_1r_var = tk.DoubleVar()
        self.delta_t_neu_var = tk.DoubleVar()
        self.delta_t_rad_var = tk.DoubleVar()
        self.delta_t_qf_var = tk.DoubleVar()
        self.delta_t_gw_var = tk.DoubleVar()
        self.delta_t_dm_var = tk.DoubleVar()
        self.delta_t_ns_var = tk.DoubleVar()
        self.delta_t_ps_var = tk.DoubleVar()
        self.delta_t_acc_var = tk.DoubleVar()
        self.delta_t_qm_var = tk.DoubleVar()
        self.delta_t_ent_var = tk.DoubleVar()
        self.delta_t_tun_var = tk.DoubleVar()
        self.delta_t_mea_var = tk.DoubleVar()
        self.delta_m1_var = tk.DoubleVar()
        self.delta_m2_var = tk.DoubleVar()

        # Adding input fields for the remaining parameters
        tk.Label(self.input_frame, text="Delta m1:").pack()
        delta_m1_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=delta_m1_var).pack()

        tk.Label(self.input_frame, text="Delta m2:").pack()
        delta_m2_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=delta_m2_var).pack()

        tk.Label(self.input_frame, text="v1:").pack()
        v1_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=v1_var).pack()

        tk.Label(self.input_frame, text="v2:").pack()
        v2_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=v2_var).pack()

        tk.Label(self.input_frame, text="E:").pack()
        E_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=E_var).pack()

        tk.Label(self.input_frame, text="Phi:").pack()
        Phi_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=Phi_var).pack()

        tk.Label(self.input_frame, text="Psi:").pack()
        Psi_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=Psi_var).pack()

        tk.Label(self.input_frame, text="Delta t_grav:").pack()
        delta_t_grav_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=delta_t_grav_var).pack()

        tk.Label(self.input_frame, text="Delta t_lens:").pack()
        delta_t_lens_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=delta_t_lens_var).pack()

        tk.Label(self.input_frame, text="Delta t_bh:").pack()
        delta_t_bh_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=delta_t_bh_var).pack()

        tk.Label(self.input_frame, text="Delta t_ctc:").pack()
        delta_t_ctc_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=delta_t_ctc_var).pack()

        tk.Label(self.input_frame, text="Delta t_1r:").pack()
        delta_t_1r_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=delta_t_1r_var).pack()

        tk.Label(self.input_frame, text="Delta t_neu:").pack()
        delta_t_neu_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=delta_t_neu_var).pack()

        tk.Label(self.input_frame, text="Delta t_rad:").pack()
        delta_t_rad_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=delta_t_rad_var).pack()

        tk.Label(self.input_frame, text="Delta t_qf:").pack()
        delta_t_qf_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=delta_t_qf_var).pack()

        tk.Label(self.input_frame, text="Delta t_gw:").pack()
        delta_t_gw_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=delta_t_gw_var).pack()

        tk.Label(self.input_frame, text="Delta t_dm:").pack()
        delta_t_dm_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=delta_t_dm_var).pack()

        tk.Label(self.input_frame, text="Delta t_ns:").pack()
        delta_t_ns_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=delta_t_ns_var).pack()

        tk.Label(self.input_frame, text="Delta t_ps:").pack()
        delta_t_ps_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=delta_t_ps_var).pack()

        tk.Label(self.input_frame, text="Delta t_acc:").pack()
        delta_t_acc_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=delta_t_acc_var).pack()

        tk.Label(self.input_frame, text="Delta t_qm:").pack()
        delta_t_qm_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=delta_t_qm_var).pack()

        tk.Label(self.input_frame, text="Delta t_ent:").pack()
        delta_t_ent_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=delta_t_ent_var).pack()

        tk.Label(self.input_frame, text="Delta t_tun:").pack()
        delta_t_tun_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=delta_t_tun_var).pack()

        tk.Label(self.input_frame, text="Delta t_mea:").pack()
        delta_t_mea_var = tk.DoubleVar()
        tk.Entry(self.input_frame, textvariable=delta_t_mea_var).pack()


# Define the Pseudo-Equation
def delta(delta_x, delta_y, delta_z, m1, m2, rc1, rc2, v1, v2, E, Phi, Psi, delta_t_grav, delta_t_lens, delta_t_bh,
          delta_t_ctc, delta_t_1r, delta_t_neu, delta_t_rad, delta_t_qf, delta_t_gw, delta_t_dm, delta_t_ns,
          delta_t_ps, delta_t_acc, delta_t_qm, delta_t_ent, delta_t_tun, delta_t_mea):
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


# Defining the function to calculate and display the result
def calculate():
    # Retrieving input values
    delta_x = gui.delta_x_var.get()
    delta_y = gui.delta_y_var.get()
    delta_z = gui.delta_z_var.get()
    m1 = gui.m1_var.get()
    m2 = gui.m2_var.get()
    rc1 = gui.rc1_var.get()
    rc2 = gui.rc2_var.get()
    v1 = gui.v1_var.get()
    v2 = gui.v2_var.get()
    E = gui.E_var.get()
    Phi = gui.Phi_var.get()
    Psi = gui.Psi_var.get()
    delta_t_grav = gui.delta_t_grav_var.get()
    delta_t_lens = gui.delta_t_lens_var.get()
    delta_t_bh = gui.delta_t_bh_var.get()
    delta_t_ctc = gui.delta_t_ctc_var.get()
    delta_t_1r = gui.delta_t_1r_var.get()
    delta_t_neu = gui.delta_t_neu_var.get()
    delta_t_rad = gui.delta_t_rad_var.get()
    delta_t_qf = gui.delta_t_qf_var.get()
    delta_t_gw = gui.delta_t_gw_var.get()
    delta_t_dm = gui.delta_t_dm_var.get()
    delta_t_ns = gui.delta_t_ns_var.get()
    delta_t_ps = gui.delta_t_ps_var.get()
    delta_t_acc = gui.delta_t_acc_var.get()
    delta_t_qm = gui.delta_t_qm_var.get()
    delta_t_ent = gui.delta_t_ent_var.get()
    delta_t_tun = gui.delta_t_tun_var.get()
    delta_t_mea = gui.delta_t_mea_var.get()

    # Calculating result using the delta_t function
    result = delta(delta_x, delta_y, delta_z, m1, m2, rc1, rc2, v1, v2, E, Phi, Psi,
                   delta_t_grav, delta_t_lens, delta_t_bh, delta_t_ctc, delta_t_1r, delta_t_neu,
                   delta_t_rad, delta_t_qf, delta_t_gw, delta_t_dm, delta_t_ns, delta_t_ps,
                   delta_t_acc, delta_t_qm, delta_t_ent, delta_t_tun, delta_t_mea)

    # Updating the result label
    result_label.config(text=f"Result: {result}")


gui = Delta_T_GUI(tk.Tk())
tk.mainloop()