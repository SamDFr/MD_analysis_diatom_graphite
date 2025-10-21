#!/export/apps/anaconda3/2021.05/bin/python

import os
import re
import ase
import shutil
from   tqdm        import tqdm
from   ase.io      import read, write
from   ase.io.vasp import write_vasp_xdatcar

def write_vdatcar(output_file, trajectory):
    """
    Write the VDATCAR file with velocities from the ASE trajectory.
    Velocities are converted from m/s to Å/fs.
    """
   # print(f"Writing VDATCAR data to {output_file}")
    conversion_factor = 1e-1  # Convert from m/s to Å/fs
    with open(output_file, 'w') as f:
        #f.write("Velocities from LAMMPS trajectory (in Å/fs)\n\n")
        for t, atoms in enumerate(trajectory):
            #f.write(f"Direct configuration= {t + 1}\n")
            velocities = atoms.get_velocities()
            if velocities is None:
                raise ValueError("Velocities are not present in the trajectory.")
            velocities_in_afs = velocities * conversion_factor
            for v in velocities_in_afs:
                f.write(f"  {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
    #print("Finished writing VDATCAR data.")
    
def parse_lammps_log(file_path):
    """Parses the LAMMPS log file to extract relevant data."""
    
    # Initialize storage for parsed data
    parsed_data   = []
    previous_line = None

    with open(file_path, 'r') as file:
        i = 0
        for line in file:
            # Strip leading and trailing whitespace
            line = line.strip()
            # Skip the line if it's the same as the previous line
            if line == previous_line:
                continue
            # Parse lines with the specified format
            match = re.match(r'^(\d+)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)$', line)
            
            if match:
                previous_line = line
                step, total_energy, potential_energy, kinetic_energy, temperature, pressure = match.groups()
                # Convert to output format
                formatted_line = f"{i} {float(temperature):.1f} {float(total_energy):.8E} {float(potential_energy):.8E} {float(potential_energy):.8E} {float(kinetic_energy):.5E}"
                parsed_data.append(formatted_line)
                i += 1
    return parsed_data[1:]

# Prompt the user for the NO molecule incidence
incidence = input("Enter the NO molecule incidence (Normal/Oriented): ")

# Validate the incidence input
if incidence not in ["Normal", "Oriented"]:
    print("Invalid incidence. Please enter either 'Normal' or 'Oriented'")
    exit(1)

# Prompt the user for the energy of the NO molecule
energy = input("Enter the energy of the NO molecule (25/50/100/300): ")

# Validate the energy input
if energy not in ["25", "50", "100", "300"]:
    print("Invalid energy. Please enter one of the following values: 25, 50, 100, 300")
    exit(1)

print("\nEnter the number of the initial and the final trajectories")
Trays_input = input(" from aenet dyn output to transform into VASP output (FORMAT: INITRAJ, FINTRAJ): ")
tray_range  = Trays_input.split(',')

#LAMMPS_PATH = f'{incidence}_{energy}eV/'
LAMMPS_PATH = f'{incidence}_300K_{energy}meV/'
output_PATH = f'VASP_format_{incidence}_{energy}meV/'

# Check if the directory exists, if not, create it
if not os.path.exists(output_PATH):
    os.makedirs(output_PATH)
    print(f"Directory {output_PATH} created.")

# Example usage:
for i in tqdm(range(int(tray_range[0]), int(tray_range[1]) + 1), ncols=100):
    
    # Define the file paths for input and output
    input_trajectory_PATH = os.path.join(LAMMPS_PATH, str(i))
    input_trajectory_file = f"{input_trajectory_PATH}/nve.lammpstrj"
    input_log_file        = f"{input_trajectory_PATH}/log.lammps"
    output_XDATCAR        = f"{output_PATH}XDATCAR-{i}"
    output_VDATCAR        = f"{output_PATH}VDATCAR-{i}"
    output_aimd_Resume    = f"{output_PATH}aimd_resume-{i}.dat"
    
    if os.path.exists(input_trajectory_PATH) == False:
        print()
        print("There is not dynamic " + str(i) + " in " + str(energy) + " eV")
        continue

    # Parse the input file
    if os.path.exists(input_log_file) == False or os.path.getsize(input_log_file) == 0:
        print()
        print("log.lammps file for trayectory " + str(i) + " in " + str(energy) + " eV present error")
        continue
    else:
        aimd_resume = parse_lammps_log(input_log_file)

    # Read the LAMMPS trajectory file
    if os.path.exists(input_trajectory_PATH) == False or os.path.getsize(input_trajectory_PATH) == 0:
        print()
        print("dynamics.lammpstrj file for trayectory " + str(i) + " in " + str(energy) + " eV present error")
        continue
    else:
        trajectory = read(input_trajectory_file, index=':', format="lammps-dump-text", specorder=[7, 8, 6], units="metal",)

    # Check if the dynamics is correct
    #if len(trajectory) != len(aimd_resume):
        #print()
        #print(f"Warning  !!! -> Trayectory {i} have diferents input files size -> XDATCAR, VDATCAR: {len(trajectory)} and AIMD_resume: {len(aimd_resume)}")
        # Define the folder to delete (here, the input folder for trajectory i)
        #if os.path.exists(input_trajectory_PATH):
            #print(f"Deleting folder: {input_trajectory_PATH}")
            #shutil.rmtree(input_trajectory_PATH)
        #else:
            #print(f"Folder {input_trajectory_PATH} does not exist; nothing to delete.")
        # Skip further processing for this trajectory
        #continue     

    # Write the XDATCAR file
    ase.io.vasp.write_vasp_xdatcar(output_XDATCAR, trajectory, label=None)
    
    # Write the VDATCAR file
    write_vdatcar(output_VDATCAR, trajectory)
    
    # Write to the new file in the required format
    with open(output_aimd_Resume, 'w') as file:
        for entry in aimd_resume:
            file.write(entry + '\n')

print("Done !!!")
