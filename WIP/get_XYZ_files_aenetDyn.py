#!/export/apps/anaconda3/2021.05/bin/python

import os
import math
import time
import numpy        as     np
from   tqdm         import tqdm
from   scipy.signal import savgol_filter

##########################################    
###       Here are the functions       ###
##########################################

def get_NO_massCenter(N_pos, O_pos):
    # Get the NO's Mass Center 
    massCenter_coord = []
    for i in range(3):
        massCenter_coord.append((N_mass * N_pos[i] + O_mass * O_pos[i])/(N_mass + O_mass))
    return massCenter_coord

def get_NO2_massCenter(N_pos, O1_pos, O2_pos):
    massCenter_coord = []
    for i in range(3):
        massCenter_coord.append((N_mass * N_pos[i] + O_mass * (O1_pos[i] + O2_pos[i]))/(N_mass + 2 * O_mass))
    return massCenter_coord

def get_distance(P1, P2):
    return np.sqrt((P1[0] - P2[0])**2 + (P1[1] - P2[1])**2 + (P1[2] - P2[2])**2)

def parse_XDATCAR(file_name):
    xdatcar_file = open(file_name, 'r')
    lines        = xdatcar_file.readlines()
    xdatcar_file.close()
  
    # Extract the data from the POSCAR file
    scale_factor    = float(lines[1])
    lattice_vectors = []
    for i in range(2, 5):
        lattice_vectors.append([float(x) for x in lines[i].split()])    
    lattice_vectors = np.array(lattice_vectors)
  
    element_symbols = lines[5].split()
    element_counts  = np.array([int(x) for x in lines[6].split()])
    n_atoms         = np.sum(element_counts)
    cfg             = int(np.ceil((len(lines) - 8)/(n_atoms + 1)))
    time_step       = 1
    
    # Getting the coordinates
    coordinates = []
    for i in range(int(cfg)):
        for j in range(7 + i * (n_atoms + 1), 8 + (i + 1) * n_atoms + i):
            if j != 7 + i * n_atoms + i:
                coordinates.append([float(x) for x in lines[j].split()[:3]])                   
    coordinates = np.array(coordinates)
    
    # Create a dictionary with the extracted data
    xdatcar_data = {
        'scale_factor'   : scale_factor,
        'lattice_vectors': lattice_vectors,
        'time'           : cfg * time_step,
        'atoms_counts'   : n_atoms,
        'coordinates'    : coordinates
    }
    return xdatcar_data

def get_config(data, time, n_atoms):
    # Get the coordinates or velocities of the atoms at the given time 
    return data[(int(time) - 1) * n_atoms: int(time) * n_atoms]

# Function to determine the time at which NO2 starts to form
def get_reactTimes(filtered_y, t_end):
    # Criteria for stopping the search of the Transition State
    E_barrier = 0.75 # eV
    time_IS   = 0
    time_FS   = 0
    # Here start the loop for every point
    for i in range(t_end):
        # Initialize the pivot
        if i == 0:
            piv, t = filtered_y[i], i
    
        # Start the conditions
        if piv <= filtered_y[i]:
            piv, t = filtered_y[i], i
            #print(piv, t)
        else:
            gap = piv - filtered_y[i]
            # To detect if there is a minimun
            if gap < piv - filtered_y[i - 1]:
                piv, t = filtered_y[i], i
            
            # Big decay reach
            if gap > E_barrier:
                time_IS = t
                for j in range(i, len(filtered_y)):
                    gap = piv - filtered_y[j]
                    if gap < piv - filtered_y[j - 1]:
                        time_FS = j
                        break
                break
                
    return time_IS, time_FS

def get_L3_z(coordinates):
    n = 0
    z = 0
    for atom in coordinates[3:]:
        if atom[2] > 5:
            z += atom[2]
            n += 1
    return z / n

def write_xyz(filename, comment, coordinates, velocities, n_atoms):
    """
    Write an XYZ file with given atomic symbols and Cartesian coordinates.
    Parameters:
        filename (str)    : the name of the output file
        comment (list)    : a list of properties of the configuration
        coordinates (list): a list of lists with the x, y, z coordinates for each atom
        velocities (list) : a list of lists with the x, y, z velocities for each atom
    """
    with open(filename, 'w') as f:
        for i, row in enumerate(comment):
            f.write(str(n_atoms) + "\n")
            f.write("Trayectory, t (fs), T (K), Et (eV), Ep (eV), Ep_f (eV), Ek (eV), Bounces ")
            for elem in row:
                f.write(str(elem) + ' ')
            f.write('\n')
            for j in range(XDATCAR_data['atoms_counts']):
                if j == 0:
                    elem = 'N '
                elif 0 < j and j < 3:
                    elem = 'O '
                else:
                    elem = 'C ' 
                f.write(elem + '{:20.15f}'.format(coordinates[i][j][0]) +' '
                             + '{:20.15f}'.format(coordinates[i][j][1]) +' '
                             + '{:20.15f}'.format(coordinates[i][j][2]) +' '
                             + '{:20.15f}'.format(velocities[i][j][0])  +' '
                             + '{:20.15f}'.format(velocities[i][j][1])  +' '
                             + '{:20.15f}'.format(velocities[i][j][2])  +'\n')

def get_noPBC(coord, time_turn, time_end, n_atoms, lattice): 
    # Fix coordinates if the molecule go out of the cell
    for i in range(time_turn, time_end):
        for j in range(3):
             while(get_distance(coord[i*n_atoms + j], coord[(i-1)*n_atoms + j]) > 5):
                # Loop for X and Y
                for k in range(2):
                    if float(coord[i*n_atoms + j][k] - coord[(i-1)*n_atoms + j][k]) < -lattice[k][k] * 0.75:
                        coord[i*n_atoms + j] += lattice[k]
                    elif float(coord[i*n_atoms + j][k] - coord[(i-1)*n_atoms + j][k]) > lattice[k][k] * 0.75: 
                        coord[i*n_atoms + j] -= lattice[k]
    return coord

def get_noPBC_initial(coord, vel, lattice):
    # Fix coordinates if NO are splited at the begining for the PBC
    if get_distance(coord[0], coord[1]) > 5:
        mol_mCenter_vel = get_NO_massCenter(vel[0], vel[1])
        for k in range(2):
            if float(coord[0][k] - coord[1][k]) < -lattice[k][k] * 0.75:
                if mol_mCenter_vel[k] < 0: 
                    coord[0] += lattice[k]
                else:
                    coord[1] -= lattice[k]
            elif float(coord[0][k] - coord[1][k]) > lattice[k][k] * 0.75:
                if mol_mCenter_vel[k] > 0:
                    coord[0] -= lattice[k]
                else:
                    coord[1] += lattice[k]   
    return coord

def get_noPBC_end(coord, mol_mCenter_vel, lattice):
    # Fix coordinates if NO are splited at the end for the PBC
    for k in range(2):
        if float(coord[0][k] - coord[1][k]) < -lattice[k][k] * 0.75:
            if mol_mCenter_vel[k] < 0: 
                coord[0] += lattice[k]
            else:
                coord[1] -= lattice[k]
        elif float(coord[0][k] - coord[1][k]) > lattice[k][k] * 0.75:
            if mol_mCenter_vel[k] > 0:
                coord[0] -= lattice[k]
            else:
                coord[1] += lattice[k]   
    return coord

def get_minValue_range(data, indices, local_min_distance):
    # Take only the min, means only bounces and not returning points
    i = 0
    while i < len(indices):
        current_index = indices[i]
        next_index    = indices[i + 1] if i + 1 < len(indices) else None

        # Take the min height around the minimun if there is more than 1
        if next_index and abs(next_index - current_index) < local_min_distance:
            if data[next_index] < data[current_index]:
                indices = np.delete(indices, i)
            else:
                indices = np.delete(indices, i + 1)
            # No increment for 'i' as the length of zero_indices has changed
        else:
            i += 1
    return indices

def get_bounces(h_data, v_data, height):
    
    # Analize the velocity change first
    t_bounce = []
    for i, velocity in enumerate(v_data[1:]):
        if v_data[i-1] < 0 and velocity > 0:
            t_bounce.append(i + 1)
    
    # Take only the minimun values in an range and the bouncing
    t_bounce = get_minValue_range(v_data, t_bounce, 150)
    t_bounce = np.array([elem for elem in t_bounce if h_data[elem] <= height])

    return t_bounce

def get_Params(coord, velocity):
    '''
    Get the most important distances and angles
    '''
    
    # Get distances between O2 and the bounded carbons atoms C1, C2
    q_O2C1_norm           = np.linalg.norm(coord[2] - coord[97])     # (distance)
    q_O2C2_norm           = np.linalg.norm(coord[2] - coord[79])     # (distance)
    
    #Set positions and velocities easier to use
    q_N                   = coord[0]
    q_O1                  = coord[1]
    q_O2                  = coord[2]
    v_N                   = velocity[0]
    v_O1                  = velocity[1]
    v_O2                  = velocity[2]
    
    # Get intermolecular vector N-O1
    q_NO1                 = q_O1 - q_N                    # vector
    q_NO1_norm            = np.linalg.norm(q_NO1)         # au (distance)
    
    # Get vector and direction between N and O2
    q_NO2                 = q_O2 - q_N                    # vector
    q_NO2_norm            = np.linalg.norm(q_NO2)         # (distance)
    
    # Reactive trajectory or not
    NO2_form              = True if ((q_O2C1_norm > 2 and q_O2C2_norm > 2) or q_NO2_norm < 2) else False
    
    # Center of mass params
    mol_massCenter        = get_NO2_massCenter(q_N, q_O1, q_O2) if NO2_form else get_NO_massCenter(q_N, q_O1)  # coordinates
    mol_mCenter_vel       = get_NO2_massCenter(v_N, v_O1, v_O2) if NO2_form else get_NO_massCenter(v_N, v_O1)  # vector
        
    # Molecule High respect HOPG surface
    d_mol_L3              = mol_massCenter[2] - get_L3_z(coord) # au (distance)
       
    # Create a dictionary with the calculated data
    data = {
        'mol_massCenter'  : mol_massCenter  ,
        'mol_mCenter_vel' : mol_mCenter_vel ,
        'd_NO1'           : q_NO1_norm      ,
        'd_NO2'           : q_NO2_norm      ,
        'd_mol_L3'        : d_mol_L3        ,
        'NO2_form'        : NO2_form        ,
    }
    return data

def get_times(coord, vel, Ep_f, n_atoms, run_time):
    
    mol_height   = []
    mol_vel_z    = []
    time_end     = run_time - 1
        
    for i in range(run_time):
        
        params = get_Params(coord[i*n_atoms:(i+1)*n_atoms], vel[i*n_atoms:(i+1)*n_atoms] if i != (run_time - 1) else vel[(i-1)*n_atoms:(i)*n_atoms])
        
        mol_height.append(params['d_mol_L3'])
        mol_vel_z.append(params['mol_mCenter_vel'][2])
        
        # Stablish the conditions
        mol_go_out = True if (params['d_mol_L3'] > detector_high and params['mol_mCenter_vel'][2] > 0) else False
        mol_go_pbc = True if (params['d_NO1'] > 5                and params['mol_mCenter_vel'][2] > 0) else False
        
        # Finish the analisys at the scattered point or at the end
        if detector_high != 0:
            # Determine the end if the Periodic Boundary Conditions are not allowed
            if slab_pbc == False and mol_go_pbc:
                time_end = i
                break
            
            # Determine the end if the Periodic Boundary Conditions are allowed
            if mol_go_out:
                time_end = i + 1
                break
    
    height           = 5.2
    max_height       = 10.0
    time_bounce      = get_bounces(mol_height, mol_vel_z, height)
    while len(time_bounce) == 0:
        print(f' No Turning point founded closer than {height} Å, increasing minimun height allowed')
        height      += 0.1  
        
        if height > max_height:
            print(f'Exceeded maximum height limit {max_height} Å. No turning point found within the allowed range.')
            time_bounce = np.zeros(2)
            break
        time_bounce  = get_bounces(mol_height, mol_vel_z, height) 
    time_turn        = int(time_bounce[0])
    time_IS, time_FS = get_reactTimes(Ep_f, time_end)
    time_TS          = int((time_IS + time_FS) / 2)
    
    # Create a dictionary with the calculated data
    data = {
        'Turn_Point': time_turn   ,
        'End'       : time_end    ,
        'IS'        : time_IS     ,
        'TS'        : time_TS     ,
        'FS'        : time_FS     ,
        'Bounce'    : time_bounce ,
    }
    return data

###################################################    
### Some constants and variables declaration    ###
###################################################

# Prompt the user for the NO molecule incidence
incidence = input("Enter the NO molecule incidence (Normal/Oriented): ")

# Validate the incidence input
if incidence not in ["Normal", "Oriented"]:
    print("Invalid incidence. Please enter either 'normal' or 'oriented'.")
    exit(1)

# Some variables and arrays needed to the code 
print()
Energies_input = input("Enter the energy value in eV separated by commas: ")
E              = [e.strip() for e in Energies_input.split(',')]

print()
print("Enter the number of the initial and the final trajectories")
Trays_input    = input("you want to get XYZ files (FORMAT: INITRAJ, FINTRAJ): ")
tray_range     = Trays_input.split(',')
t_start        = int(tray_range[0])
t_end          = int(tray_range[1])

#print()
#Oxy_pbc        = input("Write 0 if interaction with multiple adsorbed oxygens are allowed and 1 if not : ")
#slab_pbc       = True if Oxy_pbc == '0' else False 
slab_pbc       = True 

#print()
#detector_high = float(input("Enter the high (Angstrom) over HOPG to consider the molecule scattered (0 to stop at the end of the simulation): "))
detector_high  = 7.0
#print()

# Some constants
O_mass         = 15.9999    # u
N_mass         = 14.0067    # u

#Savitzky-Golay filter parameters
filter_size    = 199
poly_order     = int(2)

time_flag      = ['IS', 'TS', 'FS', 'Turn_Point', 'End']
all_flags      = ['IS', 'TS', 'FS', 'Turn_Point', 'End', 'NO2_End', 'Start']
data           = {}

################################    
### Here start the action    ###
################################

for energy in E:
    folder     = f'VASP_format_{incidence}_{energy}eV/'
    
    # Dictionary declaration for the flags to analyze
    for elem in ['comment', 'cfg_coord', 'cfg_vel']:
        for flag in all_flags:
            key = f"{flag}_{elem}"
            data[key] = []
    
    # Loop over all the trajectories to analyze
    for i in tqdm(range(t_start, t_end + 1), desc="Processing " + str(energy) + " eV", ncols=100):
        
        #########################################################
        ###   Load the AIMD_resume, XDATCAR and VDATCAR files ###
        #########################################################
        
        # Load the trayectory files if there exits and are not empty
        if os.path.exists(folder + "aimd_resume-" + str(i) + ".dat") == False or os.path.getsize(folder + "/aimd_resume-" + str(i) + ".dat") == 0:      
            print()
            print("AIMD resume file for trayectory " + str(i) + " in " + str(energy) + " eV present error")
            continue
        else:
            AIMD_resume   = np.loadtxt(folder + "aimd_resume-" + str(i) + ".dat")
            
            # Get energy from the Savgol filter
            Epot_filtered = savgol_filter(AIMD_resume[:, 3], filter_size, poly_order)
        
        # Load velocities
        if os.path.exists(folder + "VDATCAR-" + str(i)) == False or os.path.getsize(folder + "/VDATCAR-" + str(i)) == 0:      
            print()
            print("VDATCAR resume file for trayectory " + str(i) + " in " + str(energy) + " eV present error")
            continue
        else:
            VDATCAR_data = np.loadtxt(folder + "VDATCAR-" + str(i))
        
        # Load the positions
        if os.path.exists(folder + "XDATCAR-" + str(i)) == False or os.path.getsize(folder + "/XDATCAR-" + str(i)) == 0:      
            print()
            print("XDATCAR resume file for trayectory " + str(i) + " in " + str(energy) + " eV present error")
            continue
        else:
            XDATCAR_data = parse_XDATCAR(folder + "XDATCAR-" + str(i))
            
            # Convert Direct coordenates into Cartesian coordinates
            XDATCAR_data['coordinates'] = XDATCAR_data['coordinates'] @ XDATCAR_data['lattice_vectors']
            
            # Fix coordinates if NO are splited at the begining by the PBC
            XDATCAR_data['coordinates'] = get_noPBC_initial(XDATCAR_data['coordinates'], VDATCAR_data, XDATCAR_data['lattice_vectors'])

        # Check all the files have the same snapshots
        if (len(VDATCAR_data)/XDATCAR_data['atoms_counts']) != XDATCAR_data['time'] or len(AIMD_resume) != XDATCAR_data['time'] or len(AIMD_resume) != (len(VDATCAR_data)/XDATCAR_data['atoms_counts']):
            print()
            print("Warning  !!! -> Trayectory", i, "have diferents input files size -> XDATCAR:", XDATCAR_data['time'], 'VDATCAR:',  len(VDATCAR_data)/XDATCAR_data['atoms_counts'], 'AIMD_resume:', len(AIMD_resume))
            #continue

        #########################################################
        ###          Get all times to generate XYZ            ###
        #########################################################
        
        # Get all the relevant times during the trajectory 
        time_data = get_times(XDATCAR_data['coordinates'], VDATCAR_data, Epot_filtered, XDATCAR_data['atoms_counts'], XDATCAR_data['time'])
 
        # Take into account if the simulation end before the reaction finish
        if time_data['FS'] < time_data['IS']:
            print("Warning  !!! -> Trayectory " + str(i) + " have IS time: " + str(time_data['IS']) + " fs and FS time: " + str(time_data['FS']) + " fs, simulation should be continued")
            time_data['FS'] = time_data['End']
            time_data['TS'] = int((time_data['IS'] + time_data['FS']) / 2)
 
        # Take into account if the molecule go out of the slab before the reaction finish
        if time_data['End'] < time_data['FS']:
            print("Warning  !!! -> Trayectory " + str(i) + " have FS time: " + str(time_data['FS']) + " fs and end time: " + str(time_data['End']) + " fs, will not be take into account")
            continue
        
        # Check for electronic convergence at the last ionic step
        if time_data['End'] == AIMD_resume.shape[0]:
            print("There was a problem in the electronic calculation for time " + str(time_data['End']) + " fs at trayectory " + str(i))
            time_data['End'] -= 1

        #########################################################
        ###          Get Trayectory Start Configurations      ###
        #########################################################     
        
        data['Start_comment'].append([i, AIMD_resume[0][0], AIMD_resume[0][1], AIMD_resume[0][2], AIMD_resume[0][3], round(Epot_filtered[0], 5), AIMD_resume[0][5], np.sum(time_data['Bounce'] <= 1)])
        data['Start_cfg_coord'].append(get_config(XDATCAR_data['coordinates'], 1, XDATCAR_data['atoms_counts']))
        data['Start_cfg_vel'].append(get_config(VDATCAR_data, 1, XDATCAR_data['atoms_counts'])) 

        #########################################################
        ###          Get all the others Configuration         ###
        #########################################################
        
        # Fix coordinates moved by PBC
        for elem in time_flag:
            
            if time_data[elem] != 0: 
                coord_cfg    = get_config(XDATCAR_data['coordinates'], time_data[elem], XDATCAR_data['atoms_counts']) 
                vel_cfg      = get_config(VDATCAR_data, (time_data[elem] - 1 if time_data[elem] == XDATCAR_data['time'] - 1 else time_data[elem]), XDATCAR_data['atoms_counts'])
                config_param = get_Params(coord_cfg, vel_cfg)
            
                if config_param['NO2_form']:
                    if config_param['d_NO1'] > 4 or config_param['d_NO2'] > 4:
                        coord_cfg = get_config(get_noPBC(XDATCAR_data['coordinates'], time_data['Turn_Point'], time_data['End'], XDATCAR_data['atoms_counts'], XDATCAR_data['lattice_vectors']), time_data[elem], XDATCAR_data['atoms_counts'] )
                else:
                    if config_param['d_NO1'] > 4:
                        coord_cfg = get_noPBC_end(coord_cfg, config_param['mol_mCenter_vel'], XDATCAR_data['lattice_vectors'])
    
                # Add the configurations to the list
                aimd_pos = time_data[elem] - 1
                data[elem + '_comment'].append([i, AIMD_resume[aimd_pos][0], AIMD_resume[aimd_pos][1], AIMD_resume[aimd_pos][2], AIMD_resume[aimd_pos][3], round(Epot_filtered[aimd_pos], 5), AIMD_resume[aimd_pos][5], np.sum(time_data['Bounce'] <= time_data[elem] - 1)])
                data[elem + '_cfg_coord'].append(coord_cfg)
                data[elem + '_cfg_vel'].append(get_config(VDATCAR_data, (time_data[elem] - 1 if time_data[elem] == XDATCAR_data['time'] - 1 else time_data[elem]), XDATCAR_data['atoms_counts']))  
            
                #########################################################
                ###          Get Trayectory NO2 End Configuration     ###
                #########################################################
            
                if elem == 'End' and config_param['NO2_form'] and time_data['IS'] != 0:
                    data['NO2_End_comment'].append([i, AIMD_resume[aimd_pos][0], AIMD_resume[aimd_pos][1], AIMD_resume[aimd_pos][2], AIMD_resume[aimd_pos][3], round(Epot_filtered[aimd_pos], 5), AIMD_resume[aimd_pos][5], np.sum(time_data['Bounce'] <= time_data[elem] - 1)])
                    data['NO2_End_cfg_coord'].append(coord_cfg)
                    data['NO2_End_cfg_vel'].append(get_config(VDATCAR_data, ((time_data[elem] - 1) if (time_data[elem] == XDATCAR_data['time'] - 1) else time_data[elem]), XDATCAR_data['atoms_counts'])) 
                    
                        
                # Check if the reactivity is obteined from energy and geometry
                if config_param['NO2_form'] and time_data['IS'] == 0:
                    print("Trayectory " + str(i) + " should be continued, reaction start but simulation was stopped before finish")

    #########################################################
    ###          Get Trayectory NO2 End Configuration     ###
    #########################################################
    
    # Selector if there is a test or in the cluster
    conditions  = ("PBC_" if slab_pbc else "noPBC_") + str(int(detector_high)) + "A" 
    #OUTPUT_PATH = f"rcut_6.0_30t_30t_FPv0_seed10_100k_epoch_batch1024_lrE5_{incidence}_{conditions}_XYZ_files/"
    #OUTPUT_PATH = f"{folder.replace(f'_{incidence}_{energy}eV', '')}_{incidence}_{conditions}_XYZ_files/"
    OUTPUT_PATH = f'{incidence}_{conditions}_XYZ_files/'
    
    # Check if the path exist
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    # Write all the XYZ files
    for flag in all_flags:
        write_xyz(f"{OUTPUT_PATH}{energy}eV_Tray_{flag}.xyz", data[flag + '_comment'], data[flag + '_cfg_coord'], data[flag + '_cfg_vel'], XDATCAR_data['atoms_counts'])

print("Done !!!")
