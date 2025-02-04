import os
import subprocess
import argparse

# This script is used to generate samples of the HW2D model using the HW2D library.

if __name__ == "__main__":
    # Initialize the argument parser and add the arguments
    parser = argparse.ArgumentParser(description="Set parameters for the model and movie generation.")

    parser.add_argument("--sample_index", type=int, required=True, 
                        help="Index of the sample from SLURM job array"
                        )
    parser.add_argument("--save_path", type=str, required=True, 
                        help="Path to save the output files."
                        )
    parser.add_argument("--step_size", type=float, default=0.02, 
                        help="Time step size for the simulation."
                        )
    parser.add_argument("--end_time", type=int, default=800, 
                        help="End time for the simulation."
                        )
    parser.add_argument("--grid_pts", type=int, default=512, 
                        help="Number of grid points. Suggested range: 128 to 1024."
                        )
    parser.add_argument("--c1", type=float, required=True, 
                        help="Transition scale between hydrodynamic and adiabatic regimes."
                        )
    parser.add_argument("--k0", type=float, default=0.15, 
                        help="Determines k-focus. Suggested values: 0.15 for high-k, 0.0375 for low-k."
                        )
    parser.add_argument("--N", type=int, default=3, 
                        help="Number of simulations or configurations."
                        )
    parser.add_argument("--nu", type=float, required=True, 
                        help="Viscosity parameter."
                        )
    parser.add_argument("--kappa_coeff", type=float, required=True, 
                        help="Coefficient for d/dy phi. Defaults to 1.0."
                        )
    parser.add_argument("--buffer_length", type=int, default=100, 
                        help="Buffer length for simulations."
                        )
    parser.add_argument("--snaps", type=int, default=50, 
                        help="Number of snapshots, each representing a time difference."
                        )
    parser.add_argument("--downsample_factor", type=int, default=4, 
                        help="Downsampling factor. 4 results in a 128x128 resolution."
                        )
    parser.add_argument("--movie", type=int, default=0, 
                        help="Enable or disable movie generation (0 or 1)."
                        )
    parser.add_argument("--speed", type=int, default=5, 
                        help="Speed of movie playback."
                        )
    parser.add_argument("--debug", type=int, default=0, 
                        help="Enable debug mode (0 or 1)."
                        )
    parser.add_argument("--num_samples", type=int, default=1, 
                        help="Number of samples to generate per job."
                        )
    args = parser.parse_args()

    # Assign parameters to variables
    task_id = args.sample_index
    save_path = args.save_path
    step_size = args.step_size
    end_time = args.end_time
    grid_pts = args.grid_pts
    c1 = args.c1
    k0 = args.k0
    N = args.N
    nu = args.nu
    kappa_coeff = args.kappa_coeff
    buffer_length = args.buffer_length
    snaps = args.snaps
    downsample_factor = args.downsample_factor
    movie = args.movie
    speed = args.speed
    debug = args.debug
    num_samples = args.num_samples

    # Directory to store output files
    folder_name = f"k{kappa_coeff}_c{c1}_nu{nu}"

    #current_dir = os.path.dirname(os.path.abspath(__file__))
    #path = os.path.join(current_dir, '..', '..', folder_name)
    path = os.path.join(save_path, folder_name)

    # Create the output directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Number of samples to generate
    num_samples = num_samples  
    j = 0
    output_path = os.path.join(path, f"sample_{j}_{task_id}.h5")
    while os.path.exists(output_path): # don't overwrite existing files
            j += 1
            output_path = os.path.join(path, f"sample_{j}_{task_id}.h5")
    for i in range(num_samples):
        output_path = os.path.join(path, f"sample_{i+j}_{task_id}.h5")
        
        # Call the model with parameters
        print(f'Running sample {i+j} with task_id {task_id}')
        subprocess.run([
            "python", "-m", "hw2d",
            f"--step_size={step_size}",
            f"--end_time={end_time}",
            f"--grid_pts={grid_pts}",
            f"--c1={c1}",
            f"--k0={k0}",
            f"--N={N}",
            f"--nu={nu}",
            f"--kappa_coeff={kappa_coeff}",
            f"--output_path={output_path}",
            f"--buffer_length={buffer_length}",
            f"--snaps={snaps}",
            f"--downsample_factor={downsample_factor}",
            f"--movie={movie}",
            f"--speed={speed}",
            f"--debug={debug}"
        ])

    # Save the parameters used for the samples generation
    params_file_path = os.path.join(path, "parameters.txt")
    with open(params_file_path, 'w') as params_file:
        params_file.write("Parameters used for samples generation:\n")
        params_file.write(f"step_size={step_size}\n")
        params_file.write(f"end_time={end_time}\n")
        params_file.write(f"grid_pts={grid_pts}\n")
        params_file.write(f"c1={c1}\n")
        params_file.write(f"k0={k0}\n")
        params_file.write(f"N={N}\n")
        params_file.write(f"nu={nu}\n")
        params_file.write(f"kappa_coeff={kappa_coeff}\n")
        params_file.write(f"buffer_length={buffer_length}\n")
        params_file.write(f"snaps={snaps}\n")
        params_file.write(f"downsample_factor={downsample_factor}\n")
        params_file.write(f"movie={movie}\n")
        params_file.write(f"speed={speed}\n")
        params_file.write(f"debug={debug}\n")