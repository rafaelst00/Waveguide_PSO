# Waveguide_PSO
This python program optimizes speaker waveguide shapes with a particle swarm optimization algorithm. It is uses on Advance Transition Horns (ATH) to create the form with the OS-SE formula and simulates the frequency response with ABEC (now AKABAK). Then  a objective function is used to evaluate the frequency response.

# Getting started
To start the tool ath (https://at-horns.eu/) and ABEC (http://www.randteam.de/_Software/ABEC3_Pro/Download-ABEC-Pro.html) are necessary. It is helpful to get to know and test both tools before you try to automate. To continue set the paths in single_test.py and try to run it. The way python calls ABEC is not very robust becuase it works through the GUI. For it to work it has to see the status symbols of the ABEC window. These are seen with coordinates, which may need to be changed depending on the system (abec_simulation.py -> get_abec_status_markers). If single_test.py works you can continue to pso_optimize.

# Values to set
There are different Values that need to be set, so that the algorithm runs to your expectation. First should be the mesh settings in create_mesh.py Here it is helpful to read the documenation of ATH as the format is the same. Some parameters are optimized through the PSO, but for the single_test.py have to be set in ATHOSSEParams. These define the essential form of the waveguide. In pso_optimize these values have boundaries in which the PSO will try to find an optimal combination. The boundaries should be set carefully to allow the PSO to explore enough, but if the values are too wide it will take very long to find good solutions.

Then there are the ObjectiveWeights. The objective function uses multiple scores on multiple polar planes to evaluate the frequency response of the waveguide driver combination. It evaluates 4 planes from vertical to horizonal and then looks at the frequency response from 10 to 60 degrees in 10 degree steps. The different planes (horizontal, vertical and in between) can be custom weighted. Then the listening angles can be weighted as well - for example giving the inner angles more weight.
The main scoring functions are
tonal_balance        # Keeps off-axis tonal shape similar to 0° after removing average level loss.
freq_rise            # Penalizes response rising with frequency; falling/flat is allowed.
freq_ripple          # Penalizes narrow frequency ripple using second frequency difference.
angular_smoothness=  # Penalizes sudden angular jumps using second angle difference.
angular_monotonicity # Penalizes larger angles being louder than smaller angles.
You can also set a max_width_mm and max_height_mm to punish waveguides that are bigger than your design allows for.

Lastly you can set the PSO parameters: cognitive coefficient c1, social coefficien c2, inertia weight w, the particle amount and the iterations. c1, c2 and w have good default values.

