
Rules of the road for feature extraction script:
 - every parameter gets a sum/mean/median/stdv/min/max at all levels 
 - some of these may only apply at the cell level and not at the individual organelle/object level... 
 - keep everythin that we are measuring oragnized in @feature_extraction_parameters.md  
 - also if any of these calcualtions are very comutationaly expensive then desingate them in a 'full_features' optional parameter to turn on or off to speed things up 

Features are aggregated (sum, mean, median, stdv, min, max) up scales: 
1. object (organelle)
2. cell 
3. guideRNA 
4. gene (n=4 guides per gene, or 100s(?) of NTCs)
5. pathway annoation 

Comprehensive set of geometric, topological, and textural properties of networked organelles, building upon your initial list.
[] = not implemented
[+] = implemented
[+] (full_features) = implemented but only extracted when full_features=True (expensive)


I. Organelle Level (sum, mean, median, stdv, min, max)
[+] intensity_mean (regionprops)
[+] intensity_max (regionprops)
[+] intensity_min (regionprops)
[+] intensity_std (regionprops - previously manual)
[+] intensity_median (regionprops - previously manual)
[+] intensity_q25 (25th percentile / lower quartile - manual)
[+] intensity_q75 (75th percentile / upper quartile - manual)
[+] intensity_iqr (interquartile range = q75 - q25 - manual)
[+] intensity_mad (median absolute deviation - robust variability - manual)
[+] intensity_integrated (sum of all pixel intensities - total signal - manual)
[+] intensity_range (dynamic range = max - min - derived)
[+] intensity_cv (coefficient of variation = std / mean - normalized variability - derived)
[+] frangi intensity

Shape Features (from regionprops - FREE):
[+] equivalent_diameter_area: Diameter of circle with same area
[+] area_filled: Area with internal holes filled
[+] area
[+] axis_major_length
[+] axis_minor_length
[+] extent
[+] orientation
[+] eccentricity
[+] hu moments_hu[]0 moments_hu[]1 moments_hu[]2 moments_hu[]3 moments_hu[]4 moments_hu[]5 moments_hu[]6
[+] moments_weighted_hu_0 through _6: 7 intensity-weighted Hu moments (shape weighted by intensity distribution)
[+] inertia_eigval_0, inertia_eigval_1: Rotational inertia eigenvalues (indicates elongation)
[+] aspect_ratio: axis_major_length / axis_minor_length

Cheap Approximations (ALWAYS extracted - derived from axis lengths, ~FREE):
[+] perimeter_approx: Ramanujan ellipse approximation: π * (3(a+b) - sqrt((3a+b)(a+3b)))
    - Uses semi-axes from axis_major/minor_length (already extracted)
    - ~0.5% error vs true perimeter, good for relative comparisons
[+] circularity_approx: (4π * area) / perimeter_approx² - how close to a circle (1.0 = perfect circle)
    - Derived from perimeter_approx instead of expensive boundary tracing
[+] solidity_approx: Uses extent as proxy (extent = area / bounding_box_area)
    - True solidity requires convex hull computation
    - Extent correlates 0.7-0.9 with solidity for biological shapes

Expensive Shape Features (full_features=True):
[+] euler_number (full_features): Objects minus holes (topological measure)
    - No cheap approximation exists - measures unique hole topology
    - Requires topology analysis
[+] area_convex (full_features): True convex hull area
    - No cheap approximation exists
    - solidity_approx uses extent as proxy, but area_convex gives true convex hull

DEPRECATED - Expensive features replaced by cheap approximations above:
    These are commented out in code but can be re-enabled if needed:
    - perimeter: True boundary tracing (~10-50ms per object)
    - perimeter_crofton: More accurate perimeter via Crofton formula
    - solidity: area / area_convex (use solidity_approx instead)
    - convexity: perimeter / convex_hull_perimeter

Centroid/Location Features (from regionprops - FREE):
[+] centroid_y, centroid_x: Object center coordinates (in physical units)
[+] centroid_weighted_y, centroid_weighted_x: Intensity-weighted center (mass center)
    - This is equivalent to CellProfiler's "mass displacement" concept
    - Difference between centroid and centroid_weighted indicates intensity polarity
[+] equivalent_diameter: The diameter of a circle with the same area as the region. This normalizes size measurement.
[+] eccentricity: The eccentricity of the ellipse that has the same second[]moments as the region. A value of 0 indicates a circle, while a value close to 1 indicates a line.
[+] feret_diameter_max (full_features): The longest distance between any two points along the object's boundary, also known as the maximum caliper diameter.
[] feret_diameter_min: The shortest distance between any two points along the object's boundary, also known as the minimum caliper diameter.
[+] zernike_moments (full_features): A set of orthogonal moments used to describe the shape of an object. They can capture high[]frequency details and are robust to noise. You can specify the order and radius for these moments.
[+] haralick_features (full_features): A set of texture features derived from the Gray[]Level Co[]occurrence Matrix (GLCM). Key Haralick features include:
[+]   1. contrast: Measures local variations in the gray[]level co[]occurrence matrix.
[+]   2. correlation: Measures the joint probability of occurrence of specified pixel pairs.
[+]   3. energy (or Angular Second Moment): Provides the sum of squared elements in the GLCM. It is a measure of the textural uniformity.
[+]   4. homogeneity (or Inverse Difference Moment): Measures the closeness of the distribution of elements in the GLCM to the GLCM diagonal.


II. Network & Topology Properties (sum, mean, median, stdv, min, max)

These features treat the organelle as a graph to describe its connectivity and branching patterns.

[+] branch_length 
[+] branch_thickness
[+] tortuosity
[+] num_nodes (or Junctions): The total count of points where three or more branches intersect.
[+] num_endpoints: The total count of branch endings (nodes with a degree of 1).
[+] num_branches: The total count of individual segments connecting nodes and endpoints.
[+] average_degree: The average number of branches connected to each node, indicating the overall connectivity of the network.

network properites at cell Level (sum, mean, median, stdv, min, max)...

[+] network_length_density (network[]dependent): The total length of all branches divided by the area of the bounding box or the convex hull of the network.
[+] branching_density (network[]dependent): The number of nodes (junctions) per unit area.
[+] fractal_dimension (network[]dependent, full_features): Measures the complexity and space[]filling capacity of the network. A higher value indicates a more intricate, fragmented, or convoluted structure. This is particularly useful for describing complex branching patterns.
[+] euler_number: In 2D, this is the number of objects minus the number of holes in those objects. It provides a topological measure of the structure's complexity. 
[+] largest_connected_component_size (network[]dependent): The area or length of the largest single interconnected part of the network, which helps distinguish between fragmented and continuous networks.
[+] mesh_area (or Loop Size) (network[]dependent, full_features): The average area of the empty spaces enclosed by the network branches. This requires identifying cycles in the network graph.


III. Subcellular Localization Features (sum, mean, median, stdv, min, max)

These features describe the spatial relationship between organelle objects and cellular landmarks (nucleus, cell boundary). They are biologically informative for detecting phenotypes like ER stress, mitochondrial perinuclear clustering, Golgi dispersal, and lysosome/autophagosome trafficking defects.

Implementation uses KDTree-based queries for ~175x speedup over distance transform approach.

[+] distance_from_cell_edge: Distance from object centroid to nearest cell boundary pixel (physical units).
[+] distance_from_nucleus: Distance from object centroid to nearest nuclear boundary pixel (physical units).
[+] distance_from_nucleus_centroid: Euclidean distance from object centroid to nucleus centroid (physical units).
[+] normalized_radial_position: 0 = at nucleus, 1 = at cell edge. Normalized position within the cell.

Aggregation interpretation:
- Low std + high mean distance = uniformly peripheral organelles
- High std = heterogeneous distribution
- Low mean distance = perinuclear clustering


