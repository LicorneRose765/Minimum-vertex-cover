# Minimum-vertex-cover

![Rust workflow](https://github.com/LicorneRose765/Minimum-vertex-cover/actions/workflows/rust.yml/badge.svg)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/LicorneRose765/ClockSystem/badge)](https://securityscorecards.dev/viewer/?uri=github.com/LicorneRose765/ClockSystem)

[![codecov](https://codecov.io/gh/LicorneRose765/Minimum-vertex-cover/graph/badge.svg?token=AC37S9XQPX)](https://codecov.io/gh/LicorneRose765/Minimum-vertex-cover)

Comparison of algorithms (exact and heuristic) for the minimum vertex cover problem

Documentation can be found [here](https://licornerose765.github.io/Minimum-vertex-cover/)

## Algorithms
### Exact algorithms
* Naive method : iterate over all possible subsets of vertices and check if it is a vertex cover.  
use : `cargo run -r --bin naive_search <file_name> <time_limit>`
* Branch and bound : Algorithm based on the paper presented by Wang et al. 
[Source](https://doi.org/10.3390/math7070603)  
use : `cargo run -r --bin bnb <file_name> <time_limit> [-c]`

### Heuristic algorithms
* NuMVC : Algorithm proposed by Cai et al. in their paper : __NuMVC: An Efficient Algorithm for the Minimum Vertex Cover Problem__.
You can find the paper [here](	https://doi.org/10.1613/jair.3907)  
use :`cargo run -r --bin numvc <file_name> <time_limit> [-c]` 

**Remark**: Time limit is in seconds and filename is the name of a file in the resources/graphs folder.

## Bins 
* **naive_method**: Find the MVC of the graph using the naive method.   
use : `cargo run -r --bin naive_search <file_name>`
* __add_graph_to_yaml__: Update the graph information in the yaml file (get the graphs in the resources/graphs folder)  
use : `cargo run -r --bin add_graph_to_yaml`
* **bnb**: Find the MVC of the graph (or the complement if -c is added) using the branch and bound algorithm.  
use : `cargo run -r --bin bnb <file_name> [-c]`
* **clique**: Find the value of the maximum clique of the graph by find the MVC of the complement using the BnB algorithm.  
use : `cargo run -r --bin clique <file_name>`
* **numvc**: Find the MVC of the graph (or the complement if -c is added) using the NuMVC algorithm.  
use: `cargo run -r --bin numvc <file_name> <time_limit> [-c]`