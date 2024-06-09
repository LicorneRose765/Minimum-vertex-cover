//! NAIVE algorithm binary.
//!
//! Run the NAIVE algorithm on the given graph and print the result. 
//! Since this algorithm will iterate over all possible subset of vertices, it is not recommended to run it on large graphs.
//!
//! # Arguments
//! * `graph_name` - Name of the graph file in the src/resources/graphs directory.
//! * `time_limit` - Time limit in seconds.
//! * `on complement` - Optional flag to run the algorithm on the complement of the graph.
//!
//! # Example
//! This example will run the NAIVE algorithm on the complement of the queen5_5.clq graph for maximum 60 seconds.
//! ```bash
//! cargo run -r --bin naive_search queen5_5.clq 60 -c
//! ```
use std::env;

use vertex::naive_search;

fn main() {
    let args: Vec<String> = env::args().collect();
    let res = vertex::read_arguments(args, &naive_search);

    if res.is_none() {
        eprintln!("Usage: cargo run [-r] --bin naive_search <graph_name> <time_limit> [(on complement) -c]");
        return;
    }
    let res = res.unwrap();
    println!("Result : {}", res);
}