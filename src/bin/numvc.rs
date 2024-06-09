//! NUMVC algorithm binary.
//!
//! Run the NUMVC algorithm on the given graph and print the result.
//! 
//! # Arguments
//! * `graph_name` - Name of the graph file in the src/resources/graphs directory.
//! * `time_limit` - Time limit in seconds.
//! * `on complement` - Optional flag to run the algorithm on the complement of the graph.
//!
//! # Example
//! This example will run the NUMVC algorithm on the complement of the brock200_1.clq graph for maximum 60 seconds.
//! ```bash
//! cargo run -r --bin numvc brock200_1.clq 60 -c
//! ```
use std::env;

use vertex::{numvc, read_arguments};

fn main() {
    let args: Vec<String> = env::args().collect();
    let res = read_arguments(args, &numvc);
    
    if res.is_none() {
        eprintln!("Usage: cargo run [-r] --bin numvc <graph_name> <time_limit> [(on complement) -c]");
        return;
    }
    let res = res.unwrap();
    println!("Result : {}", res);
}