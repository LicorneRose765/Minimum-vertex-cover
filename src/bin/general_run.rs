//! General binary to run any algorithm.
//!
//! You can use this binary to run any algorithm on a given graph.
//!
//!  # Arguments
//! * `algorithm_name` - Name of the algorithm to run.
//! * `graph_name` - Name of the graph file in the src/resources/graphs directory.
//! * `time_limit` - Time limit in seconds.
//! * `on complement` - Optional flag to run the algorithm on the complement of the graph.
//!
//! # Example
//! ```bash
//! cargo run -r --bin general_run <algorithm_name> <graph_name> <time_limit> [(on complement) -c]
//! ```

use vertex::read_arguments;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: cargo run [-r] --bin general_run <algorithm_name> <graph_name> <time_limit> [(on complement) -c]");
        return;
    }
    let algorithm_name = args[1].as_str();
    let args: Vec<String> = args[1..].to_vec();
    let res = match algorithm_name {
        "naive_search" => read_arguments(args, &vertex::naive_search),
        "numvc" => read_arguments(args, &vertex::numvc),
        "bnb" => read_arguments(args, &vertex::branch_and_bound),
        "samvc" => read_arguments(args, &vertex::samvc),
        _ => {
            eprintln!("Unknown algorithm: {}", args[1]);
            eprintln!("Available algorithms: naive_search, numvc, bnb, samvc");
            return;
        }
    };
    
    let res = res.unwrap();
    println!("Result : {}", res);
}