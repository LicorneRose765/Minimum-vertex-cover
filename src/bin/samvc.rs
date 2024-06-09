//! SAMVC algorithm binary.
//!
//! # Arguments
//! * `graph_name` - Name of the graph file in the src/resources/graphs directory.
//! * `time_limit` - Time limit in seconds.
//! * `on complement` - Optional flag to run the algorithm on the complement of the graph.
//!
//! # Example
//! This example will run the SAMVC algorithm on the complement of the brock200_1.clq graph for maximum 60 seconds.
//! ```bash
//! cargo run -r --bin samvc brock200_1.clq 60 -c
//! ```
use std::env;

use vertex::samvc;

fn main() {
    let args: Vec<String> = env::args().collect();
    let res = vertex::read_arguments(args, &samvc);

    if res.is_none() {
        eprintln!("Usage: cargo run [-r] --bin samvc <graph_name> <time_limit> [(on complement) -c]");
        return;
    }
    let res = res.unwrap();
    println!("Result : {}", res);

    /* Uncomment this block to add time to yaml
            let comment = format!("Initial temperature = {}, Final temperature = {}, Cooling rate = {}",
                                 initial_temp,
                                 final_temp,
                                 cooling_rate);
            add_time_to_yaml(&res.graph_id,
                             res.value,
                             res.time,
                             res.is_time_limit,
                             "SAMVC",
                             &comment).expect("Error while adding time to yaml");
            */
}