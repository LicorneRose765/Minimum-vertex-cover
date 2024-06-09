//! Find the value of the maximum clique.
//!
//! Run the B&B algorithm on the given graph and print the value of the maximum clique. 
//!
//! # Arguments
//! * `graph_name` - Name of the graph file in the src/resources/graphs directory.
//! * `time_limit` - Time limit in seconds.
//! * `on complement` - Optional flag to run the algorithm on the complement of the graph.
//!
//! # Example
//! This example will run find the maximum clique of the queen5_5.clq graph using the branch and bound algorithm.
//! The search cannot take more than 60 seconds.
//! ```bash
//! cargo run -r --bin clique queen5_5.clq 60
//! ```
use std::env;

use petgraph::prelude::UnGraphMap;
use round::round;

use vertex::branch_and_bound;
use vertex::graph_utils::{complement, is_vertex_cover, load_clq_file};
use vertex::result_utils::{Clock, MVCResult};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() == 3 {
        let graph = match load_clq_file(&format!("src/resources/graphs/{}", args[1])) {
            Ok(graph) => graph,
            Err(e) => panic!("Error while loading graph : {}", e),
        };

        let time_limit = match args[2].parse::<u64>() {
            Ok(t) => t,
            Err(_) => {
                eprintln!("Error: time limit must be a positive integer");
                eprintln!("Usage: cargo run [-r] --bin clique <graph_name> <time_limit>");
                return;
            }
        };

        find_max_clique(&args[1], &graph, time_limit);
    } else {
        eprintln!("Usage: cargo run [-r] --bin clique <graph_name> <time_limit>");

    }
}



fn find_max_clique(graph_id: &str, graph: &UnGraphMap<u64, ()>, time_limit: u64) {
    let g = complement(graph);
    let density = (2 * g.edge_count()) as f64 / (g.node_count() * (g.node_count() - 1)) as f64;
    println!("Finding max clique of the graph. Specificity of the complement : \nOrder = {} and size = {}. Density = {}",
             g.node_count(),
             g.edge_count(),
             density);

    let mut clock = Clock::new(time_limit);

    let res = branch_and_bound(&g, &mut clock, None, None);
    clock.stop_timer();

    assert!(is_vertex_cover(&g, &res.1));

    let clique_val = graph.node_count() as u64 - res.0;


    let res = match MVCResult::new(graph_id.to_string(), clique_val, res.1, clock.get_time(), clock.is_time_up(), true, true) {
        Ok(res) => res,
        Err(e) => panic!("Error while creating MVCResult : {}", e),

    };

    output_reaction(res, &clock);

}



fn output_reaction(res: MVCResult, clock: &Clock) {
    println!("================ Result ===================\n{}", res);
    println!("======== Details about performance ========");
    println!("Time spent in deg : {}%", round(clock.get_subroutine_duration("deg_lb").as_secs_f64() * 100.0
        / clock.get_time().duration.as_secs_f64(), 4));
    println!("Time spent in clq : {}%", round(clock.get_subroutine_duration("clq_lb").as_secs_f64() * 100.0
            / clock.get_time().duration.as_secs_f64(), 4));
    println!("Time spent in max deg : {}%", round(clock.get_subroutine_duration("max_deg").as_secs_f64() * 100.0
                / clock.get_time().duration.as_secs_f64(), 4));
    println!("Time spent in copy : {}%", round(clock.get_subroutine_duration("copy").as_secs_f64() * 100.0
                    / clock.get_time().duration.as_secs_f64(), 4));

    let _comment = "Multithreaded lower bound";
    /* add_time_to_yaml(&res.graph_id,
                     res.value,
                     res.time,
                     res.is_time_limit,
                     "clique",
                     comment).expect("Error while adding time to yaml");
     */
}