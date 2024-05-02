use std::env;

use vertex::{branch_and_bound, run_algorithm};
use vertex::graph_utils::load_clq_file;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 3 && args.len() <= 4{
        let graph = load_clq_file(&format!("src/resources/graphs/{}", args[1]))
            .expect("Error while loading graph");

        let time_limit = match args[2].parse::<u64>() {
            Ok(t) => t,
            Err(_) => {
                eprintln!("Error: time limit must be a positive integer");
                eprintln!("Usage: cargo run [-r] --bin bnb <graph_name> <time_limit> [(on complement) -c]");
                return;
            }
        };


        if args.len() == 4 && args[3] == "-c" {
            match run_algorithm(&args[1], &graph, &branch_and_bound, None, time_limit, true, false) {
                Ok(res) => println!("Result : {}", res),
                Err(e) => println!("Error : {}", e),
            }
            return;
        }

        match run_algorithm(&args[1], &graph, &branch_and_bound, None, time_limit, false, false) {
            Ok(res) => println!("Result : {}", res),
            Err(e) => println!("Error : {}", e),
        };
        return
    }
    eprintln!("Usage: cargo run [-r] --bin bnb <graph_name> <time_limit> [(on complement) -c]");
}