use std::env;

use vertex::graph_utils::load_clq_file;
use vertex::naive_search;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() == 3 {
        let graph = load_clq_file(&format!("src/resources/graphs/{}", args[1]))
            .expect("Error while loading graph");

        let time_limit = match (&args[2]).parse::<u64>() {
            Ok(t) => t,
            Err(_) => {
                eprintln!("Error: time limit must be a positive integer");
                eprintln!("Usage: cargo run [-r] --bin naive_search <graph_name> <time_limit>");
                return;
            }
        };

        // Run algorithm without updating value
        println!("/!\\ This algorithm compute the MVC value on the complement graph by default /!\\");
        let res = vertex::run_algorithm(&args[1], &graph, &naive_search, None, time_limit, true, false)
            .unwrap_or_else(|e| {
                panic!("Error while running algorithm : {}", e);
            });

        println!("Result : {}", res);
        /*add_time_to_yaml(&res.graph_id,
                         res.value,
                         res.time,
                         res.is_time_limit,
                         "naive_search",
                         "").expect("Error while adding time to yaml file");
         */
    } else {
        println!("Usage: cargo run [-r] --bin naive_search <graph_name> <time_limit>");
    }
}