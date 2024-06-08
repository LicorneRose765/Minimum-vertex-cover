use std::env;

use vertex::{run_algorithm, samvc};
use vertex::graph_utils::{add_time_to_yaml, load_clq_file};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 3 && args.len() <= 4 {
        let graph = load_clq_file(&format!("src/resources/graphs/{}", args[1]))
            .expect("Error while loading graph");

        let time_limit = match args[2].parse::<u64>() {
            Ok(t) => t,
            Err(_) => {
                eprintln!("Error: time limit must be a positive integer");
                eprintln!("Usage: cargo run [-r] --bin samvc <graph_name> <time_limit> [(on complement) -c]");
                return;
            }
        };

        if args.len() == 4 && args[3] == "-c" {
            let initial_temp = 100.0;
            let final_temp = 0.01;
            let cooling_rate = 0.95;
            let params = vec![initial_temp, final_temp, cooling_rate];
            let res = match run_algorithm(&args[1], &graph, &samvc, Some(params.as_slice()), time_limit, true, false) {
                Ok(res) => {
                    println!("Result : {}", res);
                    res
                },
                Err(e) => {
                    println!("Error : {}", e);
                    return;
                },
            };
            
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
            return;
        }

        match run_algorithm(&args[1], &graph, &samvc, None, time_limit, false, false) {
            Ok(res) => println!("Result : {}", res),
            Err(e) => println!("Error : {}", e),
        };
    } else {
        eprintln!("Usage: cargo run [-r] --bin samvc <graph_name> <time_limit> [(on complement) -c]");
    }
}