use std::fs::File;
use std::io::BufRead;

use vertex::{run_algorithm, samvc};
use vertex::graph_utils::add_time_to_yaml;

fn main() {
    // Read all lines of benchmark.txt and put it in a vector
    let file = File::open("src/resources/benchmark.txt").unwrap();
    let reader = std::io::BufReader::new(file);
    let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();

    let algorithm = &samvc;
    let algorithm_str = "SAMVC";
    let on_complement = true;
    let time_limit = 600; // 1 hour time limit

    // For each line, execute the algorithm with the given parameters
    for line in lines {
        let graph_name = line.trim();
        println!("==================== {} =======================", graph_name);

        let graph = vertex::graph_utils::load_clq_file(
            &format!("src/resources/graphs/{}", graph_name)).unwrap();
        let res = run_algorithm(graph_name, &graph, algorithm, None, time_limit, on_complement, false);
        match res {
            Ok(res) => {
                println!("Result : {}", res);
                let comment = "Final benchmark - 10 minutes time limit";
                match add_time_to_yaml(&res.graph_id,
                                       res.value,
                                       res.time,
                                       res.is_time_limit,
                                       algorithm_str,
                                       comment) {
                    Ok(_) => {}
                    Err(e) => eprintln!("Error while adding time to yaml : {}", e),
                }
            }
            Err(e) => {
                eprintln!("Error while running algorithm : {}", e);
            }
        }
        println!("==================================================");
    }
}