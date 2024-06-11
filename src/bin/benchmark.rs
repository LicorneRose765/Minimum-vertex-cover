use std::fs::File;
use std::io::BufRead;
use std::ops::Add;

use vertex::{branch_and_bound, run_algorithm};
use vertex::graph_utils::add_time_to_yaml;

fn main() {
    // Read all lines of benchmark-HoG.txt and put it in a vector
    let file = File::open("src/resources/benchmark-DIMACS-small.txt").unwrap();
    let reader = std::io::BufReader::new(file);
    // Read all lines not starting with #
    let lines: Vec<String> = reader.lines()
        .map(|l| l.unwrap())
        .filter(|l| !l.starts_with('#'))
        .collect();


    let algorithm = &branch_and_bound;
    let algorithm_str = "Branch and Bound (with satLB)";
    let on_complement = true;
    let time_limit = 1800; // 30 minutes time limit

    // For each line, execute the algorithm with the given parameters
    for line in lines {
        let graph_name = line.trim();
        println!("==================== {} =======================", graph_name);
        let start_time = chrono::Local::now();
        println!("Start time : {}",start_time.format("%H:%M:%S"));
        println!("Max end time : {}", start_time.add(chrono::Duration::seconds(time_limit as i64)).format("%H:%M:%S"));
        println!("==================================================");

        let graph = vertex::graph_utils::load_clq_file(
            &format!("src/resources/graphs/{}", graph_name)).unwrap();
        let res = run_algorithm(graph_name, &graph, algorithm, None, time_limit, on_complement, false);
        match res {
            Ok(res) => {
                println!("Result : {}", res);
                let comment = "Final benchmark - 30 minutes time limit - with satLB";
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