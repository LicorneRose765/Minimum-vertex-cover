use itertools::Itertools;
use petgraph::prelude::UnGraphMap;

use crate::branch_and_bound::b_and_b;
use crate::errors::YamlError;
use crate::graph_utils::{copy_graph, get_optimal_value, is_vertex_cover};
use crate::numvc::{add_weight_to_graph, numvc_algorithm};
use crate::result_utils::{Clock, MVCResult};

pub mod graph_utils;
pub mod errors;
pub mod maxsat;
pub mod result_utils;
mod numvc;
mod branch_and_bound;
mod samvc;
mod mvcgraph;

/// Type alias for an algorithm that takes an UnGraphMap as input and computes the minimum vertex cover of the graph.
type Algorithm = dyn Fn(&UnGraphMap<u64, ()>, &mut Clock, Option<&[f64]>, Option<u64>) -> (u64, Vec<u64>);

/// Read the command line arguments given as parameters and run the algorithm based on the given arguments.
///
/// # Arguments
/// * `args` - The command line arguments given to the program. (These can be accessed using `env::args().collect()`)
/// * `algorithm` - The algorithm to run on the graph.
///
/// # Returns
/// * `Some(MVCResult)` - The result of the algorithm if the arguments are correct.
/// * `None` - If the arguments are incorrect. The error message is printed to the standard error and None is returned.
///
/// # Example
/// ```rust
/// use std::env;
/// use vertex::{read_arguments, branch_and_bound};
/// 
/// let args: Vec<String> = env::args().collect(); 
/// let res = read_arguments(args, &branch_and_bound);
/// if res.is_none() { 
///     eprintln!("Usage: cargo run [-r] --bin bnb <graph_name> <time_limit> [(on complement) -c]");
///     return;
/// }
/// let res = res.unwrap();
/// println!("Result : {}", res);
/// ```
pub fn read_arguments(args: Vec<String>, algorithm: &Algorithm) -> Option<MVCResult> {
    if args.len() >= 3 && args.len() <= 4 {
        let graph = graph_utils::load_clq_file(&format!("src/resources/graphs/{}", args[1]))
            .expect("Error while loading graph");

        let time_limit = match args[2].parse::<u64>() {
            Ok(t) => t,
            Err(_) => {
                eprintln!("Error: time limit must be a positive integer");
                return None;
            }
        };

        if args.len() == 4 && args[3] == "-c" {
            let res = match run_algorithm(&args[1], &graph, algorithm, None, time_limit, true, false) {
                Ok(res) => res,
                Err(e) => {
                    println!("Error : {}", e);
                    return None;
                }
            };
            return Some(res);
        }

        let res = match run_algorithm(&args[1], &graph, algorithm, None, time_limit, false, false) {
            Ok(res) => res,
            Err(e) => {
                println!("Error : {}", e);
                return None;
            }
        };
        Some(res)
    } else {
        None
    }
}

/// Naïve algorithm that searches for the minimum vertex cover of a given graph.
///
/// The algorithm list all possible subsets of the vertices of the graph and check if each
/// subset is a vertex cover going from the smallest subset to the largest one.
///
/// This algorithm can be used on any graph with order < 65.
///
/// # Example
/// ```rust
/// use petgraph::prelude::UnGraphMap;
/// use vertex::naive_search;
/// use vertex::result_utils::Clock;
///
/// let mut graph = Box::new(UnGraphMap::<u64, ()>::new());
/// for i in 0..4 {
///    graph.add_node(i);
/// }
/// graph.add_edge(0, 1, ());
/// graph.add_edge(1, 2, ());
/// graph.add_edge(2, 0, ());
/// graph.add_edge(2, 3, ());
///
/// let expected_vertex_cover = 2; //[0, 2] or [1, 2]
/// assert_eq!(naive_search(&graph, &mut Clock::new(3600), None, None).0, expected_vertex_cover);
/// ```
pub fn naive_search(graph: &UnGraphMap<u64, ()>, clock: &mut Clock, _params: Option<&[f64]>, _optimal: Option<u64>) -> (u64, Vec<u64>) {
    let possible_values: Vec<u64> = (0..graph.node_count() as u64).collect();
    for i in 0..graph.node_count() {
        for t in possible_values.iter().combinations(i) {
            if clock.is_time_up() {
                return (0, Vec::new());
            }
            let subset: Vec<u64> = itertools::cloned(t).collect();


            if is_vertex_cover(graph, &subset) {
                return (subset.len() as u64, subset);
            }
        }
    }
    (0, Vec::new())
}

/// Run a given algorithm on a given graph and print the result.
///
/// It is the default function when you want to test your algorithm on a certain graph.
/// It prints the result and tell you if it is optimal or not based on the data in the yaml file.
/// The algorithm must take an UnGraphMap as input and returns u64.
///
/// # Example
/// ```rust
/// use vertex::graph_utils::load_clq_file;
/// use vertex::{naive_search, run_algorithm};
///
/// let mut graph = load_clq_file("src/resources/graphs/test.clq").unwrap();
/// // Naive search does not need parameters : run it with None
/// let res = run_algorithm("test.clq", &graph, &naive_search, None , 3600, false, false).unwrap_or_else(|e| {
///    panic!("Error while running algorithm : {}", e);
/// });
/// println!("{}", res);
/// ```
pub fn run_algorithm(graph_id: &str,
                     graph: &UnGraphMap<u64, ()>,
                     f: &Algorithm,
                     params: Option<&[f64]>,
                     time_limit: u64,
                     cmpl: bool,
                     is_clq: bool) -> Result<MVCResult, YamlError> {
    let g: UnGraphMap<u64, ()>;
    if cmpl {
        g = graph_utils::complement(graph);
        let density = (2 * g.edge_count()) as f64 / (g.node_count() * (g.node_count() - 1)) as f64;
        println!("Running algorithm the complement of the graph. Order = {} and size = {}. Density = {}",
                 g.node_count(),
                 g.edge_count(),
                 density);
    } else {
        g = copy_graph(graph);
        let density = (2 * g.edge_count()) as f64 / (g.node_count() * (g.node_count() - 1)) as f64;
        println!("Running algorithm on the graph. Order = {} and size = {}, density = {}",
                 graph.node_count(),
                 graph.edge_count(),
                 density);
    }

    let mut clock: Clock = Clock::new(time_limit);

    let optimal = if cmpl {
        if is_clq {
            get_optimal_value(graph_id, Some("src/resources/clique_data.yml"))
        } else {
            get_optimal_value(graph_id, Some("src/resources/compl_data.yml"))
        }
    } else {
        get_optimal_value(graph_id, None)
    };

    let res = f(&g, &mut clock, params, optimal.unwrap_or(None));

    let elapsed = clock.get_time();
    if !clock.is_time_up() {
        assert!(is_vertex_cover(&g, &res.1));
        assert_eq!(res.0, res.1.len() as u64);
    }

    MVCResult::new(graph_id.to_string(), res.0, res.1, elapsed, clock.is_time_up(), cmpl, is_clq)
}

/// Branch and bound algorithm that searches for the minimum vertex cover of a given graph.
///
/// * Branch : The algorithm branches on the vertex with max degree.
/// It creates 2 branches : one where the vertex is in the vertex cover and one where its neighbors are in the vertex cover.
/// * Bound : The algorithm has 2 lower bounds : clqLB and degLB. (see the paper linked in README for more details)
///
/// The clock is used to stop the algorithm if it reaches the time limit defined in the clock.
/// It is also used to measure the time taken by the algorithm (and some of its subroutines).
///
/// # Example
/// ```rust
/// use petgraph::prelude::UnGraphMap;
/// use vertex::branch_and_bound;
/// use vertex::graph_utils::load_clq_file;
/// use vertex::result_utils::Clock;
///
/// let graph = load_clq_file("src/resources/graphs/test.clq")
///             .expect("Error while loading graph");
/// let mut clock = Clock::new(3600); // 1 hour time limit
///
/// let res = branch_and_bound(&graph, &mut clock, None, None);
///
/// assert_eq!(res.0, 3);
/// assert_eq!(res.1, vec![0, 4, 2]);
/// ```
///
pub fn branch_and_bound(graph: &UnGraphMap<u64, ()>, clock: &mut Clock, _params: Option<&[f64]>, optimal: Option<u64>) -> (u64, Vec<u64>) {
    // Initialize the upper bound to the number of nodes in the graph
    // and the vertex cover found so far is empty
    let upper_bound_vc = &graph.nodes().collect();
    let u = b_and_b(graph, graph, graph.node_count() as u64,
                    upper_bound_vc, vec![], clock);

    assert!(is_vertex_cover(graph, &u.1));
    if optimal.is_some() {
        //assert_eq!(u.0, optimal.unwrap());
    }
    u
}

/// Local search based algorithm that search for the minimum vertex cover of a given graph.
///
/// This algorithm comes from an article by Cai et al. (2013)
///
/// # Parameters
/// - `graph`: The graph on which we want to compute the vertex cover
/// - `clock`: The clock used to measure the time taken by the algorithm and stop it if it reaches the time limit
/// - `params`: The parameters of the algorithm. It is a vector of 2 elements. The first element is the threshold and the second element is rho.
/// - `optimal`: The optimal value of the minimum vertex cover. It is used to stop the algorithm if it finds a vertex cover with this value.
///
/// # Example
/// ```rust
/// use petgraph::prelude::UnGraphMap;
/// use vertex::numvc;
/// use vertex::graph_utils::load_clq_file;
/// use vertex::result_utils::Clock;
///
///
/// let graph = load_clq_file("src/resources/graphs/test.clq")
///             .expect("Error while loading the graph");
/// let mut clock = Clock::new(3600); // 1 hour time limit
/// let res = numvc(&graph, &mut clock, Some(&vec![0.0, 0.0]), Some(3));
///
/// assert_eq!(res.0, 3);
/// assert_eq!(res.1, vec![0, 4, 3]);
/// ```
pub fn numvc(graph: &UnGraphMap<u64, ()>, clock: &mut Clock, params: Option<&[f64]>, optimal: Option<u64>) -> (u64, Vec<u64>) {
    // ======== Default parameters ========
    let mut threshold = 0.5 * graph.node_count() as f64;
    let mut rho = 0.3;
    // ======== Parameters ========
    if params.is_some() && params.unwrap().len() == 2 {
        threshold = params.unwrap()[0];
        rho = params.unwrap()[1];
    }
    // ======== Algorithm ========
    let mut g = add_weight_to_graph(graph, 1);
    clock.restart();
    let res = numvc_algorithm(&mut g, clock, threshold, rho, optimal);
    clock.stop_timer();

    assert!(is_vertex_cover(graph, &res));
    (res.len() as u64, res)
}

/// Simulated Annealing based algorithm that search for the minimum vertex cover of a given graph.
///
/// This algorithm use 3 parameters that have default values : 
/// * initial_temp = 50.0
/// * final_temp = 0.01
/// * cooling_rate = 0.95
///
/// # Parameters
/// - `graph`: The graph on which we want to compute the vertex cover
/// - `clock`: The clock used to measure the time taken by the algorithm and stop it if it reaches the time limit
/// - `params`: The parameters of the algorithm. It is a vector of 3 elements. 
/// The first element is the initial temperature, the second element is the final temperature and the third element is the cooling rate.
/// - `optimal`: The optimal value of the minimum vertex cover. It is used to stop the algorithm if it finds a vertex cover with this value.
///
/// # Example
/// ```rust
/// use petgraph::prelude::UnGraphMap;
/// use vertex::samvc;
/// use vertex::graph_utils::load_clq_file;
/// use vertex::result_utils::Clock;
///
/// let graph = load_clq_file("src/resources/graphs/test.clq")
///       .expect("Error while loading the graph");
/// let mut clock = Clock::new(3600); // 1 hour time limit
/// let res = samvc(&graph, &mut clock, Some(&vec![50.0, 0.01, 0.95]), Some(3));
///
/// assert_eq!(res.0, 3);
/// assert_eq!(res.1, vec![0, 4, 3]);
/// ```
pub fn samvc(graph: &UnGraphMap<u64, ()>, clock: &mut Clock, params: Option<&[f64]>, optimal: Option<u64>) -> (u64, Vec<u64>) {
    // ======== Default parameters ========
    let mut initial_temp = 100.0;
    let mut final_temp = 0.01;
    let mut cooling_rate = 0.995;
    // ======== Parameters ========
    if params.is_some() && params.unwrap().len() == 3 {
        initial_temp = params.unwrap()[0];
        final_temp = params.unwrap()[1];
        cooling_rate = params.unwrap()[2];
    }
    let max_call = 10;
    // Test : Run the algorithm until 10 call without improvement
    let mut best_solution = Vec::new();

    let mut call = 0;
    clock.restart();
    while call < max_call {
        // Call the algorithm max_call time without improvement
        
        let res = if best_solution.is_empty() {
            samvc::samvc_algorithm(graph, clock, optimal, final_temp, initial_temp, cooling_rate, None)
        } else {
            samvc::samvc_algorithm(graph, clock, optimal, final_temp, initial_temp, cooling_rate, Some(best_solution.clone()))
        };
        

        if is_vertex_cover(graph, &res) {
            // If we found a better solution, we replace it and reset the call counter
            if best_solution.is_empty() || res.len() < best_solution.len() {
                best_solution = res;
                call = 0;
                
                if optimal.is_some() && best_solution.len() as u64 == optimal.unwrap() {
                    break;
                }
            } else {
                // We did not find a better solution, we increment the call counter
                call += 1;
            }
        }
    }
    clock.stop_timer();

    (best_solution.len() as u64, best_solution)
}

#[cfg(test)]
mod algorithms_tests {
    use super::*;

    #[test]
    fn test_naive_algorithm() {
        let mut graph = Box::new(UnGraphMap::<u64, ()>::new());
        for i in 0..4 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, ());
        graph.add_edge(1, 2, ());
        graph.add_edge(2, 0, ());
        graph.add_edge(2, 3, ());

        let expected_vertex_cover = 2;
        assert_eq!(naive_search(&graph, &mut Clock::new(3600), None, None).0, expected_vertex_cover);
    }

    #[test]
    fn test_naive_on_empty_graph() {
        let graph = Box::new(UnGraphMap::<u64, ()>::new());
        assert_eq!(naive_search(&graph, &mut Clock::new(3600), None, None).0, 0);
    }

    #[test]
    fn test_naive_time_up() {
        let mut graph = Box::new(UnGraphMap::<u64, ()>::new());
        for i in 0..4 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, ());
        graph.add_edge(1, 2, ());
        graph.add_edge(2, 0, ());
        graph.add_edge(2, 3, ());

        let clock = &mut Clock::new(0);
        assert_eq!(naive_search(&graph, clock, None, None).0, 0);
        assert!(clock.is_time_up());
    }
}