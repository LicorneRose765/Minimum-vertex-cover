use std::collections::HashMap;
use std::fmt::Display;
use std::ops::Add;
use std::time::Duration;

use itertools::Itertools;
use petgraph::prelude::UnGraphMap;
use serde::{Deserialize, Serialize};

use crate::branch_and_bound::b_and_b;
use crate::errors::{ClockError, YamlError};
use crate::graph_utils::{copy_graph, get_optimal_value, is_optimal_value, is_vertex_cover};
use crate::numvc::{add_weight_to_graph, numvc_algorithm};

pub mod graph_utils;
pub mod mvcgraph;
pub mod errors;
mod numvc;
mod branch_and_bound;


/// Type alias for an algorithm that takes an UnGraphMap as input and computes the minimum vertex cover of the graph.
type Algorithm = dyn Fn(&UnGraphMap<u64, ()>, &mut Clock, Option<&[f64]>, Option<u64>) -> (u64, Vec<u64>);

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
/// use vertex::{Clock, naive_search};
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
/// use vertex::{Clock, branch_and_bound};
/// use vertex::graph_utils::load_clq_file;
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
/// You can set it to a negative value if you don't want to use it and let the algorithm run until it reaches the time limit.
///
/// # Example
/// ```rust
/// use petgraph::prelude::UnGraphMap;
/// use vertex::{Clock, numvc};
/// use vertex::graph_utils::load_clq_file;
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
    let mut threshold = 0.0;
    let mut rho = 0.0;
    if params.is_some() && params.unwrap().len() == 2 {
        threshold = params.unwrap()[0];
        rho = params.unwrap()[1];
    }
    let mut g = add_weight_to_graph(graph, 1);
    clock.restart();
    let res = numvc_algorithm(&mut g, clock, threshold, rho, optimal);
    clock.stop_timer();

    assert!(is_vertex_cover(graph, &res));

    (res.len() as u64, res)
}

/// Struct representing the time taken by an algorithm (in minutes, seconds, milliseconds and microseconds)
///
/// # Example
/// ```rust
/// use std::time::Duration;
/// use vertex::ElapseTime;
///
/// let duration = Duration::new(190, 1001000); // 3 mins 10 second 1 ms 1 microsecond
///
/// let elapsed = ElapseTime::new(duration);
///
/// assert_eq!(elapsed.min, 3);
/// assert_eq!(elapsed.sec, 10);
/// assert_eq!(elapsed.ms, 1);
/// assert_eq!(elapsed.micro, 1);
/// ```
#[derive(Debug, PartialEq, Deserialize, Serialize)]
pub struct ElapseTime {
    pub duration: Duration,
    pub min: u128,
    pub sec: u128,
    pub ms: u128,
    pub micro: u128,
}

impl ElapseTime {
    pub fn new(duration: Duration) -> ElapseTime {
        let elapsed = duration.as_micros();
        let min = elapsed / 60_000_000;
        let sec = (elapsed / 1_000_000) % 60;
        let ms = (elapsed / 1_000) % 1_000;
        let micro = elapsed % 1_000;
        ElapseTime {
            duration,
            min,
            sec,
            ms,
            micro,
        }
    }
}

impl Display for ElapseTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}min {}s {}ms {}µs", self.min, self.sec, self.ms, self.micro)
    }
}

/// Struct representing the result of an algorithm
pub struct MVCResult {
    /// The id of the graph. Example : "test.clq"
    pub graph_id: String,
    /// The value of the minimum vertex cover calculated by the algorithm
    pub value: u64,
    /// The set of vertices that form the minimum vertex cover
    pub set: Vec<u64>,
    /// Whether the value is optimal or not. (Found in the clique and graph data yaml files)
    pub is_optimal: Option<bool>,
    /// The time taken by the algorithm
    pub time: ElapseTime,
    /// Whether the algorithm was stopped because it reached the time limit
    pub is_time_limit: bool,
    /// Whether the algorithm was run on the complement of the graph
    pub is_compl: bool,
    /// Whether the value is the value of the maximum clique of the graph
    pub is_clq: bool,
}

impl MVCResult {
    pub fn new(graph_id: String, value: u64, mvc: Vec<u64>, time: ElapseTime, is_time_limit: bool, is_compl: bool, is_clq: bool) -> Result<MVCResult, YamlError> {
        let is_optimal = if is_compl {
            if is_clq {
                is_optimal_value(&graph_id, value, Some("src/resources/clique_data.yml"))?
            } else {
                is_optimal_value(&graph_id, value, Some("src/resources/compl_data.yml"))?
            }
        } else {
            is_optimal_value(&graph_id, value, None)?
        };
        Ok(MVCResult {
            graph_id,
            value,
            set: mvc,
            is_optimal,
            time,
            is_time_limit,
            is_compl,
            is_clq,
        })
    }
}

impl Display for MVCResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let opt_message = {
            if self.is_optimal.is_some() {
                if self.is_optimal.unwrap() {
                    "\t The value is optimal (as long as the data is correct in the yaml file)".to_string()
                } else {
                    let true_opt = if self.is_compl {
                        if self.is_clq {
                            get_optimal_value(&self.graph_id, Some("src/resources/clique_data.yml")).unwrap_or(Some(0))
                        } else {
                            get_optimal_value(&self.graph_id, Some("src/resources/compl_data.yml")).unwrap_or(Some(0))
                        }
                    } else {
                        get_optimal_value(&self.graph_id, None).unwrap_or(Some(0))
                    };
                    format!("\t The value is not optimal and the correct value is {}", true_opt.unwrap_or(0)).to_string()
                }
            } else {
                "\t The graph is not in the yaml file".to_string()
            }
        };

        let time_limit_message = {
            if self.is_time_limit {
                "\n\t The algorithm was stopped because it reached the time limit".to_string()
            } else {
                "".to_string()
            }
        };

        write!(f, "Minimum vertex cover for the {:?} graph = {}\n{}\n\t Time taken by the algorithm : {} {}",
               self.graph_id,
               self.value,
               opt_message,
               self.time,
               time_limit_message)
    }
}


/// Struct representing a clock used to measure the time taken by an algorithm and stop it if it reaches the time limit.
///
/// This clock is based on the std::time::Instant struct.
/// The clock can also be used to measure the time taken by some subroutines of the algorithm.
///
/// # Example
/// ```rust
/// use std::time::Duration;
/// use vertex::Clock;
///
/// let mut clock = Clock::new(3600); // 1 hour time limit
///
/// let elapsed = clock.get_time();
///
/// clock.enter_subroutine("subroutine1");
/// // Do something
/// clock.exit_subroutine("subroutine1").expect("The subroutine was not entered before");
/// clock.enter_subroutine("subroutine2");
/// // Do something
/// clock.exit_subroutine("subroutine2").expect("The subroutine was not entered before");
///
/// if clock.is_time_up() {
///    println!("Time is up !");
/// }
/// println!("Time taken by the algorithm : {}", elapsed);
/// println!("Time taken by subroutine1 : {}", clock.get_subroutine_duration("subroutine1").as_millis());
///
///
///```
pub struct Clock {
    pub start: std::time::Instant,
    limit: u64,
    elapsed: Option<Duration>,

    // Hashmap containing the time taken by each subroutine of the algorithm.
    // Key : name of the subroutine
    // Value : (start time, time taken)
    details: HashMap<String, (Option<std::time::Instant>, Duration)>,
}
impl Clock {
    pub fn new(limit: u64) -> Clock {
        Clock {
            start: std::time::Instant::now(),
            limit,
            elapsed: None,
            details: HashMap::new(),
        }
    }

    /// Restarts the clock. (Sets the start time to now)
    pub fn restart(&mut self) {
        self.start = std::time::Instant::now();
    }

    /// Returns the time off the clock since it was created.
    pub fn get_time(&self) -> ElapseTime {
        if self.elapsed.is_none() {
            let elapsed = self.start.elapsed();
            ElapseTime::new(elapsed)
        } else {
            ElapseTime::new(self.elapsed.unwrap())
        }
    }

    /// Stops the clock.
    pub fn stop_timer(&mut self) {
        if self.elapsed.is_none() {
            self.elapsed = Some(self.start.elapsed());
        }
    }

    /// Returns true if the time limit is reached.
    pub fn is_time_up(&self) -> bool {
        let elapsed = self.start.elapsed();
        elapsed.as_secs() >= self.limit
    }

    /// Enters a subroutine of the algorithm and start the timer for this subroutine.
    /// It creates a new start time for this subroutine but don't reset the duration.
    ///
    /// If the subroutine was already entered before, it will reset the start time and add the time taken since the last time it was entered.
    /// # Example
    /// ```rust
    /// use std::time::Duration;
    /// use vertex::Clock;
    ///
    /// let mut clock = Clock::new(3600);
    ///
    /// clock.enter_subroutine("subroutine1");
    /// // Do something
    /// clock.enter_subroutine("subroutine2");
    /// // Do something
    /// clock.exit_subroutine("subroutine2").expect("The subroutine was not entered before");
    ///
    /// clock.enter_subroutine("subroutine1");
    /// // Add the time taken since the last time we entered subroutine1
    /// clock.exit_subroutine("subroutine1").expect("The subroutine was not entered before");
    /// ```
    pub fn enter_subroutine(&mut self, name: &str) {
        if self.details.contains_key(name) {
            // Keep the duration but change the start time
            let (time, duration) = self.details.get(name).unwrap();
            if time.is_none() {
                self.details.insert(name.to_string(), (Some(std::time::Instant::now()), *duration));
            } else {
                let new_duration = duration.add(time.unwrap().elapsed());
                self.details.insert(name.to_string(), (Some(std::time::Instant::now()), new_duration));
            }
        } else {
            self.details.insert(name.to_string(), (Some(std::time::Instant::now()), Duration::new(0, 0)));
        }
    }

    /// Exits a subroutine of the algorithm and add the time taken since the last time it was entered.
    /// If the subroutine was already exit before, it does nothing.
    ///
    /// # Throws
    /// ClockError if the subroutine was not entered before.
    pub fn exit_subroutine(&mut self, name: &str) -> Result<(), ClockError>{
        let (start, duration) = match self.details.get(name) {
            Some((start, duration)) => (start, duration),
            None => return Err(ClockError::new("The subroutine was not entered before")),
        };
        if !start.is_none() {
            // If the subroutine is not exit, we exit it
            let elapsed = start.unwrap().elapsed();
            self.details.insert(name.to_string(), (None, *duration + elapsed));
        }
        Ok(())
    }

    /// Returns the time taken by a subroutine of the algorithm.
    ///
    /// The time taken is the sum of all the time taken by this subroutine since the first time it was entered.
    /// If the subroutine was not entered before, it a duration of 0.
    ///
    /// # Example
    /// ```rust
    /// use std::time::Duration;
    /// use vertex::{Clock, ElapseTime};
    ///
    /// let mut clock = Clock::new(3600);
    ///
    /// clock.enter_subroutine("subroutine1");
    /// // Do something
    /// clock.exit_subroutine("subroutine1").expect("The subroutine was not entered before");
    ///
    /// let elapsed = clock.get_subroutine_duration("subroutine1");
    /// println!("Time taken by subroutine1 : {}", ElapseTime::new(elapsed));
    /// println!("Percentage of time taken by subroutine1 : {}%", elapsed.as_secs_f64() * 100.0 / clock.get_time().duration.as_secs_f64());
    /// ```
    pub fn get_subroutine_duration(&self, name: &str) -> Duration {
        if self.details.contains_key(name) {
            let (_, duration) = self.details.get(name).unwrap();
            *duration
        } else {
            Duration::new(0, 0)
        }
    }
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