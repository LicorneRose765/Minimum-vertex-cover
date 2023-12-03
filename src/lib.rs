extern crate graph;

use std::fmt::Display;
use std::time::Duration;

use petgraph::prelude::UnGraphMap;
use serde::{Deserialize, Serialize};

use crate::graph_utils::{copy_graph, get_optimal_value, is_optimal_value, is_vertex_cover};

pub mod graph_utils;
pub mod format;
pub mod branch_and_bound;


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

pub struct Result {
    pub graph_id: String,
    pub value: u64,
    pub set: Vec<u64>,
    pub is_optimal: Option<bool>,
    pub time: ElapseTime,
    pub is_time_limit: bool,
}

impl Result {
    pub fn new(graph_id: String, value: u64, mvc: Vec<u64>, time: ElapseTime, is_time_limit: bool) -> Result {
        let is_optimal = is_optimal_value(&graph_id, value, None);
        Result {
            graph_id,
            value,
            set: mvc,
            is_optimal,
            time,
            is_time_limit,
        }
    }
}

impl Display for Result {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let opt_message = {
            if !self.is_optimal.is_none() {
                if self.is_optimal.unwrap() {
                    "\t The value is optimal (as long as the data is correct in the yaml file)".to_string()
                } else {
                    let true_opt = get_optimal_value(&self.graph_id, None).unwrap_or(0);
                    format!("\t The value is not optimal and the correct value is {}", true_opt).to_string()
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

pub struct Clock {
    pub start: std::time::Instant,
    pub limit: u64, // in seconds
}

impl Clock {
    pub fn new(limit: u64) -> Clock {
        Clock {
            start: std::time::Instant::now(),
            limit,
        }
    }

    pub fn get_time(&self) -> ElapseTime {
        let elapsed = self.start.elapsed();
        ElapseTime::new(elapsed)
    }

    pub fn is_time_up(&self) -> bool {
        let elapsed = self.start.elapsed();
        elapsed.as_secs() >= self.limit
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
/// use vertex::{Clock, naive_search};
///
/// let mut graph = UnGraphMap::<u64, ()>::new();
/// for i in 0..4 {
///    graph.add_node(i);
/// }
/// graph.add_edge(0, 1, ());
/// graph.add_edge(1, 2, ());
/// graph.add_edge(2, 0, ());
/// graph.add_edge(2, 3, ());
///
/// let expected_vertex_cover = 2; //[0, 2] or [1, 2]
/// assert_eq!(naive_search(&graph, &Clock::new(3600)).0, expected_vertex_cover);
/// ```
pub fn naive_search(graph: &UnGraphMap<u64, ()>, clock: &Clock) -> (u64, Vec<u64>) {
    if graph.node_count() > 64 {
        panic!("This algorithm can only be used on graph with less than 65 vertices")
    }
    let possible_values: Vec<u64> = (0..graph.node_count() as u64).collect();
    let mut found = false;
    let mut res = 0;
    let mut res_subset: Vec<u64> = Vec::new();
    for subset in get_subsets(&possible_values) {
        if clock.is_time_up() {
            break;
        }
        if !found || res > subset.len() as u64 {
            if is_vertex_cover(graph, &subset) {
                res = subset.len() as u64;
                res_subset = subset;
                found = true;
            }
        }
    }
    (res, res_subset)
}

/// Run a given algorithm on a given graph and print the result. It is the default function when you want
/// to test your algorithm on a certain graph. It prints the result and tell you if it is optimal or not based
/// on the data in the yaml file.
/// The algorithm must take an UnGraphMap as input and return a u64.
///
/// # Example
/// ```rust
/// use petgraph::prelude::UnGraphMap;use vertex::graph_utils::load_clq_file;
/// use vertex::{naive_search, run_algorithm};
///
/// let mut graph = load_clq_file("src/resources/graphs/test.clq").unwrap();
/// let res = run_algorithm("test.clq", &graph, &naive_search, false);
/// println!("{}", res);
/// ```
pub fn run_algorithm(graph_id: &str,
                     graph: &UnGraphMap<u64, ()>,
                     f: &dyn Fn(&UnGraphMap<u64, ()>, &Clock) -> (u64, Vec<u64>),
                     cmpl: bool) -> Result {
    let g: UnGraphMap<u64, ()>;
    if cmpl {
        g = graph_utils::complement(graph);
        println!("Running algorithm the complement of the graph. Order = {} and size = {}",
                 g.node_count(),
                 g.edge_count());
    } else {
        println!("Running algorithm on the graph. Order = {} and size = {}",
                 graph.node_count(),
                 graph.edge_count());
        g = copy_graph(graph);
    }

    let limit = 3600;

    let clock: Clock = Clock::new(limit);

    let res = f(&g, &clock);

    let elapsed = clock.get_time();

    assert!(is_vertex_cover(&g, &res.1));
    assert_eq!(res.0, res.1.len() as u64);

    let res = Result::new(graph_id.to_string(), res.0, res.1, elapsed, clock.is_time_up());
    return res;
}


pub struct SubsetIterator<T> where T: Clone {
    pub set: Vec<T>,
    pub n: usize,
    pub n_times: usize,
    pub i: u64,
}

impl<T> Iterator for SubsetIterator<T> where T: Clone {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= (1 << self.n) as u64 {
            return None;
        }
        let mut subset: Vec<T> = Vec::new();
        for j in 0..self.n {
            if (self.i & (1 << j)) != 0 {
                subset.push(self.set[j].clone());
            }
        }
        self.i += 1;
        Some(subset)
    }
}


/// Returns an iterator over all the subsets of a given set. (The size of the set is capped at around 100)
fn get_subsets<T>(s: &[T]) -> SubsetIterator<T> where T: Clone {
    let n = s.len();

    SubsetIterator {
        set: s.to_vec(),
        n,
        n_times: 1 << n,
        i: 1,
    }
}

#[cfg(test)]
mod algorithms_tests {
    use super::*;

    #[test]
    fn test_naive_algorithm() {
        let mut graph = UnGraphMap::<u64, ()>::new();
        for i in 0..4 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, ());
        graph.add_edge(1, 2, ());
        graph.add_edge(2, 0, ());
        graph.add_edge(2, 3, ());

        let expected_vertex_cover = 2;
        assert_eq!(naive_search(&graph, &Clock::new(3600)).0, expected_vertex_cover);
    }

    #[test]
    fn test_get_subset() {
        let initial_set = vec![1, 2, 3];
        let expected_subset = vec![
            vec![],
            vec![1],
            vec![2],
            vec![3],
            vec![1, 2],
            vec![1, 3],
            vec![2, 3],
            vec![1, 2, 3],
        ];
        let expected_subset: Box<Vec<Vec<u64>>> = Box::new(expected_subset);
        let out = get_subsets(&initial_set);
        for val in out {
            assert!(expected_subset.contains(&val));
        }
    }
}