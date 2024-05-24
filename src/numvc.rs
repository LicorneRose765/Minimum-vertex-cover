use petgraph::prelude::UnGraphMap;
use rand::rngs::ThreadRng;
use rand::seq::IteratorRandom;

use crate::Clock;

// To use it here.
#[allow(dead_code)]
pub fn numvc_algorithm(
    graph: &mut UnGraphMap<u64, i32>,
    clock: &mut Clock,
    mut threshold: f64,
    mut rho: f64,
    optimal: Option<u64>,
) -> Vec<u64> {
    println!("Starting NuMVC algorithm");
    
    // Initialize random number generator
    let mut rng = rand::thread_rng();
    // Params
    let mut iter = 1; // Number of iterations
    if threshold == 0.0 {
        // Default value of threshold is 0.5 * |V|
        threshold = 0.5 * graph.node_count() as f64;
    }
    if rho == 0.0 {
        rho = 0.3;
    }
    // Initialize dscores of vertices
    let mut dscores: Vec<i32> = vec![0; graph.node_count()];
    let mut solution: Vec<u64> = vec![];
    let mut solution_age: Vec<i32> = vec![0; graph.node_count()];
    // Time since the vertex was added/removed from the solution.
    // Added : positive value, Removed : negative value

    update_dscores(graph, &mut solution_age, &mut dscores);

    // Initialize confchange array (CC strategy)
    let mut confchange: Vec<u64> = vec![1; graph.node_count()];

    // Construct the first solution greedily by selecting the vertex with the highest dscore
    compute_greedy_vc(graph, &mut solution, &mut solution_age, &iter, &mut dscores);
    let mut best_solution = solution.clone();

    // Main loop
    while !clock.is_time_up() {
        if iter % 1000 == 0 {
            println!("Iteration {}, best solution : {}", iter, best_solution.len());
        }
        if is_vertex_cover_with_weight(graph, &solution_age) {
            best_solution.clone_from(&solution);
            if optimal.is_some() && solution.len() <= optimal.unwrap() as usize {
                // We found the optimal solution
                println!("Found optimal solution with {} iterations", iter);
                break;
            }

            // Remove the worst vertex from the solution
            let v = get_vertex_with_highest_dscore_from_solution(&dscores, &solution, &solution_age);
            remove(graph, &mut solution, &mut solution_age, &mut dscores, &mut confchange, v, iter);

            continue
        }
        // Select a vertex from c with the highest dscore
        let u = get_vertex_with_highest_dscore_from_solution(&dscores, &solution, &solution_age);
        /* C := C\{u}, confChange(u) := 0 and confChange(z) := 1 for each z in
       * N(u); */
        remove(graph, &mut solution, &mut solution_age, &mut dscores, &mut confchange, u, iter);


        // Select a vertex to add
        let v = pick_new_vertex(graph, &dscores, &confchange, &solution_age, &mut rng);

        /* C := C plus {v}, confChange(z) := 1 for each z in N(v); */
        add(graph, &mut solution, &mut solution_age, &mut dscores, &mut confchange, v, iter);

        /* w(e) := w(e) + 1 for each uncovered edge e; */
        update_weights(graph, &solution_age, &mut dscores);

        /* if w >= y then w(e) := [p*w(e)] for each edge e; */
        let mean = compute_mean_weight(graph);
        if mean >= threshold {
            reduce_weights(graph, rho, &mut dscores, &solution_age);
        }

        iter += 1;
    }
    println!("Number of iterations : {}", iter);
    best_solution
}

fn remove(
    graph: &UnGraphMap<u64, i32>,
    solution: &mut Vec<u64>,
    solution_age: &mut [i32],
    dscores: &mut [i32],
    confchange: &mut [u64],
    vertex: u64,
    iter: i32,
) {
    solution.retain(|&x| x != vertex);
    solution_age[vertex as usize] = -iter;
    confchange[vertex as usize] = 0;
    dscores[vertex as usize] *= -1;

    for i in graph.neighbors(vertex) {
        if solution_age[i as usize] <= 0 {
            // If the vertex is not in the solution
            confchange[i as usize] = 1;
            dscores[i as usize] += graph.edge_weight(vertex, i).unwrap();
        } else {
            dscores[i as usize] -= graph.edge_weight(vertex, i).unwrap();
        }
    }
}

fn add(
    graph: &UnGraphMap<u64, i32>,
    solution: &mut Vec<u64>,
    solution_age: &mut [i32],
    dscores: &mut [i32],
    confchange: &mut [u64],
    vertex: u64,
    iter: i32,
) {
    solution.push(vertex);
    solution_age[vertex as usize] = iter;
    dscores[vertex as usize] *= -1;

    for i in graph.neighbors(vertex) {
        if solution_age[i as usize] <= 0 {
            // If the vertex is not in the solution
            confchange[i as usize] = 1;
            dscores[i as usize] -= graph.edge_weight(vertex, i).unwrap();
        } else {
            dscores[i as usize] += graph.edge_weight(vertex, i).unwrap();
        }
    }
}

/// Add 1 to the weight of all edges that are covered by the solution (in place)
///
/// # Parameters
/// - `graph`: The graph on which the vertex cover is computed
/// - `solution_age`: The array containing the age of the vertices in the solution. (The vertices that are not in the solution have an age of 0)
fn update_weights(
    graph: &mut UnGraphMap<u64, i32>,
    solution_age: &[i32],
    dscores: &mut [i32]
) {
    for edge in graph.all_edges_mut() {
        if solution_age[edge.0 as usize] <= 0 && solution_age[edge.1 as usize] <= 0 {
            // If solution_old > 0, it means that the vertex is currently in the solution
            dscores[edge.0 as usize] += 1;
            dscores[edge.1 as usize] += 1;
            *edge.2 += 1;
        }
    }
}

/// Get the cost of a solution
///
/// The cost of a solution X is computed as follows:
/// ` cost(G, X) = SUM_{e in E and e not covered by X} w(e) `
/// where w(e) is the weight of edge e
///
/// # Parameters
/// - `graph`: The graph on which the vertex cover is computed
/// - `solution`: The solution for which the cost is computed
fn get_cost(
    graph: &UnGraphMap<u64, i32>,
    solution_age: &[i32]
) -> i32 {
    let mut cost = 0;
    for (a, b, w) in graph.all_edges() {
        if solution_age[a as usize] <= 0 && solution_age[b as usize] <= 0 {
            cost += w;
        }
    }
    cost
}

/// Compute the dscore of a vertex in a current solution C.
///
/// The dscore of a vertex v in a current solution C is computed as follows:
/// ` dscore(v) = cost(v, C) - cost(v, C') `
/// Where c' is the solution C with vertex v removed or added. (depends on if v is in C or not)
///
/// # Parameters
/// - `graph`: The graph on which the vertex cover is computed
/// - `solution`: The current solution
/// - `vertex`: The vertex for which the dscore is computed
fn compute_dscore(
    graph: &UnGraphMap<u64, i32>, // Use u64 as edge weights
    solution_age: &mut [i32],
    vertex: u64
) -> i32 {
    let c1 = get_cost(graph, solution_age);
    let c2;
    let age = solution_age[vertex as usize];
    if age > 0 {
        solution_age[vertex as usize] = 0; // Remove vertex from solution
        c2 = get_cost(graph, solution_age);
        solution_age[vertex as usize] = age; // Add vertex back to solution
    } else {
        solution_age[vertex as usize] = 1; // Add vertex to solution
        c2 = get_cost(graph, solution_age);
        solution_age[vertex as usize] = age; // Remove vertex from solution
    }
    c1 - c2
}

/// Update the dscores of all vertices in the graph. (In place)
///
/// # Parameters
/// - `graph`: The graph on which the vertex cover is computed
/// - `solution`: The current solution
/// - `dscores`: The dscores of the vertices
fn update_dscores(
    graph: &UnGraphMap<u64, i32>,
    solution_age: &mut [i32],
    dscores: &mut [i32]
) {
    for vertex in graph.nodes() {
        dscores[vertex as usize] = compute_dscore(graph, solution_age, vertex);
    }
}

/// Get the vertex with the highest dscore.
///
/// By definition of dscore, this vertex is not in the current solution.
///
/// # Parameters
/// - `dscores`: The dscores of the vertices
fn get_vertex_with_highest_dscore(dscores: &[i32]) -> u64 {
    if let Some((max_index, _)) = dscores.iter().enumerate().max_by_key(|&(_, &item)| item) {
        max_index as u64
    } else {
        panic!("Error in vertex cover computation (should not happen)")
    }
}

/// Returns the vertex with highest dscore from the solution.
///
/// If there is a tie, the oldest vertex is selected.
///
/// # Parameters
/// - `dscores`: The dscores of the vertices
/// - `solution`: The current solution
/// - `solution_age`: The age of the vertices in the solution.
/// Every vertex has an age. The vertices not in the solution have a negative age (-iter when it was removed).
///
/// # Returns
/// The vertex with the highest dscore from the solution.
fn get_vertex_with_highest_dscore_from_solution(
    dscores: &[i32],
    solution: &[u64],
    solution_age: &[i32]) -> u64 {
    let mut max_vertex = match solution.first() {
        Some(v) => *v,
        None => panic!("Trying to get vertex with highest dscore from an empty solution")
    };

    for vertex in solution.iter() {
        match dscores[*vertex as usize].cmp(&dscores[max_vertex as usize]) {
            std::cmp::Ordering::Greater => max_vertex = *vertex,
            std::cmp::Ordering::Equal => {
                if solution_age[*vertex as usize] > solution_age[max_vertex as usize] {
                    max_vertex = *vertex;
                }
            }
            std::cmp::Ordering::Less => {}
        }
    }
    max_vertex
}

/// This functions creates a greedy solution by iteratively selecting the vertex with the highest dscore
/// and adding it to the solution.
///
/// # Parameters
/// - `graph`: The graph on which the vertex cover is computed
/// - `solution`: The current solution
/// - `solution_age`: The age of the vertices in the solution.
/// - `iter`: The current iteration of the algorithm
/// - `dscores`: The dscores of the vertices
///
/// # Side effects
/// These arrays are modified in place:
/// - `solution`
/// - `solution_age`
/// - `dscores`
fn compute_greedy_vc(
    graph: &UnGraphMap<u64, i32>,
    solution: &mut Vec<u64>,
    solution_age: &mut [i32],
    iter: &i32,
    dscores: &mut [i32],
) {
    let mut confchange = vec![1; graph.node_count()];
    while !is_vertex_cover_with_weight(graph, solution_age) {
        let v = get_vertex_with_highest_dscore(dscores);
        add(graph, solution, solution_age, dscores, &mut confchange, v, *iter);
    }
}

/// Get an uncovered edge randomly from the graph.
///
/// # Parameters
/// - `graph`: The graph on which the vertex cover is computed
/// - `solution_age`: The age of the vertices in the solution. (Used to determine if an edge is covered or not)
/// The value in solution age are either positive if the vertex is in the solution or negative if it is not.
///
/// # Returns
/// An uncovered edge (i, j) from the graph.
///
/// # Panics
/// If there is no uncovered edge in the graph. (This can't happen in the context of the NuMVC algorithm)
fn get_one_uncovered_edge(graph: &UnGraphMap<u64, i32>, solution_age: &[i32], rng: &mut ThreadRng) -> (u64, u64) {
    let edges = graph.all_edges()
        .filter(|(i, j, _)| solution_age[*i as usize] <= 0 && solution_age[*j as usize] <= 0);

    // Select randomly an edge from the iterator
    let edge: (u64, u64, &i32) = edges.choose(rng).unwrap();

    (edge.0, edge.1)
}

/// Pick a new vertex to add to the solution.
///
/// The selection is done as follows:
/// - Select one uncovered edge (u, v) from the graph randomly.
/// - If confchange\[u] == 0, return v
/// - If confchange\[v] == 0, return u
/// - Otherwise, return the vertex with the highest dscore breaking ties by selecting the oldest vertex.
///
/// # Parameters
/// - `graph`: The graph on which the vertex cover is computed
/// - `dscores`: The dscores of the vertices
/// - `confchange`: The confchange array (CC-strategy)
/// - `solution_old`: The age of the vertices in the solution.
///
/// # Returns
/// The vertex to add to the solution.
fn pick_new_vertex(
    graph: &UnGraphMap<u64, i32>,
    dscores: &[i32],
    confchange: &[u64],
    solution_old: &[i32],
    rng: &mut ThreadRng
) -> u64 {
    let (u, v) = get_one_uncovered_edge(graph, solution_old, rng);

    if confchange[u as usize] == 0 {
        v
    } else if confchange[v as usize] == 0 {
        u
    } else {
        // Select vertex with highest dscore
        let mut max_vertex = u;
        match dscores[v as usize].cmp(&dscores[u as usize]) {
            std::cmp::Ordering::Greater => max_vertex = v,
            std::cmp::Ordering::Equal => {
                if solution_old[v as usize] > solution_old[u as usize] {
                    max_vertex = v;
                }
            }
            std::cmp::Ordering::Less => {}
        }
        max_vertex
    }
}

/// Used to compute the mean weight of the edges in the graph. To determine if we have to reduce them
fn compute_mean_weight(graph: &UnGraphMap<u64, i32>) -> f64 {
    let mut sum = 0;
    for (_, _, w) in graph.all_edges() {
        sum += w;
    }
    sum as f64 / graph.edge_count() as f64
}

/// Reduce the weights of the edges in the graph by a factor of rho. (in place)
fn reduce_weights(graph: &mut UnGraphMap<u64, i32>, rho: f64, dscores: &mut [i32], solution_age: &[i32]) {
    dscores.fill(0);

    for (a, b, w) in graph.all_edges_mut() {
        *w = (*w as f64 * rho).floor() as i32;

        // Update dscore
        if solution_age[a as usize] <= 0 && solution_age[b as usize] <= 0 {
            dscores[a as usize] += *w;
            dscores[b as usize] += *w;
        } else if a != b {
            if solution_age[a as usize] > 0 {
                dscores[a as usize] -= *w;
            } else {
                dscores[b as usize] -= *w;
            }
        }
    }
}

/// Check if the solution is a vertex cover of the graph.
///
/// A solution is a vertex cover if all edges are covered by the solution.
///
/// # Parameters
/// - `graph`: The graph on which the vertex cover is computed
/// - `solution_age`: The age of the vertices in the solution. (Used to determine if an edge is covered or not)
/// If the value in solution age is > 0, the vertex is in the solution. Otherwise, it is not.
///
/// # Returns
/// True if the solution is a vertex cover. False otherwise.
pub fn is_vertex_cover_with_weight(graph: &UnGraphMap<u64, i32>, solution_age: &[i32]) -> bool {
    for (i, j, _) in graph.all_edges() {
        if solution_age[i as usize] <= 0 && solution_age[j as usize] <= 0 {
            return false;
        }
    }
    true
}

/// Creates a copy of a graph and add weight to all of its edges.
///
/// # Parameters
/// - `graph`: The graph on which we want to add weights
/// - `base_value`: THe initial value of the weight.
///
/// # Returns
/// The graph with `base_value` on all of its edges as weight.
pub fn add_weight_to_graph(
    graph: &UnGraphMap<u64, ()>,
    base_value: i32
) -> UnGraphMap<u64, i32> {
    let mut res = UnGraphMap::<u64, i32>::new();
    for i in graph.nodes() {
        res.add_node(i);
    }
    for (i, j, _) in graph.all_edges() {
        res.add_edge(i, j, base_value);
    }
    res
}

#[cfg(test)]
mod numvc_tests {
    use petgraph::graphmap::UnGraphMap;

    use crate::graph_utils::{complement, load_clq_file};
    use crate::numvc;

    use super::*;

    #[test]
    fn test_numvc() {
        let graph = load_clq_file("src/resources/graphs/test.clq")
                    .expect("Error while loading the graph");
        let mut clock = Clock::new(3600); // 1 hour time limit
        let res = numvc(&graph, &mut clock, Some(&[0.0,0.0]), Some(3));
        assert_eq!(res.0, 3);
        assert_eq!(res.1, vec![0, 4, 3]);
    }

    #[test]
    fn test_numvc_with_direct_cutoff() {
        let graph = load_clq_file("src/resources/graphs/test.clq")
                    .expect("Error while loading the graph");
        let mut clock = Clock::new(0); // 0 seconds time limit
        let res = numvc(&graph, &mut clock, Some(&[0.0,0.0]), Some(3));
        assert_eq!(res.0, 3);
        assert_eq!(res.1, vec![0, 4, 3]);
    }

    #[test]
    fn test_numvc_c125_9() {
        let graph = load_clq_file("src/resources/graphs/C125.9.clq")
            .expect("Error while loading the graph");
        let mut clock = Clock::new(1); // 1 hour time limit
        let res = numvc(&complement(&graph), &mut clock, Some(&[0.0, 0.0]), Some(91));
        assert_eq!(res.0, 91);
    }

    #[test]
    fn test_add() {
        // house-shaped graph with weights = 1
        let mut graph = UnGraphMap::<u64, i32>::new();
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, 1);
        graph.add_edge(0, 2, 1);
        graph.add_edge(1, 3, 1);
        graph.add_edge(2, 4, 1);
        graph.add_edge(3, 4, 1);

        let mut solution_age = vec![0,0,0,0,0];
        let mut solution = vec![];
        let mut dscores = vec![2,2,2,2,2];
        let mut confchange = vec![1, 1, 1, 1, 1];
        add(&graph, &mut solution, &mut solution_age, &mut dscores, &mut confchange, 0, 1);
        assert_eq!(solution, vec![0]);
        assert_eq!(solution_age, vec![1, 0, 0, 0, 0]);
        assert_eq!(dscores, vec![-2, 1, 1, 2, 2]);
        assert_eq!(confchange, vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_add_with_solution() {
        // house-shaped graph with weights = 1
        let mut graph = UnGraphMap::<u64, i32>::new();
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, 1);
        graph.add_edge(0, 2, 1);
        graph.add_edge(1, 3, 1);
        graph.add_edge(2, 4, 1);
        graph.add_edge(3, 4, 1);

        let mut solution_age = vec![0,1,0,0,0];
        let mut solution = vec![1];
        let mut dscores = vec![1,-2,2,1,2];
        let mut confchange = vec![1, 1, 1, 1, 1];
        add(&graph, &mut solution, &mut solution_age, &mut dscores, &mut confchange, 0, 2);

        assert_eq!(solution, vec![1, 0]);
        assert_eq!(solution_age, vec![2, 1, 0, 0, 0]);
        assert_eq!(dscores, vec![-1, -1, 1, 1, 2]);
        assert_eq!(confchange, vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_remove() {
        // house-shaped graph with weights = 1
        let mut graph = UnGraphMap::<u64, i32>::new();
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, 1);
        graph.add_edge(0, 2, 1);
        graph.add_edge(1, 3, 1);
        graph.add_edge(2, 4, 1);
        graph.add_edge(3, 4, 1);

        let mut solution_age = vec![1, 0, 0, 0, 0];
        let mut solution = vec![0];
        let mut dscores = vec![-2, 1, 1, 2, 2];
        let mut confchange = vec![1, 1, 1, 1, 1];
        remove(&graph, &mut solution, &mut solution_age, &mut dscores, &mut confchange, 0, 1);

        assert_eq!(solution, vec![]);
        assert_eq!(solution_age, vec![-1, 0, 0, 0, 0]);
        assert_eq!(dscores, vec![2, 2, 2, 2, 2]);
        assert_eq!(confchange, vec![0, 1, 1, 1, 1]);
    }

    #[test]
    fn test_remove_with_solution() {
        // house-shaped graph with weights = 1
        let mut graph = UnGraphMap::<u64, i32>::new();
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, 1);
        graph.add_edge(0, 2, 1);
        graph.add_edge(1, 3, 1);
        graph.add_edge(2, 4, 1);
        graph.add_edge(3, 4, 1);

        let mut solution_age = vec![1, 1, 0, 0, 0];
        let mut solution = vec![0,1];
        let mut dscores = vec![-1, -1, 1, 1, 2];
        let mut confchange = vec![1, 1, 1, 1, 1];
        remove(&graph, &mut solution, &mut solution_age, &mut dscores, &mut confchange, 0, 2);

        assert_eq!(solution, vec![1]);
        assert_eq!(solution_age, vec![-2, 1, 0, 0, 0]);
        assert_eq!(dscores, vec![1,-2,2,1,2]);
        assert_eq!(confchange, vec![0, 1, 1, 1, 1]);
    }

    #[test]
    fn test_update_weights() {
        // house-shaped graph with weights = 1
        let mut graph = UnGraphMap::<u64, i32>::new();
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, 1);
        graph.add_edge(0, 2, 1);
        graph.add_edge(1, 3, 1);
        graph.add_edge(2, 4, 1);
        graph.add_edge(3, 4, 1);

        let mut solution_age: Vec<i32> = vec![1, 1, -1, -1, -1]; // 0, 1 in the solution
        solution_age[0] = 1;
        solution_age[1] = 1;

        let mut dscores = vec![0; graph.node_count()];

        update_weights(&mut graph, &solution_age, &mut dscores);

        // Covered edges not modified
        assert_eq!(*graph.edge_weight(0, 1).unwrap(), 1);
        assert_eq!(*graph.edge_weight(0, 2).unwrap(), 1);
        assert_eq!(*graph.edge_weight(1, 3).unwrap(), 1);
        // Uncovered edges should have their weight increased by 1
        assert_eq!(*graph.edge_weight(2, 4).unwrap(), 2);
        assert_eq!(*graph.edge_weight(3, 4).unwrap(), 2);
    }

    #[test]
    fn test_get_cost() {
        // house-shaped graph with weights = 1
        let mut graph = UnGraphMap::<u64,i32>::new();
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, 1);
        graph.add_edge(0, 2, 1);
        graph.add_edge(1, 3, 1);
        graph.add_edge(2, 4, 1);
        graph.add_edge(3, 4, 1);

        assert_eq!(5, get_cost(&graph, &[0,0,0,0,0]));
        assert_eq!(1, get_cost(&graph, &[0,1,2,0,0]));
        assert_eq!(0, get_cost(&graph, &[1,0,0,1,1]));

    }

    #[test]
    fn test_compute_dscore() {
        // house-shaped graph with weights = 1
        let mut graph = UnGraphMap::<u64,i32>::new();
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, 1);
        graph.add_edge(0, 2, 1);
        graph.add_edge(1, 3, 1);
        graph.add_edge(2, 4, 1);
        graph.add_edge(3, 4, 1);

        assert_eq!(2, compute_dscore(&graph, &mut [0,0,0,0,0], 0));
        assert_eq!(-1, compute_dscore(&graph, &mut [1,0,0,1,1], 4));
    }

    #[test]
    fn test_get_vertex_with_highest_dscore() {
        let dscores = vec![1, 2, 3, 4, 5];
        let expected = 4;
        let res = get_vertex_with_highest_dscore(&dscores);
        assert_eq!(res, expected);
        assert_eq!(dscores, vec![1, 2, 3, 4, 5]) // dscores should not be modified
    }

    #[test]
    fn test_get_vertex_with_highest_dscore_from_solution() {
        let dscores = vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
        let solution = vec![1, 3]; // with dscore = 11 and 13
        let solution_age = vec![-1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1];
        let res = get_vertex_with_highest_dscore_from_solution(&dscores, &solution, &solution_age);
        assert_eq!(res, 3);
        assert_eq!(dscores, vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]); // dscores should not be modified

        // If two vertices have the same dscore, the one with the highest age should be selected
        let dscores = vec![10, 25, 12, 25, 14, 15, 16, 17, 18, 19, 20];
        let solution = vec![0, 1, 3]; // with dscore = 10, 25 and 25
        let solution_age = vec![1, 2, -1, 3, -1, -1, -1, -1, -1, -1, -1];
        let res = get_vertex_with_highest_dscore_from_solution(&dscores, &solution, &solution_age);
        assert_eq!(res, 3);
    }

    #[test]
    fn test_compute_greedy_vc() {
        let mut graph = UnGraphMap::<u64,i32>::new();
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0,1,1);
        graph.add_edge(0,2,1);
        graph.add_edge(0,3,1);
        graph.add_edge(0,4,1);
        graph.add_edge(1,2,1);

        let mut solution = vec![];
        let mut solution_age = vec![-1; graph.node_count()];
        let mut dscores = vec![0; graph.node_count()];
        update_dscores(&graph, &mut solution_age, &mut dscores);

        compute_greedy_vc(&graph, &mut solution, &mut solution_age, &1, &mut dscores);
        assert_eq!(vec![0, 2], solution);
        // Age of vertices in the solution should be set to 1
        assert_eq!(solution_age[0], 1);
        assert_eq!(solution_age[2], 1);
        // Age of vertices not in the solution should be set to 0
        assert_eq!(solution_age[1], -1);
        assert_eq!(solution_age[3], -1);
        assert_eq!(solution_age[4], -1);
    }

    #[test]
    fn test_get_uncovered_edge() {
        // house-shaped graph with weights = 1
        let mut graph = UnGraphMap::<u64, i32>::new();
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, 1);
        graph.add_edge(0, 2, 1);
        graph.add_edge(1, 3, 1);
        graph.add_edge(2, 4, 1);
        graph.add_edge(3, 4, 1);

        let mut solution_age = vec![-1; graph.node_count()];
        solution_age[0] = 1;
        solution_age[1] = 1;

        let (i, j) = get_one_uncovered_edge(&graph, &solution_age, &mut rand::thread_rng());
        assert_ne!(i, j);
        assert_eq!(-1, solution_age[i as usize]);
        assert_eq!(-1, solution_age[j as usize]);
    }

    #[test]
    fn test_get_uncovered_edge_with_only_one_edge_possible() {
        // S5 graph with weights = 1
        let mut graph = UnGraphMap::<u64, i32>::new();
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, 1);
        graph.add_edge(0, 2, 1);
        graph.add_edge(0, 3, 1);
        graph.add_edge(0, 4, 1);

        let solution_age = vec![-1, -1, 1, 1, 1];

        let (i, j) = get_one_uncovered_edge(&graph, &solution_age, &mut rand::thread_rng());
        assert_eq!(0, i);
        assert_eq!(1, j);
    }

    #[test]
    fn test_pick_new_vertex_confchange0() {
        let mut graph = UnGraphMap::<u64, i32>::new();
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, 1);
        graph.add_edge(0, 2, 1);
        graph.add_edge(0, 3, 1);
        graph.add_edge(2, 4, 1);

        let dscores = vec![0, 0, 0, 0, 0];
        let mut confchange = vec![1, 1, 1, 1, 1];
        let solution_age = vec![1, 1, -1, -1, -1]; // 0, 1 in the solution

        confchange[2] = 0;
        let res = pick_new_vertex(&graph, &dscores, &confchange, &solution_age, &mut rand::thread_rng());
        assert_eq!(4, res);

        confchange[2] = 1;
        confchange[4] = 0;
        let res = pick_new_vertex(&graph, &dscores, &confchange, &solution_age, &mut rand::thread_rng());
        assert_eq!(2, res);
    }

    #[test]
    fn test_pick_new_vertex() {
        let mut graph = UnGraphMap::<u64, i32>::new();
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, 1);
        graph.add_edge(0, 2, 1);
        graph.add_edge(0, 3, 1);
        graph.add_edge(2, 4, 1);

        let mut dscores = vec![0, 0, 2, 0, 0];
        let confchange = vec![1, 1, 1, 1, 1];
        let mut solution_age = vec![1, 1, 0, 0, 0]; // 0, 1 in the solution

        // 1 : vertex 2 with bigger dscore
        let res = pick_new_vertex(&graph, &dscores, &confchange, &solution_age, &mut rand::thread_rng());
        assert_eq!(2, res);

        // 2 : vertex 4 with bigger dscore
        dscores[4] = 3;
        let res = pick_new_vertex(&graph, &dscores, &confchange, &solution_age, &mut rand::thread_rng());
        assert_eq!(4, res);

        // 3 : 2 and 4 with the same dscore but 4 is older
        dscores[4] = 2;
        solution_age[2] = -1;
        let res = pick_new_vertex(&graph, &dscores, &confchange, &solution_age, &mut rand::thread_rng());
        assert_eq!(4, res);

        // 4 : 2 and 4 with the same dscore but 2 is older
        solution_age[4] = -3;
        let res = pick_new_vertex(&graph, &dscores, &confchange, &solution_age, &mut rand::thread_rng());
        assert_eq!(2, res);
    }

    #[test]
    fn test_is_vertex_cover_with_weights() {
        let mut graph = UnGraphMap::<u64, i32>::new();
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, 1);
        graph.add_edge(0, 2, 1);
        graph.add_edge(0, 3, 1);
        graph.add_edge(2, 4, 1);

        assert!(!is_vertex_cover_with_weight(&graph, &[1, 1, 0, 0, 0]));
        assert!(is_vertex_cover_with_weight(&graph, &[1, 1, 1, 1, 0]));
    }

    #[test]
    fn test_add_weight() {
        let mut graph = UnGraphMap::<u64, ()>::new();
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, ());
        graph.add_edge(0, 2, ());
        graph.add_edge(0, 3, ());
        graph.add_edge(2, 4, ());

        let graph = add_weight_to_graph(&graph, 3);
        for edge in graph.all_edges() {
            assert_eq!(*edge.2, 3);
        }
    }
}