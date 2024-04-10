use petgraph::prelude::UnGraphMap;
use rand::seq::IteratorRandom;

use crate::Clock;

// TODO: Maybe use a function which take a grave without weights and returns a graph with weights
// To use it here.
#[allow(dead_code)]
pub fn numvc(
    graph: &UnGraphMap<u64, i32>,
    clock: &mut Clock
) -> Vec<u64> {
    let mut iter = 1; // Number of iterations

    // Initialize dscores of vertices
    let mut dscores: Vec<i32> = vec![0; graph.node_count()];
    let mut solution: Vec<u64> = vec![];
    let mut solution_age: Vec<i32> = vec![0; graph.node_count()];
    // Time since the vertex was added/removed from the solution.
    // Added : positive value, Removed : negative value

    update_dscores(graph, &mut solution, &mut dscores);

    // Initialize confchange array (CC strategy)
    let mut confchange: Vec<u64> = vec![1; graph.node_count()];

    // Construct the first solution greedily by selecting the vertex with the highest dscore
    compute_greedy_vc(graph, &mut solution, &mut solution_age, &iter, &mut dscores);
    let mut best_solution = solution.clone();

    // Main loop
    while !clock.is_time_up() {
        if is_vertex_cover_with_weight(graph, &solution_age) {
            best_solution = solution.clone();
            // Remove a vertex with highest dscore from C
            let v = get_vertex_with_highest_dscore_from_solution(&mut dscores, &mut solution, &solution_age);
            solution.retain(|&x| x != v);
            solution_age[v as usize] = -iter; // Set age to negative time since it was removed
            update_dscores(graph, &mut solution, &mut dscores);
            continue
        }
        // Select a vertex from c with the highest dscore
        let u = get_vertex_with_highest_dscore_from_solution(&mut dscores, &mut solution, &solution_age);
        // Remove u from C
        solution.retain(|&x| x != u);
        solution_age[u as usize] = -iter;
        update_confchange(graph, u, &mut confchange, true);

        // Select a vertex to add
        let v = pick_new_vertex(graph, &dscores, &confchange, &solution_age);

        // Add v to C
        solution.push(v);
        solution_age[v as usize] = iter;
        update_confchange(graph, v, &mut confchange, false);


        iter += 1;
    }

    return best_solution;
}

/// Modify the confchange array in reaction of the addition or removal of a vertex. (in place)
///
/// See the CC strategy in the source paper for more information.
///
/// # Parameters
/// - `graph`: The graph on which the vertex cover is computed
/// - `u`: The vertex that was added or removed
/// - `confchange`: The confchange array
/// - `is_u_changing`: True if the vertex was removed. False if the vertex was added.
fn update_confchange(
    graph: &UnGraphMap<u64, i32>,
    u: u64,
    confchange: &mut Vec<u64>,
    is_u_changing: bool) {
    if is_u_changing {
        confchange[u as usize] = 0;
    }
    for z in graph.neighbors(u) {
        confchange[z as usize] = 1;
    }
}

/// Add 1 to the weight of all edges that are covered by the solution (in place)
///
/// # Parameters
/// - `graph`: The graph on which the vertex cover is computed
/// - `solution_age`: The array containing the age of the vertices in the solution. (The vertices that are not in the solution have an age of 0)
fn update_weights(
    graph: &mut UnGraphMap<u64, i32>,
    solution_age: &Vec<i32>, ) {
    for edge in graph.all_edges_mut() {
        if solution_age[edge.0 as usize] <= 0 && solution_age[edge.1 as usize] <= 0 {
            // If solution_old > 0, it means that the vertex is currently in the solution
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
    solution: &Vec<u64>
) -> i32 {
    let mut cost = 0;
    for (a, b, w) in graph.all_edges() {
        // TODO: modify this to use solution_age
        if !solution.contains(&a) && !solution.contains(&b) {
            cost += w;
        }
    }
    return cost;
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
    solution: &mut Vec<u64>,
    vertex: u64
) -> i32 {
    // TODO: use solution_age instead of solution
    let c1 = get_cost(graph, solution);
    let c2;
    if solution.contains(&vertex) {
        solution.retain(|&x| x != vertex); // Remove vertex from solution
        c2 = get_cost(graph, solution);
        solution.push(vertex); // Add vertex back to solution
    } else {
        solution.push(vertex); // Add vertex to solution
        c2 = get_cost(graph, solution);
        solution.retain(|&x| x != vertex); // Remove vertex from solution
    }
    return c1 - c2;
}

/// Update the dscores of all vertices in the graph. (In place)
///
/// # Parameters
/// - `graph`: The graph on which the vertex cover is computed
/// - `solution`: The current solution
/// - `dscores`: The dscores of the vertices
fn update_dscores(
    graph: &UnGraphMap<u64, i32>,
    solution: &mut Vec<u64>,
    dscores: &mut Vec<i32>
) {
    for vertex in graph.nodes() {
        dscores[vertex as usize] = compute_dscore(graph, solution, vertex);
    }
}

/// Get the vertex with the highest dscore.
///
/// By definition of dscore, this vertex is not in the current solution.
///
/// # Parameters
/// - `dscores`: The dscores of the vertices
fn get_vertex_with_highest_dscore(dscores: &Vec<i32>) -> u64 {
    if let Some((max_index, _)) = dscores.iter().enumerate().max_by_key(|&(_, &item)| item) {
        return max_index as u64;
    } else {
        panic!("Error in vertex cover computation (should not happen)")
    };
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
    dscores: &Vec<i32>,
    solution: &Vec<u64>,
    solution_age: &Vec<i32>) -> u64 {
    let mut max_vertex = match solution.get(0) {
        Some(v) => *v,
        None => panic!("Trying to get vertex with highest dscore from an empty solution")
    };

    for vertex in solution.iter() {
        if dscores[*vertex as usize] > dscores[max_vertex as usize] {
            max_vertex = *vertex;
        } else if dscores[*vertex as usize] == dscores[max_vertex as usize] {
            if solution_age[*vertex as usize] > solution_age[max_vertex as usize] {
                // If vertex is older than max_vertex, prefer it.
                max_vertex = *vertex;
            }
        }
    }

    return max_vertex;
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
    solution_age: &mut Vec<i32>,
    iter: &i32,
    dscores: &mut Vec<i32>,
) {
    while !is_vertex_cover_with_weight(graph, solution_age) {
        let v = get_vertex_with_highest_dscore(dscores);
        solution.push(v);
        solution_age[v as usize] = *iter;
        update_dscores(graph, solution, dscores);
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
fn get_one_uncovered_edge(graph: &UnGraphMap<u64, i32>, solution_age: &Vec<i32>) -> (u64, u64) {
    let edges = graph.all_edges()
        .filter(|(i, j, _)| solution_age[*i as usize] <= 0 && solution_age[*j as usize] <= 0);

    // Select randomly an edge from the iterator
    let edge: (u64, u64, &i32) = edges.choose(&mut rand::thread_rng()).unwrap();

    return (edge.0, edge.1);
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
    dscores: &Vec<i32>,
    confchange: &Vec<u64>,
    solution_old: &Vec<i32>,
) -> u64 {
    let (u, v) = get_one_uncovered_edge(graph, &solution_old);

    return if confchange[u as usize] == 0 {
        v
    } else if confchange[v as usize] == 0 {
        u
    } else {
        // Select vertex with highest dscore
        let mut max_vertex = u;
        if dscores[v as usize] > dscores[u as usize] {
            max_vertex = v;
        } else if dscores[v as usize] == dscores[u as usize] {
            // Break tie by selecting the oldest vertex
            if solution_old[v as usize] > solution_old[u as usize] {
                // Since v and u are not in the solution, the age is negative.
                // We want to select the oldest vertex, so the one with the smallest value in absolute value
                // or the biggest value in the negative.
                max_vertex = v;
            }
        }
        max_vertex
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
pub fn is_vertex_cover_with_weight(graph: &UnGraphMap<u64, i32>, solution_age: &Vec<i32>) -> bool {
    for (i, j, _) in graph.all_edges() {
        if solution_age[i as usize] <= 0 && solution_age[j as usize] <= 0 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod numvc_tests {
    use petgraph::graphmap::UnGraphMap;

    use super::*;

    #[test]
    #[ignore]
    fn test_numvc() {
        todo!("Implement test")
    }

    #[test]
    fn test_update_confchange() {
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

        let mut confchange = vec![1, 0, 0, 0, 0];
        // Remove vertex 0, confchange of 0 should be set to 0
        // Neighbors of 0 should be set to 1
        update_confchange(&graph, 0, &mut confchange, true);
        assert_eq!(vec![0, 1, 1, 0, 0], confchange);

        let mut confchange = vec![1, 0, 0, 0, 0];
        // Add vertex 0, confchange of its neighbors should be set to 1
        update_confchange(&graph, 0, &mut confchange, false);
        assert_eq!(vec![1, 1, 1, 0, 0], confchange);
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

        update_weights(&mut graph, &solution_age);

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

        assert_eq!(5, get_cost(&graph, &vec![]));
        assert_eq!(1, get_cost(&graph, &vec![1,2]));
        assert_eq!(0, get_cost(&graph, &vec![3,4,0]));

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

        assert_eq!(2, compute_dscore(&graph, &mut vec![], 0));
        assert_eq!(-1, compute_dscore(&graph, &mut vec![3,4,0], 4));
    }

    #[test]
    fn test_get_vertex_with_highest_dscore() {
        let mut dscores = vec![1, 2, 3, 4, 5];
        let expected = 4;
        let res = get_vertex_with_highest_dscore(&mut dscores);
        assert_eq!(res, expected);
        assert_eq!(dscores, vec![1, 2, 3, 4, 5]) // dscores should not be modified
    }

    #[test]
    fn test_get_vertex_with_highest_dscore_from_solution() {
        let mut dscores = vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
        let mut solution = vec![1, 3]; // with dscore = 11 and 13
        let solution_age = vec![-1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1];
        let res = get_vertex_with_highest_dscore_from_solution(&mut dscores, &mut solution, &solution_age);
        assert_eq!(res, 3);
        assert_eq!(dscores, vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]); // dscores should not be modified

        // If two vertices have the same dscore, the one with the highest age should be selected
        let mut dscores = vec![10, 25, 12, 25, 14, 15, 16, 17, 18, 19, 20];
        let mut solution = vec![0, 1, 3]; // with dscore = 10, 25 and 25
        let solution_age = vec![1, 2, -1, 3, -1, -1, -1, -1, -1, -1, -1];
        let res = get_vertex_with_highest_dscore_from_solution(&mut dscores, &mut solution, &solution_age);
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
        update_dscores(&graph, &mut solution, &mut dscores);

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

        let (i, j) = get_one_uncovered_edge(&graph, &solution_age);
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

        let (i, j) = get_one_uncovered_edge(&graph, &solution_age);
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
        let res = pick_new_vertex(&graph, &dscores, &confchange, &solution_age);
        assert_eq!(4, res);

        confchange[2] = 1;
        confchange[4] = 0;
        let res = pick_new_vertex(&graph, &dscores, &confchange, &solution_age);
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
        let res = pick_new_vertex(&graph, &dscores, &confchange, &solution_age);
        assert_eq!(2, res);

        // 2 : vertex 4 with bigger dscore
        dscores[4] = 3;
        let res = pick_new_vertex(&graph, &dscores, &confchange, &solution_age);
        assert_eq!(4, res);

        // 3 : 2 and 4 with the same dscore but 4 is older
        dscores[4] = 2;
        solution_age[2] = -1;
        let res = pick_new_vertex(&graph, &dscores, &confchange, &solution_age);
        assert_eq!(4, res);

        // 4 : 2 and 4 with the same dscore but 2 is older
        solution_age[4] = -3;
        let res = pick_new_vertex(&graph, &dscores, &confchange, &solution_age);
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

        assert_eq!(false, is_vertex_cover_with_weight(&graph, &vec![1, 1, 0, 0, 0]));
        assert_eq!(true, is_vertex_cover_with_weight(&graph, &vec![1, 1, 1, 1, 0]));
    }
}