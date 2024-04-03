use petgraph::prelude::UnGraphMap;
use crate::Clock;

// TODO: Maybe use a function which take a grave without weights and returns a graph with weights
// To use it here.
pub fn numvc(
    graph: &UnGraphMap<u64, i32>, // Use u64 as edge weights
    clock: &mut Clock
) -> Vec<u64> {
    // Initialize dscores of vertices
    let mut dscores: Vec<i32> = vec![0; graph.node_count()];
    let mut solution: Vec<u64> = vec![];
    for vertex in graph.nodes() {
        dscores[vertex as usize] = compute_dscore(graph, &mut solution, vertex);
    }

    // Initialize confchange array
    let mut confchange: Vec<u64> = vec![1; graph.node_count()];

    // Construct the first solution greedily by selecting the vertex with the highest dscore

    return solution;
}


/// Get the cost of a solution
///
/// The cost of a solution X is computed as follows:
/// ` cost(G, X) = SUM_{e in E and e not covered by X} w(e) `
/// where w(e) is the weight of edge e
fn get_cost(
    graph: &UnGraphMap<u64, i32>,
    solution: &Vec<u64>
) -> i32 {
    let mut cost = 0;
    for (a, b, w) in graph.all_edges() {
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
/// Where c' is the solution C with vertex v removed or added. (depends if v is in C or not)
pub fn compute_dscore(
    graph: &UnGraphMap<u64, i32>, // Use u64 as edge weights
    solution: &mut Vec<u64>,
    vertex: u64
) -> i32 {
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

#[cfg(test)]
mod numvc_tests {
    use petgraph::graphmap::UnGraphMap;

    use super::*;
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
    #[ignore]
    fn test_numvc() {
        todo!("Implement test")
    }
}