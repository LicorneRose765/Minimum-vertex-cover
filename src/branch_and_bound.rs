use std::cmp::max;
use std::collections::HashMap;

use petgraph::prelude::UnGraphMap;

use crate::graph_utils::{copy_graph, get_vertex_with_max_degree};

pub fn solve(graph: &UnGraphMap<u64, ()>) -> (u64, Vec<u64>) {

    let mut copy = copy_graph(graph);
    // Initialize the upper bound to the number of nodes in the graph
    // and the vertex cover found so far is empty
    let upper_bound_vc = &copy.nodes().collect();
    let u = b_and_b(graph, &mut copy, graph.node_count() as u64,upper_bound_vc ,vec![]);
    u
}

fn b_and_b<'a>(graph: &UnGraphMap<u64, ()>,
               subgraph: &UnGraphMap<u64, ()>,
               upper_bound: u64,
               upper_bound_vc: &Vec<u64>,
               vertex_cover: Vec<u64>) -> (u64, Vec<u64>) {

    // If the vertex cover found so far + the minimum vertex that needs to be added are greater than the UB
    // if vertex_cover.len() as u64 + max(deg_lb(graph), sat_lb(graph)) >= upper_bound {
    //    return upper_bound;
    // }
    // If the graph is empty. (All edges have been covered)
    if subgraph.edge_count() == 0 {
        return (vertex_cover.len() as u64, vertex_cover);
    }

    let (v, max_deg) = get_vertex_with_max_degree(subgraph, None);
    let lb_1 =   (subgraph.edge_count() / max_deg) as f64;
    let lb_1 = lb_1.ceil() as u64;

    let deg_lb = deg_lb(subgraph);
    let clq_lb = clq_lb(subgraph);
    let lb = max(deg_lb, clq_lb);

    if vertex_cover.len() as u64 + lb_1 >= upper_bound {
        return (upper_bound, upper_bound_vc.clone());
    }

    // let v = get_vertex_with_max_degree(subgraph, None).0;

    let mut subgraph = copy_graph(subgraph);

    // ====> First case <====
    // - G \ {v}
    // - C U v
    let mut vertex_cover_case1 = vertex_cover.clone();

    // Removes v + edges from v to neighbor
    subgraph.remove_node(v);
    vertex_cover_case1.push(v);
    let res_case1 = b_and_b(graph,
                            &subgraph,
                            upper_bound,
                            upper_bound_vc,
                            vertex_cover_case1);

    // ====> Second case <====
    // - G \ N*(v)
    // - C U N(v)
    let mut vertex_cover_case2 = vertex_cover.clone();

    // Remove all neighbors of v + edges from neighbors to their neighbors
    for neighbor in graph.neighbors(v) {
        vertex_cover_case2.push(neighbor);
        subgraph.remove_node(neighbor);
    }

    let res_case2 = {
        if upper_bound >= res_case1.0 {
            b_and_b(graph,
            &subgraph,
            res_case1.0,
            &res_case1.1,
            vertex_cover_case2)
        } else {
            b_and_b(graph,
            &subgraph,
            upper_bound,
            upper_bound_vc,
            vertex_cover_case2)
        }
    };

    return {
        if res_case1.0 >= res_case2.0 {
            res_case2
        } else {
            res_case1
        }
    };
}

fn deg_lb(graph: &UnGraphMap<u64, ()>) -> u64 {
    let size = graph.edge_count();
    let mut selected_vertexes = Vec::<u64>::new();
    let mut sum_degrees: usize = 0;


    let mut working = true;
    while working {
        let (max_degree_vertex, _vertex_degree) = get_vertex_with_max_degree(graph, Some(&selected_vertexes));
        selected_vertexes.push(max_degree_vertex);
        sum_degrees += graph.neighbors(max_degree_vertex).count();
        if sum_degrees >= size {
            working = false;
        }
    }

    let mut edges_left = 0;
    for (u, v, ()) in graph.all_edges() {
        if !selected_vertexes.contains(&u) && !selected_vertexes.contains(&v) {
            edges_left += 1;
        }
    }

    if edges_left == 0 {
        selected_vertexes.len() as u64
    } else {
        let estim = (edges_left / graph.neighbors(get_vertex_with_max_degree(&graph, Some(&selected_vertexes)).0).count()) as f64;
        selected_vertexes.len() as u64 + estim.floor() as u64
    }
}

#[allow(dead_code)]
fn sat_lb(_graph: &UnGraphMap<u64, ()>) -> u64 {
    todo!("Implement lower bound based on satisfiability")
}

fn clq_lb(graph: &UnGraphMap<u64, ()>) -> u64 {
    // 1) Get the complement of the graph
    // 2) Find a greedy coloring of the complement
    // 3) Each color is a independent set
    // 4) An independent set in the complement is a clique in the original graph
    // 5) Adds the numbers of nodes in each clique minus 1 (a clique is a complete graph)

    // 1) Get the complement of the graph
    let compl = complement(graph);

    // 2) Find a greedy coloring of the complement
    let color_set = greedy_coloring(&compl);


    // Adds the number of nodes in each color minus 1 = lower bound. If a value is 0, change it to 1
    color_set.iter().map(|&x| x as u64 - 1).sum::<u64>()
}

fn complement(graph: &UnGraphMap<u64, ()>) -> UnGraphMap<u64, ()> {
    let mut complement = UnGraphMap::<u64,()>::new();
    for node in graph.nodes() {
        complement.add_node(node);
    }

    for a in graph.nodes() {
        for b in graph.nodes() {
            if a != b && !graph.contains_edge(a, b) {
                complement.add_edge(a, b, ());
            }
        }
    }
    complement
}

// Color the graph such that every node has a different color than its neighbors.
// This algorithm returns a vector containing the number of vertex in each color.
fn greedy_coloring(graph: &UnGraphMap<u64, ()>) -> Vec<usize> {
    // 1. Create a color set. The vertex degree of each vertex is calculated and the vertex degrees are added to
    let mut color_set = Vec::new(); // color_set[i] = j means that color i has j vertexes
    let mut colors = HashMap::new();
    for i in graph.nodes() {
        colors.insert(i, 0);
    }

    let mut vertices_ordered_by_deg: Vec<_> = graph.nodes().collect();
    vertices_ordered_by_deg.sort_by_key(|&i| std::cmp::Reverse(graph.neighbors(i).count()));



    for vertex in vertices_ordered_by_deg {
        let mut color   = 0;
        let mut color_found = false;
        while !color_found {
            if color_set.len() <= color {
                color_set.push(1);
                colors.insert(vertex, color);
                color_found = true;
            } else {
                let is_conflict = {
                    let mut is_conflict = false;
                    for neighbor in graph.neighbors(vertex) {
                        if *colors.get(&neighbor).unwrap() == color {
                            is_conflict = true;
                        }
                    }
                    is_conflict
                };
                if is_conflict {
                    color += 1;
                } else {
                    color_set[color] += 1;
                    colors.insert(vertex, color);
                    color_found = true;
                }
            }
        }
    }
    color_set

}

#[cfg(test)]
mod branch_and_bound_tests {
    use crate::graph_utils::load_clq_file;

    use super::*;

    #[test]
    fn test_complement() {
        let mut g = UnGraphMap::<u64, ()>::new();
        for i in 0..4 {
            g.add_node(i);
        }
        g.add_edge(0, 1, ());
        g.add_edge(0, 2, ());
        g.add_edge(2, 3, ());

        let compl = complement(&g);
        assert_eq!(compl.edge_count(), 3);
        assert_eq!(compl.node_count(), 4);
        assert!(compl.contains_edge(1, 3));
        assert!(compl.contains_edge(1, 2));
        assert!(compl.contains_edge(0, 3));
    }

    #[test]
    fn test_greedy_coloring() {
        let mut graph = UnGraphMap::<u64, ()>::new();
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, ());
        graph.add_edge(0, 2, ());
        graph.add_edge(1, 3, ());
        graph.add_edge(2, 4, ());
        graph.add_edge(3, 4, ());

        let res = greedy_coloring(&graph);
        assert_eq!(res, vec![2,2,1])
    }

    #[test]
    fn test_greedy_coloring_paper_graph() {
        let graph = load_clq_file("src/resources/graphs/oui.clq").unwrap();
        let res = greedy_coloring(&graph);
        assert_eq!(res, vec![3,3,1])
    }

    #[test]
    fn test_problem_with_coloring() {
        let mut graph = UnGraphMap::<u64, ()>::new();
        graph.add_node(4);
        graph.add_node(5);
        graph.add_node(6);
        graph.add_edge(4, 5, ());
        graph.add_edge(5, 6, ());

        let res = greedy_coloring(&graph);
        assert_eq!(res, vec![1,2])
    }

    #[test]
    fn test_deg_leb() {
        let graph = match load_clq_file("src/resources/graphs/test_cycle_5.clq") {
            Ok(g) => g,
            Err(e) => panic!("{}", e)

        };
        let res = deg_lb(&graph);
        assert_eq!(res, 3);
    }

    #[test]
    fn test_b_and_b() {
        let mut graph = UnGraphMap::<u64, ()>::new();
        for i in 0..4 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, ());
        graph.add_edge(1, 2, ());
        graph.add_edge(2, 0, ());
        graph.add_edge(2, 3, ());

        assert_eq!(solve(&graph).0, 2);
    }

    #[test]
    fn test_with_queen_5() {
        let graph = load_clq_file("src/resources/graphs/queen5_5.clq").unwrap();
        let res = solve(&graph);
        assert_eq!(res.0, 20);
    }

}
