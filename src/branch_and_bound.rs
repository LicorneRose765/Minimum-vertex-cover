use std::cmp::max;
use std::collections::HashMap;
use std::sync::Arc;

use petgraph::prelude::UnGraphMap;

use crate::Clock;
use crate::graph_utils::{complement, copy_graph, get_vertex_with_max_degree};

pub fn b_and_b(_graph: &UnGraphMap<u64, ()>,
               g: &UnGraphMap<u64, ()>,
               upper_bound: u64,
               upper_bound_vc: &Vec<u64>,
               vertex_cover: Vec<u64>,
               clock: &mut Clock) -> (u64, Vec<u64>) {
    if clock.is_time_up() {
        return (upper_bound, upper_bound_vc.clone());
    }

    clock.enter_subroutine("copy");
    let mut subgraph = copy_graph(g);
    clock.exit_subroutine("copy").expect("Error while exiting subroutine");

    if subgraph.edge_count() == 0 {
        // If the subgraph is empty, all edges are covered => vertex cover
        return (vertex_cover.len() as u64, vertex_cover);
    }

    clock.enter_subroutine("max_deg");
    let (v, _max_deg) = get_vertex_with_max_degree(&subgraph, None);
    clock.exit_subroutine("max_deg").expect("Error while exiting subroutine");

    if vertex_cover.len() as u64 + compute_lb(copy_graph(&subgraph), clock)  >= upper_bound {
        // We can't find a better solution in this branch, we stop and return the best known solution
        return (upper_bound, upper_bound_vc.clone());
    }

    let neighbors: Vec<u64> = subgraph.neighbors(v).collect();

    // ====> First case <====
    // - G \ {v}
    // - C U v
    let mut vertex_cover_case1 = vertex_cover.clone();

    // Removes v + edges from v to neighbor
    subgraph.remove_node(v);
    vertex_cover_case1.push(v);
    let res_case1 = b_and_b(_graph,
                            &subgraph,
                            upper_bound,
                            upper_bound_vc,
                            vertex_cover_case1, clock);

    // ====> Second case <====
    // - G \ N*(v)
    // - C U N(v)
    let mut vertex_cover_case2 = vertex_cover.clone();

    // Remove all neighbors of v + edges from neighbors to their neighbors
    for neighbor in neighbors {
        vertex_cover_case2.push(neighbor);
        subgraph.remove_node(neighbor);
    }

    let res_case2 = {
        if upper_bound >= res_case1.0 {
            b_and_b(_graph,
                    &subgraph,
                    res_case1.0,
                    &res_case1.1,
                    vertex_cover_case2, clock)
        } else {
            b_and_b(_graph,
                    &subgraph,
                    upper_bound,
                    upper_bound_vc,
                    vertex_cover_case2,
                    clock)
        }
    };

    {
        if res_case1.0 >= res_case2.0 {
            res_case2
        } else {
            res_case1
        }
    }
}

fn compute_lb(graph: UnGraphMap<u64, ()>, clock: &mut Clock) -> u64 {
    let graph = Arc::new(graph);

    // First thread : deg_lb
    let shared_deg = Arc::clone(&graph);
    let shared_clq = Arc::clone(&graph);
    let handle_deg = std::thread::spawn(move || {
        deg_lb(&shared_deg)
    });

    let handle_clq = std::thread::spawn(move || {
        clq_lb(&shared_clq)
    });
    clock.enter_subroutine("deg_lb");
    let deg_lb = handle_deg.join().unwrap();
    clock.exit_subroutine("deg_lb").expect("Error while exiting subroutine");

    clock.enter_subroutine("clq_lb");
    let clq_lb = handle_clq.join().unwrap();
    clock.exit_subroutine("clq_lb").expect("Error while exiting subroutine");
    max(deg_lb, clq_lb)
}

fn deg_lb(graph: &UnGraphMap<u64, ()>) -> u64 {

    let size = graph.edge_count();
    let mut selected_vertexes = Vec::<u64>::new();
    let mut sum_degrees: usize = 0;

    let mut subgraph = copy_graph(graph);

    let mut working = true;
    while working {
        // Get the vertex with the highest degree in the subgraph.
        let (max_degree_vertex, _vertex_degree) = get_vertex_with_max_degree(&subgraph, None);
        selected_vertexes.push(max_degree_vertex);
        sum_degrees += graph.neighbors(max_degree_vertex).count();
        subgraph.remove_node(max_degree_vertex);
        if sum_degrees >= size {
            working = false;
        }
    }

    let edges_left = subgraph.edge_count();
    let next_vertex = get_vertex_with_max_degree(&subgraph, None).0;
    if edges_left == 0 {
        selected_vertexes.len() as u64
    } else {
        let estimation = (edges_left / graph.neighbors(next_vertex).count()) as f64;
        (selected_vertexes.len() as f64 + estimation).floor() as u64
    }
}

#[allow(dead_code)]
fn sat_lb(_graph: &UnGraphMap<u64, ()>) -> u64 {
    todo!("Implement lower bound based on satisfiability")
}


fn clq_lb(graph: &UnGraphMap<u64, ()>) -> u64 {
    // 1) Get the complement of the graph
    // 2) Find a greedy coloring of the complement
    // 3) Each color is an independent set
    // 4) An independent set in the complement is a clique in the original graph
    // 5) Adds the numbers of nodes in each clique minus 1 (a clique is a complete graph)

    // 1) Get the complement of the graph
    let compl = complement(graph);

    // 2) Find a greedy coloring of the complement
    let color_set = welch_powell(&compl);


    // Adds the number of nodes in each color minus 1 = lower bound. If a value is 0, change it to 1
    color_set.iter().map(|&x| (x-1) as u64).sum::<u64>()
}

// Color the graph such that every node has a different color than its neighbors.
// This algorithm returns a vector containing the number of vertex in each color.
//
// This is a LDO algorithm (Largest Degree Order) : the vertices are ordered by decreasing degree.
#[allow(dead_code)]
fn greedy_coloring(graph: &UnGraphMap<u64, ()>) -> Vec<usize> {
    // 1. Create a color set. The vertex degree of each vertex is calculated and the vertex degrees are added to
    let mut color_set = Vec::new(); // color_set[i] = j means that color i has j vertices
    let mut colors = HashMap::new();
    for i in graph.nodes() {
        colors.insert(i, 0);
    }

    let mut vertices_ordered_by_deg: Vec<_> = graph.nodes().collect();
    // Sort vertices by decreasing degree
    vertices_ordered_by_deg.sort_by_key(|&i| std::cmp::Reverse(graph.neighbors(i).count()));


    for vertex in vertices_ordered_by_deg {
        let mut color = 0;
        let mut color_found = false;
        while !color_found {
            if color_set.len() <= color {
                // We need to add a new color to color this vertex
                color_set.push(1);
                colors.insert(vertex, color);
                color_found = true;
            } else {
                // Check if the color is already used by a neighbor
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
                    // The color is already used by a neighbor, we try the next color
                    color += 1;
                } else {
                    // The color is not used by a neighbor, we color the vertex with this color
                    color_set[color] += 1;
                    colors.insert(vertex, color);
                    color_found = true;
                }
            }
        }
    }

    assert!(is_good_coloring(graph, &colors));

    color_set
}

#[allow(dead_code)]
fn _renumber(graph: &UnGraphMap<u64, ()> ,
            vertex: u64,
            color_v: usize,
            max_color: usize,
            colors: &mut HashMap<u64, usize>,
            color_set: &mut [usize],
            color_edges: &mut [Vec<u64>],
) {
    for k1 in 1..max_color-1 {
        // If a neighbor of vertex has the same color as vertex, we renumber vertex
        let mut q=0;
        let mut found = false;
        for test_q in &color_edges[k1] {
            if graph.contains_edge(vertex, *test_q) {
                if found {
                    // More than one neighbor has the same color as vertex, we don't renumber vertex
                    return;
                }
                q = *test_q;
                found = true;
                break;
            }
        }
        if !found {
            // No neighbor has the same color as vertex, we don't renumber vertex
            return;
        }
        for k2 in (k1+1)..max_color {
            let number_of_conflicts = color_edges[k2].iter().filter(|&x| graph.contains_edge(q, *x)).count();
            if number_of_conflicts == 0 {
                // We can renumber vertex

                // 1 - remove vertex from color_v
                color_edges[color_v].retain(|&x| x != vertex);
                color_set[color_v] -= 1;

                // 2 - remove q from color k1
                color_edges[k1].retain(|&x| x != q);

                // 3 - add vertex to color k1
                color_edges[k1].push(vertex);
                colors.insert(vertex, k1); // Change the color of vertex to k1

                // 3 - add q to color k2
                color_edges[k2].push(q);
                colors.insert(q, k2); // Change the color of q to k2
                color_set[k2] += 1;

                return;
            }
        }

    }
}

fn is_good_coloring(graph: &UnGraphMap<u64, ()>, colors: &HashMap<u64, usize>) -> bool {
    for vertex in graph.nodes() {
        for neighbor in graph.neighbors(vertex) {
            if colors.get(&vertex) == colors.get(&neighbor) {
                return false;
            }
        }
    }
    true
}

#[allow(dead_code)]
#[allow(clippy::needless_range_loop)]
fn welch_powell(graph: &UnGraphMap<u64, ()>) -> Vec<usize> {
    // sort vertices by decreasing degree
    let sorted_vertices = {
        let mut vertices: Vec<_> = graph.nodes().collect();
        vertices.sort_by_key(|&i| std::cmp::Reverse(graph.neighbors(i).count()));
        vertices
    };

    // create an array of colors such as color[i] = j means that vertex i has color j
    let mut color: Vec<i32> = {
        let mut color = Vec::new();
        for _ in 0..sorted_vertices.len() {
            color.push(-1);
        }
        color
    };
    // Array such as res[i] = j means that color j colored i vertices
    let mut res: Vec<usize> = Vec::new();

    let mut vertex_to_index = HashMap::new();
    for (i, vertex) in sorted_vertices.iter().enumerate() {
        vertex_to_index.insert(vertex, i);
    }

    let mut current_color = 0;

    loop {
        let mut biggest_index = -1;
        // Find the biggest vertex that is not colored
        for i in 0..sorted_vertices.len() {
            // vertex - 1 because vertex are numbered from 1 to n
            if color[i] == -1 {
                biggest_index = i as i32;
                break;
            }
        }
        if biggest_index == -1 {
            // All vertices are colored
            break;
        }
        // Color the biggest vertex
        color[biggest_index as usize] = current_color;
        res.push(1);

        assert_eq!(color.len(), sorted_vertices.len());
        // Color vertices that are not neighbors of vertex colored with current_color
        for i in 0..sorted_vertices.len() {
            if color[i] == -1 {
                let mut can_color = true;
                for neighbor in graph.neighbors(sorted_vertices[i]) {
                    let index = *vertex_to_index.get(&neighbor).unwrap();
                    if color[index] == current_color {
                        // If a neighbor is already colored with current_color, we don't color the vertex
                        can_color = false;
                        break;
                    }
                }
                if can_color {
                    color[i] = current_color;
                    res[current_color as usize] += 1;
                }
            }
        }
        current_color += 1;
    }

    res
}

#[cfg(test)]
mod branch_and_bound_tests {
    use crate::branch_and_bound;
    use crate::graph_utils::load_clq_file;

    use super::*;

    #[test]
    fn test_greedy_coloring() {
        let mut graph = Box::new(UnGraphMap::<u64, ()>::new());
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, ());
        graph.add_edge(0, 2, ());
        graph.add_edge(1, 3, ());
        graph.add_edge(2, 4, ());
        graph.add_edge(3, 4, ());

        let res = greedy_coloring(&graph);
        assert_eq!(res, vec![2, 2, 1])
    }

    #[test]
    fn test_greedy_coloring_paper_graph() {
        let graph = load_clq_file("src/resources/graphs/oui.clq").unwrap();
        let res = greedy_coloring(&graph);
        assert_eq!(res, vec![3, 3, 1])
    }

    #[test]
    fn test_problem_with_coloring() {
        let mut graph = Box::new(UnGraphMap::<u64, ()>::new());
        graph.add_node(4);
        graph.add_node(5);
        graph.add_node(6);
        graph.add_edge(4, 5, ());
        graph.add_edge(5, 6, ());

        let res = greedy_coloring(&graph);
        assert_eq!(res, vec![1, 2])
    }

    #[test]
    fn test_deg_lb() {
        let graph = match load_clq_file("src/resources/graphs/test_cycle_5.clq") {
            Ok(g) => g,
            Err(e) => panic!("{}", e)
        };
        let res = deg_lb(&graph);
        assert_eq!(res, 3);
    }

    #[test]
    fn test_b_and_b() {
        let mut graph = Box::new(UnGraphMap::<u64, ()>::new());
        for i in 0..4 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, ());
        graph.add_edge(1, 2, ());
        graph.add_edge(2, 0, ());
        graph.add_edge(2, 3, ());

        assert_eq!(branch_and_bound(&graph, &mut Clock::new(3600), None, None).0, 2);
    }

    #[test]
    fn test_with_queen_5() {
        let graph = load_clq_file("src/resources/graphs/queen5_5.clq").unwrap();
        let res = branch_and_bound(&graph, &mut Clock::new(3600), None, None);
        assert_eq!(res.0, 20);
    }

    #[test]
    fn test_welsh() {
        let g = load_clq_file("src/resources/graphs/test_welsh.clq").unwrap();

        let res = welch_powell(&g);
        assert_eq!(res, vec![3, 5, 3]);
    }
}