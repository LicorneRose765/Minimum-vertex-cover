use petgraph::graphmap::UnGraphMap;
use rand::prelude::ThreadRng;
use rand::Rng;
use rand::seq::IteratorRandom;

use crate::Clock;
use crate::graph_utils::get_vertex_with_max_degree;

/// SAMVC algorithm is an algorithm that finds the minimum vertex cover of a graph using the
/// Simulated Annealing algorithm.
pub fn samvc_algorithm(
    graph: &UnGraphMap<u64, ()>,
    clock: &mut Clock,
    optimal: Option<u64>,
) -> Vec<u64> {
    println!("Starting SAMVC algorithm...");
    let mut rng = rand::thread_rng();

    // ======= Part 0 : Parameters =======
    let final_temperature = 0.001;
    let mut temperature = 100.0; // Initial temperature
    let cooling_rate = 0.9995;
    let max_iter_per_temp = (graph.node_count() as u64)* (graph.node_count() as u64); // Number of iter for each temperature
 

    // ======= Part 1 : Initialization =======
    let mut current_solution = generate_initial_solution(graph);
    let mut best_solution = current_solution.clone();
    let mut best_cost = current_solution.len() as u64;
    let mut current_cost = best_cost;

    if let Some(optimal) = optimal {
        // If the optimal solution is found using the initial solution, return it
        if best_cost == optimal {
            println!("best cost == greedy");
            return best_solution;
        }
    }

    // Iterate until the temperature is below the final temperature
    while !clock.is_time_up() && temperature > final_temperature {
        let mut iter = 0;

        // Iterate until the number of iterations without improvement is reached
        while !clock.is_time_up() && iter < max_iter_per_temp {
            let (next_solution, next_cost) = generate_next_solution(graph, &current_solution, &current_cost, &mut rng);
            let delta = next_cost as f64 - current_cost as f64;

            // If the new solution is better, accept it
            if delta < 0.0 {
                current_solution = next_solution;
                current_cost = next_cost;

                // If the new solution is the best so far, update the best solution
                if current_cost < best_cost {
                    best_solution.clone_from(&current_solution);
                    best_cost = current_cost;

                    if let Some(optimal) = optimal {
                        if best_cost == optimal {
                            return best_solution;
                        }
                    }
                }
            } else {
                // If the new solution is worse, accept it with a probability
                if do_accept_solution(delta, temperature, &mut rng) {
                    current_solution = next_solution;
                    current_cost = next_cost;
                }
            }
            iter += 1;
        }

        // Cooling
        println!("Temperature: {}", temperature);
        temperature *= cooling_rate;
    }

    best_solution
}


/// Compute the probability of accepting a worse solution given the temperature and the cost delta.
///
/// The probability is computed as $exp^{-(delta / temperature)}.
fn compute_probability(delta: f64, temperature: f64) -> f64 {
    (delta / temperature).exp()
}

/// Return true if the solution should be accepted, false otherwise.
///
/// The solution is accepted if the acceptance probability is greater than a random number between 0 and 1.
fn do_accept_solution(delta: f64, temperature: f64, rng: &mut ThreadRng) -> bool {
    let acceptance_probability = compute_probability(delta, temperature);
    // Generate a number between 0 and 1 and compare it to the acceptance probability
    acceptance_probability > rng.gen::<f64>()
}

/// Compute the next solution by randomly selecting a vertex and flipping it. Also, it computes the
/// cost of the new solution and returns it.
fn generate_next_solution(graph: &UnGraphMap<u64, ()>, current_solution: &[u64], current_cost: &u64, rng: &mut ThreadRng) -> (Vec<u64>, u64) {
    let mut next_solution= current_solution.to_vec();
    // Randomly select a vertex to flip
    let vertex_to_flip = graph.nodes().choose(rng).unwrap();

    let mut next_cost = *current_cost;
    // flip the vertex
    if next_solution.contains(&vertex_to_flip) {
        next_solution.retain(|&x| x != vertex_to_flip);

        // Decrease the cost by 1.
        next_cost -= 1;
        // Increase the cost by 1 for each edge that is not covered by the new solution.
        for (a, b, ()) in graph.edges(vertex_to_flip) {
            if (a != vertex_to_flip && !next_solution.contains(&a))
                || (b != vertex_to_flip && !next_solution.contains(&b)) {
                next_cost += 1;
            }
        }
    } else {
        next_solution.push(vertex_to_flip);

        // Increase the cost by 1.
        next_cost += 1;
        // Decrease the cost by 1 for each edge that is covered by the new solution but not by the
        // previous solution.
        for (a, b, ()) in graph.edges(vertex_to_flip) {
            if (a != vertex_to_flip && !next_solution.contains(&a))
                || (b != vertex_to_flip && !next_solution.contains(&b)) {
                next_cost -= 1;
            }
        }
    }

    (next_solution, next_cost)
}

/// This function will generate an initial solution for the SAMVC algorithm.
///
/// The generated solution is computed greedily by selecting the vertex with the highest degree
/// until all edges are covered and the vertex cover is formed.
///
/// # Arguments
/// * `graph` - The graph to generate the initial solution from.
///
/// # Returns
/// * A vector of vertex indices that form the initial solution.
fn generate_initial_solution(graph: &UnGraphMap<u64, ()>) -> Vec<u64> {
    let mut vertex_cover = vec![];
    let mut graph = graph.clone();

    while graph.edge_count() > 0 {
        let v = get_vertex_with_max_degree(&graph, None).0;
        vertex_cover.push(v);
        graph.remove_node(v);
    }
    vertex_cover
}


#[cfg(test)]
mod tests {
    use crate::graph_utils::load_clq_file;
    use crate::samvc;

    use super::*;

    #[test]
    fn test_compute_solution_small_graph() {
        let graph = load_clq_file("src/resources/graphs/test.clq")
            .expect("Error while loading the graph");
        let mut clock = Clock::new(300); // 1 hour time limit
        let res = samvc(&graph, &mut clock, Some(&[0.0,0.0]), Some(3));
        assert_eq!(res.0, 3);
        assert_eq!(res.1, vec![0, 4, 3]);
    }
}


