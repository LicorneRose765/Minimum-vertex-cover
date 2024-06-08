//! [DEPRECATED]
//! This module contains a custom graph implementation that is used to represent undirected graphs.
//! We wanted to test if it was faster than the petgraph implementation. It is not.
use std::collections::HashMap;
use std::fmt::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[allow(dead_code)]
pub fn load_clq_file(path: &str) -> Result<MVCGraph, Box<dyn std::error::Error>> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) => return Err(format!("File {:?} not found \n {:?}", path, e).into()),
    };
    let reader = BufReader::new(file);

    let mut g = MVCGraph::new();
    let mut exp_edges = 0;
    let mut edges = 0;

    for line in reader.lines() {
        let line = line?;
        let values: Vec<&str> = line.split_whitespace().collect();

        match values[0] {
            "c" => {
                continue;
            }
            "p" => {
                if values[1] != "edge" && values[1] != "col" {
                    // IDK why col but ok
                    return Err("Expecting edge/col format".into());
                }
                let order = values[2].parse::<u64>()?;
                exp_edges = values[3].parse::<u64>()?;
                for i in 0..order {
                    g.add_node(i);
                }
            }
            "e" => {
                if g.order() == 0 {
                    return Err("Expecting graph order".into());
                }
                let i = values[1].parse::<u64>()? - 1;
                let j = values[2].parse::<u64>()? - 1;

                g.add_edge(i, j);
                edges += 1;
            }
            _ => {
                return Err(format!("Invalid file format for line {:?}", line).into());
            }
        }
    }
    if edges != exp_edges {
        return Err(format!("Expecting {} edges but read {} edges", exp_edges, edges).into());
    }
    if g.order() == 0 {
        return Err("Expecting graph order".into());
    }
    Ok(g)
}

/// Structure representing an undirected graph. The graph is represented using a hashmap where a key in the hashmap represent a
/// vertex and the value is a list of neighbors of this vertex.
#[derive(Default)]
pub struct MVCGraph {
    // Hashmap of all the nodes in the graph where map[i] = [j, k, l] means that node i has outgoing edges to nodes j, k, and l
    graph_map: HashMap<u64, Vec<u64>>,
    order: u64,
    size: u64,
}

#[allow(dead_code)]
impl MVCGraph {
    /// Create an empty graph
    pub fn new() -> Self {
        Default::default()
    }

    /// Returns the order (number of nodes) of the graph
    pub fn order(&self) -> u64 {
        self.order
    }

    /// Returns the size (number of edges) of the graph
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: u64) {
        self.graph_map.insert(node, Vec::new());
        self.order += 1;
    }

    /// Add an edge to the graph. The edge (from, to) is added and the edge (to, from) is also added.
    pub fn add_edge(&mut self, from: u64, to: u64) {
        if self.has_edge(from, to) {
            return;
        }
        if !self.has_node(from) {
            self.add_node(from);
        }
        if !self.has_node(to) {
            self.add_node(to);
        }
        self.graph_map.get_mut(&from).unwrap().push(to);
        self.graph_map.get_mut(&to).unwrap().push(from);
        self.size += 1;
    }

    /// Removed an edge from the graph
    pub fn remove_edge(&mut self, from: u64, to: u64) -> Result<u64, Error> {
        if self.graph_map.contains_key(&from) && self.graph_map.contains_key(&to) {
            let edges = self.graph_map.get_mut(&from).unwrap();
            let mut index = 0;
            for edge in edges.iter() {
                if *edge == to {
                    break;
                }
                index += 1;
            }
            edges.remove(index);

            let edges = self.graph_map.get_mut(&to).unwrap();
            let mut index = 0;
            for edge in edges.iter() {
                if *edge == from {
                    break;
                }
                index += 1;
            }
            edges.remove(index);

            self.size -= 1;
            Ok(self.size)
        } else {
            Err(Error)
        }
    }

    /// Remove a node from the graph. Delete the value in the map and remove the value if the neighbor list of all its neighbor
    pub fn remove_node(&mut self, node: u64) {
        if self.graph_map.contains_key(&node) {
            let edges = self.graph_map.get(&node).unwrap().clone();
            for edge in edges.iter() {
                let edges = self.graph_map.get_mut(edge).unwrap();
                let mut index = 0;
                for edge in edges.iter() {
                    if *edge == node {
                        break;
                    }
                    index += 1;
                }
                edges.remove(index);
            }
            self.graph_map.remove(&node);
            self.order -= 1;
            self.size -= edges.len() as u64;
        }
    }

    /// Test if the graph contains the node
    pub fn has_node(&self, node: u64) -> bool {
        self.graph_map.contains_key(&node)
    }

    /// Test if the graph contains the edge (from, to)
    pub fn has_edge(&self, from: u64, to: u64) -> bool {
        if self.graph_map.contains_key(&from) && self.graph_map.contains_key(&to) {
            let edges = self.graph_map.get(&from).unwrap();
            for edge in edges.iter() {
                if *edge == to {
                    return true;
                }
            }
        }
        false
    }

    /// Returns a vector of the neighbors of the node. Returns None if the node is not in the graph.
    pub fn get_neighbors(&self, node: u64) -> Option<&Vec<u64>> {
        self.graph_map.get(&node)
    }

    /// Returns a vector of all the nodes in the graph.
    pub fn get_nodes(&self) -> Vec<u64> {
        let mut nodes = Vec::new();
        for node in self.graph_map.keys() {
            nodes.push(*node);
        }
        nodes
    }

    /// Returns a vector of all the edges (i, j) of the graph.
    pub fn get_edges(&self) -> Vec<(u64, u64)> {
        let mut edges = Vec::new();
        for (node, neighbors) in self.graph_map.iter() {
            for neighbor in neighbors.iter() {
                if !edges.contains(&(*neighbor, *node)) {
                    edges.push((*node, *neighbor));
                }
            }
        }
        edges
    }

    /// Returns the degree of a node. The degree is the number of neighbor of a vertex
    pub fn degree(&self, node: u64) -> Option<u64> {
        if self.graph_map.contains_key(&node) {
            Some(self.graph_map.get(&node).unwrap().len() as u64)
        } else {
            None
        }
    }

    /// Returns the complement of the graph as a new graph.
    pub fn get_complement(&self) -> MVCGraph {
        let mut complement = MVCGraph::new();
        for node in self.get_nodes() {
            complement.add_node(node);
        }
        for node in self.get_nodes() {
            for neighbor in self.get_nodes() {
                if node != neighbor && !self.has_edge(node, neighbor) {
                    complement.add_edge(node, neighbor);
                }
            }
        }
        complement
    }

    /// Test if the vector is a vertex cover of the graph.
    pub fn is_vertex_cover(&self, vertex_cover: &[u64]) -> bool {
        for edge in self.get_edges() {
            if !vertex_cover.contains(&edge.0) && !vertex_cover.contains(&edge.1) {
                return false;
            }
        }
        true
    }
}


// Implement clone for MVCGraph
impl Clone for MVCGraph {
    fn clone(&self) -> MVCGraph {
        MVCGraph {
            graph_map: self.graph_map.clone(),
            order: self.order,
            size: self.size,
        }
    }
}


#[cfg(test)]
mod mvc_test {
    use crate::mvcgraph::MVCGraph;

    #[test]
    fn test_add_node() {
        let mut graph = MVCGraph::new();
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);
        assert_eq!(graph.order(), 3);
        assert_eq!(graph.size(), 0);
    }

    #[test]
    fn test_add_edge() {
        let mut graph = MVCGraph::new();
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 1);
        assert_eq!(graph.order(), 3);
        assert_eq!(graph.size(), 3);
    }

    #[test]
    fn test_remove_edge() {
        let mut graph = MVCGraph::new();
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 1);
        graph.remove_edge(1, 2).unwrap();
        assert_eq!(graph.order(), 3);
        assert_eq!(graph.size(), 2);
        assert!(!graph.has_edge(1, 2));
    }

    #[test]
    fn test_remove_node() {
        let mut graph = MVCGraph::new();
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 1);
        graph.remove_node(1);
        assert_eq!(graph.order(), 2);
        assert_eq!(graph.size(), 1);
        assert!(!graph.has_edge(1, 2));
        assert!(!graph.has_edge(3, 1));
    }

    #[test]
    fn test_remove_keep_values() {
        let mut graph = MVCGraph::new();
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 1);
        graph.remove_node(1);
        assert_eq!(graph.order(), 2);
        assert_eq!(graph.size(), 1);
        assert!(!graph.has_edge(1, 2));
        assert!(!graph.has_edge(3, 1));
        assert!(graph.has_node(2));
        assert!(graph.has_node(3));
    }

    #[test]
    fn test_has_node() {
        let mut graph = MVCGraph::new();
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);
        assert!(graph.has_node(1));
        assert!(graph.has_node(2));
        assert!(graph.has_node(3));
        assert!(!graph.has_node(4));
    }

    #[test]
    fn test_has_edge() {
        let mut graph = MVCGraph::new();
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 1);
        assert!(graph.has_edge(1, 2));
        assert!(graph.has_edge(2, 3));
        assert!(graph.has_edge(3, 1));
        assert!(graph.has_edge(1, 3));
        assert!(graph.has_edge(2, 1));
        assert!(graph.has_edge(3, 2));
        assert!(!graph.has_edge(1, 4));
    }

    #[test]
    fn test_get_neighbors() {
        let mut graph = MVCGraph::new();
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 1);
        let neighbors = graph.get_neighbors(1).unwrap();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&2));
        assert!(neighbors.contains(&3));
    }

    #[test]
    fn test_get_nodes() {
        let mut graph = MVCGraph::new();
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);
        let nodes = graph.get_nodes();
        assert_eq!(nodes.len(), 3);
        assert!(nodes.contains(&1));
        assert!(nodes.contains(&2));
        assert!(nodes.contains(&3));
    }

    #[test]
    fn test_get_edges() {
        let mut graph = MVCGraph::new();
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 1);
        let edges = graph.get_edges();
        assert_eq!(edges.len(), 3);
        assert!(edges.contains(&(1, 2)) || edges.contains(&(2, 1)));
        assert!(edges.contains(&(2, 3)) || edges.contains(&(3, 2)));
        assert!(edges.contains(&(3, 1)) || edges.contains(&(1, 3)));
    }

    #[test]
    fn test_degree() {
        let mut graph = MVCGraph::new();
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 1);
        assert_eq!(graph.degree(1).unwrap(), 2);
        assert_eq!(graph.degree(2).unwrap(), 2);
        assert_eq!(graph.degree(3).unwrap(), 2);
        assert_eq!(graph.degree(4), None);
    }

    #[test]
    fn test_complement() {
        let mut g = MVCGraph::new();
        for i in 0..4 {
            g.add_node(i);
        }
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(2, 3);

        let complement = g.get_complement();
        assert_eq!(complement.size(), 3);
        assert_eq!(complement.order(), 4);
        assert!(complement.has_edge(1, 3));
        assert!(complement.has_edge(1, 2));
        assert!(complement.has_edge(0, 3));
    }

    #[test]
    fn test_is_vertex_cover() {
        let mut graph = MVCGraph::new();
        for i in 0..3 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 0);
        let mut vertex_cover: Vec<u64> = Vec::new();
        vertex_cover.push(0);
        assert!(!graph.is_vertex_cover(&vertex_cover));
        vertex_cover.push(1);
        assert!(graph.is_vertex_cover(&vertex_cover));
        vertex_cover.push(2);
        assert!(graph.is_vertex_cover(&vertex_cover));
    }

    #[test]
    fn test_clone() {
        let mut graph = MVCGraph::new();
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 1);

        let mut graph2 = graph.clone();
        assert!(graph2.has_edge(1, 2));
        assert!(graph2.has_edge(2, 3));
        assert!(graph2.has_edge(3, 1));

        graph2.remove_node(1);
        assert_eq!(graph2.order(), 2);
        assert_eq!(graph2.size(), 1);
        assert_eq!(graph.order(), 3);
        assert_eq!(graph.size(), 3);
    }
}