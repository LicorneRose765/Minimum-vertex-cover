//! Module containing functions to manipulate graphs used in the project.

use std::fmt::Debug;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};

use petgraph::prelude::UnGraphMap;
use serde::{Deserialize, Serialize};
use serde::de::Error;
use serde_yaml::{Sequence, Value};

use crate::ElapseTime;
use crate::errors::{InvalidClqFileFormat, YamlError};

/// Check if a given vertex cover is a vertex cover of a given graph.
///
/// Parse all edges of the graph and check if at least one of the vertices of each edge is in the vertex cover.
/// If not, return false.
///
/// # Example
/// ```rust
/// use petgraph::matrix_graph::MatrixGraph;
/// use petgraph::prelude::UnGraphMap;
/// use petgraph::stable_graph::NodeIndex;
/// use vertex::graph_utils::is_vertex_cover;
///
/// let mut graph = Box::new(UnGraphMap::<u64, ()>::new());
/// for i in 0..3 {
///    graph.add_node(i);
/// }
/// graph.add_edge(0, 1, ());
/// graph.add_edge(1, 2, ());
/// graph.add_edge(2, 0, ());
/// let mut vertex_cover: Vec<u64> = Vec::new();
/// vertex_cover.push(0);
/// assert!(!is_vertex_cover(&graph, &vertex_cover));
/// vertex_cover.push(1);
/// assert!(is_vertex_cover(&graph, &vertex_cover));
/// ```
pub fn is_vertex_cover(graph: &UnGraphMap<u64, ()>, vertex_cover: &[u64]) -> bool {
    for (i, j, _) in graph.all_edges() {
        if !vertex_cover.contains(&(i)) && !vertex_cover.contains(&(j)) {
            return false;
        }
    }
    true
}

/// Check if a given array of vertices is a clique in the given graph.
///
/// # Example
/// ``` rust
/// use petgraph::prelude::UnGraphMap;
/// use vertex::graph_utils::is_clique;
///
/// let mut graph = Box::new(UnGraphMap::<u64, ()>::new());
/// for i in 0..5 {
///   graph.add_node(i);
/// }
/// graph.add_edge(0, 1, ());
/// graph.add_edge(0, 2, ());
/// graph.add_edge(1, 2, ());
///
/// assert!(is_clique(&graph, &vec![0, 1, 2]));
///
/// graph.remove_edge(0, 1);
/// assert!(!is_clique(&graph, &vec![0, 1, 2]));
/// ```
pub fn is_clique(graph: &UnGraphMap<u64, ()>, clique: &Vec<u64>) -> bool {
    for i in clique {
        for j in clique {
            if i != j && !graph.contains_edge(*i, *j) {
                return false;
            }
        }
    }
    true
}

/// Check if a given array of vertices is an independent set in the given graph.
///
/// # Example
/// ```rust
/// use petgraph::prelude::UnGraphMap;
/// use vertex::graph_utils::is_independent_set;
///
/// let mut graph = Box::new(UnGraphMap::<u64, ()>::new());
/// for i in 0..5 {
///  graph.add_node(i);
/// }
/// graph.add_edge(0, 1, ());
/// graph.add_edge(0, 2, ());
/// graph.add_edge(1, 2, ());
///
/// assert!(is_independent_set(&graph, &vec![3, 4]));
/// assert!(!is_independent_set(&graph, &vec![0, 1, 2]));
/// ```
pub fn is_independent_set(graph: &UnGraphMap<u64, ()>, independent_set: &Vec<u64>) -> bool {
    for i in independent_set {
        for j in independent_set {
            if i != j && graph.contains_edge(*i, *j) {
                return false;
            }
        }
    }
    true
}

/// Returns the complement of a given graph.
///
/// # Example
/// ```rust
/// use petgraph::prelude::UnGraphMap;
/// use vertex::graph_utils::complement;
///
/// let mut g = Box::new(UnGraphMap::<u64, ()>::new());
/// for i in 0..4 {
///  g.add_node(i);
/// }
/// g.add_edge(0, 1, ());
/// g.add_edge(1, 2, ());
/// g.add_edge(2, 3, ());
///
/// let complement = complement(&g);
/// assert_eq!(complement.node_count(), 4);
/// assert_eq!(complement.edge_count(), 3);
/// ```
pub fn complement(graph: &UnGraphMap<u64, ()>) -> UnGraphMap<u64, ()> {
    let mut complement = UnGraphMap::<u64, ()>::new();

    for i in 0..graph.node_count() {
        complement.add_node(i as u64);
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

/// Load a graph from a DIMACS .col file.
///
/// The format of the file is the following:
/// * p <#vertex> <#edges> : the number of vertices and edges of the graph
/// * e <vertex1> <vertex2> : an edge between vertex1 and vertex2
/// * c <comment> : a comment
///
/// # Throws
/// InvalidClqFileFormat if the file is not in the correct format.
///
/// # Test file
/// ```text
/// c File: test.clq
/// p edge 5 6
/// e 1 2
/// e 1 3
/// e 1 4
/// e 3 4
/// e 5 1
/// e 5 2
/// ```
///
/// # Example
/// ```rust
/// use vertex::graph_utils::load_clq_file;
///
/// let graph = load_clq_file("src/resources/graphs/test.clq").unwrap();
/// assert_eq!(graph.node_count(), 5);
/// assert!(graph.contains_edge(0, 1));
/// assert!(graph.contains_edge(0, 2));
/// assert!(graph.contains_edge(0, 3));
/// assert!(graph.contains_edge(2, 3));
/// assert!(graph.contains_edge(4, 0));
/// assert!(graph.contains_edge(4, 1));
/// ```
pub fn load_clq_file(path: &str) -> Result<UnGraphMap<u64, ()>, InvalidClqFileFormat> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) => return Err(
            InvalidClqFileFormat::new(&format!("File {:?} not found \n {:?}", path, e))
        ),
    };
    let reader = BufReader::new(file);

    let mut g = UnGraphMap::<u64, ()>::new();
    let mut exp_edges = 0;

    for line in reader.lines() {
        let line = line?;
        let values: Vec<&str> = line.split_whitespace().collect();

        match values[0] {
            "c" => {
                continue;
            }
            "p" => {
                if values[1] != "edge" && values[1] != "col" {
                    return Err(InvalidClqFileFormat::new("Expecting edge/col format"))
                }
                let order = values[2].parse::<u64>()?;
                exp_edges = values[3].parse::<usize>()?;
                for i in 0..order {
                    g.add_node(i);
                }
            }
            "e" => {
                if g.node_count() == 0 {
                    return Err(InvalidClqFileFormat::new("Expecting graph order"));
                }
                let i = values[1].parse::<u64>()? - 1;
                let j = values[2].parse::<u64>()? - 1;

                g.add_edge(i, j, ());
            }
            _ => {
                return Err(InvalidClqFileFormat::new(&format!("Invalid file format for line {:?}", line)));
            }
        }
    }
    if g.edge_count() != exp_edges {
        return Err(InvalidClqFileFormat::new(&format!("Expecting {} edges but read {} edges", exp_edges, g.edge_count())));
    }
    if g.node_count() == 0 {
        return Err(InvalidClqFileFormat::new("Expecting graph order"));
    }
    Ok(g)
}

/// Returns the string of a given file in the DIMACS .clq format.
///
/// # Example
/// ```rust
/// use petgraph::prelude::UnGraphMap;
/// use vertex::graph_utils::graph_to_string;
///
/// let mut graph = Box::new(UnGraphMap::<u64, ()>::new());
/// for i in 0..4 {
///     graph.add_node(i);
/// }
/// graph.add_edge(0, 1, ());
/// graph.add_edge(1, 2, ());
///
/// let string = graph_to_string(&graph);
/// assert_eq!(string, "p edge 4 2\ne 1 2\ne 2 3\n");
/// ```
pub fn graph_to_string(graph: &UnGraphMap<u64, ()>) -> String {
    let mut string = String::new();
    string.push_str(&format!("p edge {} {}\n", graph.node_count(), graph.edge_count()));
    for (i, j, _) in graph.all_edges() {
        string.push_str(&format!("e {} {}\n", i + 1, j + 1));
    }
    string
}

/// Returns the vertex with the maximum degree in the graph and its degree.
///
/// # Example
/// ```rust
/// use petgraph::prelude::UnGraphMap;
/// use vertex::graph_utils::get_vertex_with_max_degree;
///
/// let mut graph = Box::new(UnGraphMap::<u64, ()>::new());
/// for i in 0..10 {
///    graph.add_node(i);
/// }
/// for i in 0..9 {
///   graph.add_edge(i, i+1, ());
/// }
/// graph.add_edge(0, 9, ());
/// graph.add_edge(0, 8, ());
///
/// assert_eq!(get_vertex_with_max_degree(&graph, None).0, 0);
/// assert_eq!(get_vertex_with_max_degree(&graph, None).1, 3);
/// ```
pub fn get_vertex_with_max_degree(graph: &UnGraphMap<u64, ()>, marked_vertices: Option<&Vec<u64>>) -> (u64, usize) {
    let mut max_degree = 0;
    let mut max_degree_vertex = 0;
    for vertex in graph.nodes() {
        if marked_vertices.is_some() && marked_vertices.unwrap().contains(&vertex) {
            continue;
        }
        let degree = graph.neighbors(vertex).count();
        if degree > max_degree {
            max_degree = degree;
            max_degree_vertex = vertex;
        }
    }
    (max_degree_vertex, max_degree)
}

/// Since clone is not implemented for MatrixGraph, this function manually copies the graph.
/// It iterates over the nodes and edges of the graph and adds them to the copy.
///
/// # Example
/// ```rust
/// use petgraph::prelude::UnGraphMap;
///
/// use vertex::graph_utils::copy_graph;
///
/// let mut graph = Box::new(UnGraphMap::<u64, ()>::new());
/// for i in 0..10 {
///   graph.add_node(i);
/// }
/// for i in 0..9 {
///  graph.add_edge(i, i+1, ());
/// }
///
/// let copy = copy_graph(&graph);
/// assert_eq!(copy.node_count(), 10);
/// assert_eq!(copy.edge_count(), 9);
/// ```
pub fn copy_graph(graph: &UnGraphMap<u64, ()>) -> UnGraphMap<u64, ()> {
    let mut copy = UnGraphMap::<u64, ()>::new();
    for i in graph.nodes() {
        copy.add_node(i);
    }
    for edge in graph.all_edges() {
        copy.add_edge(edge.0, edge.1, ());
    }
    copy
}

/// Structure used to store the information of a graph such as its exact value of the MVC.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphInfo {
    pub id: String,
    format: String,
    order: usize,
    size: usize,
    val: u64,
}

/// Structure used to store the information of a computation of the MVC for a given graph.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct YamlTime {
    date: String,
    mvc_val: u64,
    time: String,
    is_time_limit: bool,
    algorithm: String,
    comment: String,
}

/// Add the graph id with its format in the yaml file located at src/resources/graph_data.yml.
///
/// The default value for mvc_val is 0, it has to be updated manually.
/// If the graph id is already in the file, it is not added again.
///
/// # Throws
/// - YamlError::IoError if the file cannot be opened or written
/// - YamlError::YAMLFormatError if the file is not in the correct format
/// - YamlError::YamlParsingError if there is an error while parsing the file
pub fn add_graph_to_yaml(id: &str, format: &str, graph: &UnGraphMap<u64, ()>, path: &str)
                         -> Result<(), YamlError> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) => return Err(YamlError::IoError(format!("unable to open file {:?}", path), e))
    };
    let mut data: Vec<GraphInfo> = serde_yaml::from_reader(file)?;

    if data.iter().any(|x| x.id == id) {
        // If the graph is already in the file, we don't add it again
        return Ok(());
    }

    let info = GraphInfo {
        id: id.to_string(),
        format: format.to_string(),
        order: graph.node_count(),
        size: graph.edge_count(),
        val: 0,
    };
    data.push(info);

    // Update the file
    let mut file = File::create(path)?;
    let tmp = serde_yaml::to_string(&data)?;
    file.write_all(tmp.as_bytes())?;

    // When we add a graph to the yaml file, we also add it to the time file
    add_graph_to_time_file(id)
}

fn add_graph_to_time_file(id: &str) -> Result<(), YamlError> {
    let time_path = "src/resources/time_result.yml";
    let mut file = match File::open(time_path) {
        Ok(file) => file,
        Err(e) => return Err(YamlError::IoError(format!("unable to open file {:?}", time_path), e))
    };
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let time: Value = serde_yaml::from_str(&contents)?;
    let mut map = match time.as_mapping() {
        Some(x) => x.clone(),
        None => return Err(YamlError::YAMLFormatError("File badly formatted".to_string(),
                                                      serde_yaml::Error::custom("The content of the file should be a map"))),
    };

    let vec: Sequence = Vec::new();

    map.insert(Value::String(id.to_string()), Value::Sequence(vec));

    let mut file = File::create(time_path)?;
    serde_yaml::to_writer(&mut file, &map)?;
    Ok(())
}

/// Update the known value of the minimum vertex cover for a given graph id.
///
/// # Parameters
/// - id : the id of the graph (ex: test.clq)
/// - mvc_val : the new value of the minimum vertex cover
/// - path : the path to the yaml file containing the graph info (optional-> None or Some(path))
///
/// # Throws
/// - YamlError::IoError if the file cannot be opened or written
/// - YamlError::YAMLParsingError if there is an error while parsing the file
/// - YamlError::NotFound if the graph id is not in the file
///
/// # example
/// ```
/// use vertex::errors::YamlError;
/// use vertex::graph_utils::update_mvc_value;
///
/// let res = match update_mvc_value("test.clq", 2, Some("src/resources/graph_data.yml")) {
///     Ok(_) => {println!("The value of the minimum vertex cover for the test.clq graph is now 2");},
///     Err(e) => {println!("Error: {:?}", e);},
/// };
///
/// update_mvc_value("test.clq", 3, None).expect("Error while updating the value of the minimum vertex cover");
/// // The value of the minimum vertex cover for the test.clq graph is now 3
/// ```
pub fn update_mvc_value(id: &str, mvc_val: u64, path: Option<&str>) -> Result<(), YamlError> {
    // TODO : this function has to be deleted later on
    let path = path.unwrap_or("src/resources/graph_data.yml");
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) => return Err(YamlError::IoError(format!("unable to open file {:?}", path), e))
    };

    let mut data: Vec<GraphInfo> = serde_yaml::from_reader(file)?;

    let mut found = false;
    for info in data.iter_mut() {
        if info.id == id {
            info.val = mvc_val;
            found = true;
            break;
        }
    }
    if !found {
        return Err(YamlError::NotFound(
            format!("Graph {:?} not found in the YAML file", id),
            format!("Graph {:?} not found in {:?} to store the mvc : {:?}", id, path, mvc_val)));
    }

    // Update the file
    let mut file = File::create(path)?;
    let tmp = serde_yaml::to_string(&data)?;
    file.write_all(tmp.as_bytes())?;
    Ok(())
}

/// Check if a given value is the optimal value for a given graph id. If the graph id is not in the file, return None.
///
/// The optimal value is the value stored in the yaml file. So, if the value in the yaml file is wrong, this function will return the wrong result.
///
/// # Parameters
/// - id : the id of the graph (ex: test.clq)
/// - val : the value to check
/// - path : the path to the yaml file containing the graph info (optional-> None or Some(path))
///
/// # Throws
/// - YamlError::IoError if the file cannot be opened or is not found
/// - YamlError::YAMLParsingError if there is an error while parsing the file
///
/// # example
/// ```
/// use vertex::graph_utils::is_optimal_value;
///
/// let true_res = is_optimal_value("test.clq", 3, Some("src/resources/graph_data.yml")).unwrap_or_else(|e| {
///    panic!("Error while checking if value is optimal : {}", e);
/// });
/// assert!(true_res.unwrap());
///
/// let false_res = is_optimal_value("test.clq", 2, Some("src/resources/graph_data.yml")).unwrap_or_else(|e| {
/// panic!("Error while checking if value is optimal : {}", e);
/// });
/// assert!(!false_res.unwrap());
/// ```
pub fn is_optimal_value(id: &str, val: u64, path: Option<&str>) -> Result<Option<bool>, YamlError> {
    let path = path.unwrap_or("src/resources/graph_data.yml");
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) => return Err(YamlError::IoError(format!("unable to open file {:?}", path), e))
    };

    let data: Vec<GraphInfo> = serde_yaml::from_reader(file)?;

    for info in data.iter() {
        if info.id == id {
            return if info.val == val {
                Ok(Some(true))
            } else {
                Ok(Some(false))
            };
        }
    }
    Ok(None)
}

/// Get the optimal value for a given graph id.
/// The optimal value is the value stored in the yaml file. So, if the value in the yaml file is wrong,
/// this function will return the wrong result.
///
/// # Parameters
/// - id : the id of the graph (ex: test.clq)
/// - path : the path to the yaml file containing the graph info (optional-> None or Some(path))
///
/// # Throws
/// - YamlError::IoError if the file cannot be opened or is not found
/// - YamlError::YAMLParsingError if there is an error while parsing the file
///
/// # example
/// ```
/// use vertex::graph_utils::get_optimal_value;
///
/// let res = get_optimal_value("test.clq", Some("src/resources/graph_data.yml")).unwrap_or_else(|e| {
///   panic!("Error while getting the optimal value : {}", e);
/// });
/// assert_eq!(res, Some(3));
///
/// let none_res = get_optimal_value("unknown_graph.clq", Some("src/resources/graph_data.yml")).unwrap_or_else(|e| {
///  panic!("Error while getting the optimal value : {}", e);
/// });
/// assert_eq!(none_res, None);
///
/// ```
pub fn get_optimal_value(id: &str, path: Option<&str>) -> Result<Option<u64>, YamlError> {
    let path = path.unwrap_or("src/resources/graph_data.yml");
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) => return Err(YamlError::IoError(format!("unable to open file {:?}", path), e))
    };

    let data: Vec<GraphInfo> = serde_yaml::from_reader(file)?;

    for info in data.iter() {
        if info.id == id {
            return Ok(Some(info.val));
        }
    }
    Ok(None)
}

/// Adds a new time for the given graph to the yaml file located at src/resources/time_result.yml.
///
/// # Parameters
/// - id : the id of the graph (ex: test.clq)
/// - mvc_val : the value of the minimum vertex cover
/// - time : the time taken to compute the mvc
/// - is_time_limit : true if the time limit was reached, false otherwise
/// - algorithm : the algorithm used to compute the mvc
/// - comment : a comment about the computation
///
/// # Throws
/// - YamlError::IoError if the file cannot be opened, is not found or cannot be written
/// - YamlError::YAMLParsingError if there is an error while parsing the file
/// - YamlError::NotFound if the graph id is not in the file
/// - YamlError::YAMLFormatError if the file is not in the correct format (should be a map / content of a graph should be a vector)
///
pub fn add_time_to_yaml(id: &str, mvc_val: u64, time: ElapseTime, is_time_limit: bool, algorithm: &str, comment: &str)
                        -> Result<(), YamlError> {
    let path = "src/resources/time_result.yml";
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(e) => return Err(YamlError::IoError(format!("unable to open file {:?}", path), e))
    };
    let mut content = String::new();
    file.read_to_string(&mut content)?;

    let content: Value = serde_yaml::from_str(&content)?;
    let mut map = match content.as_mapping() {
        Some(x) => x.clone(),
        None => return Err(YamlError::YAMLFormatError("File badly formatted".to_string(),
                                                      serde_yaml::Error::custom("The content of the file should be a map"))),
    };

    let graph = match map.get(id) {
        Some(graph) => graph.clone(),
        None => return Err(YamlError::NotFound(
            format!("Graph {:?} not found in the YAML file", id),
            format!("Graph {:?} not found in {:?} to store the mvc : {:?}", id, path, mvc_val))),
    };

    let mut graph_data = get_graph_data(graph)?;

    let new_time = YamlTime {
        date: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        mvc_val,
        time: time.to_string(),
        is_time_limit,
        algorithm: algorithm.to_string(),
        comment: comment.to_string(),
    };

    let as_value = serde_yaml::to_value(new_time)?;
    graph_data.push(as_value);

    map.insert(Value::String(id.to_string()), Value::Sequence(graph_data));


    // Update the file
    let mut file = File::create(path)?;
    serde_yaml::to_writer(&mut file, &map)?;
    Ok(())
}

/// Get all the times for a given graph id.
///
/// # Parameters
/// - id : the id of the graph (ex: test.clq)
///
/// # Throws
/// - YamlError::IoError if the file cannot be opened or is not found
/// - YamlError::YAMLParsingError if there is an error while parsing the file
/// - YamlError::NotFound if the graph id is not in the file
/// - YamlError::YAMLFormatError if the file is not in the correct format.
///
pub fn get_time_data(id: &str) -> Result<Vec<YamlTime>, YamlError> {
    let path = "src/resources/time_result.yml";
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(e) => return Err(YamlError::IoError(format!("unable to open file {:?}", path), e))
    };
    let mut content = String::new();
    file.read_to_string(&mut content)?;

    let content: Value = serde_yaml::from_str(&content)?;
    let map = match content.as_mapping() {
        Some(x) => x.clone(),
        None => return Err(YamlError::YAMLFormatError("File badly formatted".to_string(),
                                                      serde_yaml::Error::custom("The content of the file should be a map"))),
    };

    let graph = match map.get(id) {
        Some(graph) => graph.clone(),
        None => return Err(YamlError::NotFound(
            format!("Graph {:?} not found in the YAML file", id),
            format!("Graph {:?} not found in {:?} to store the time", id, path))),
    };

    let graph_data = get_graph_data(graph)?;
    let mut res: Vec<YamlTime> = Vec::new();

    for time in graph_data.iter() {
        let time: YamlTime = match serde_yaml::from_value(time.clone()) {
            Ok(x) => x,
            Err(e) => return Err(YamlError::YAMLFormatError(
                "File badly formatted".to_string(), serde_yaml::Error::custom(format!("the content of the graph should be a vector: {:?}", e))))
        };
        res.push(time);
    }
    Ok(res)
}

fn get_graph_data(id: Value) -> Result<Sequence, YamlError> {
    let res = match serde_yaml::from_value(id) {
        Ok(x) => x,
        Err(e) => return Err(YamlError::YAMLFormatError(
            "File badly formatted".to_string(), serde_yaml::Error::custom(format!("the content of the graph should be a vector: {:?}", e))))
    };
    Ok(res)
}

#[cfg(test)]
mod graph_utils_tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_is_vertex_cover() {
        let mut graph = Box::new(UnGraphMap::<u64, ()>::new());
        for i in 0..3 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, ());
        graph.add_edge(1, 2, ());
        graph.add_edge(2, 0, ());
        let mut vertex_cover: Vec<u64> = Vec::new();
        vertex_cover.push(0);
        assert!(!is_vertex_cover(&graph, &vertex_cover));
        vertex_cover.push(1);
        assert!(is_vertex_cover(&graph, &vertex_cover));
        vertex_cover.push(2);
        assert!(is_vertex_cover(&graph, &vertex_cover));
    }

    #[test]
    fn test_is_clique() {
        let mut graph = Box::new(UnGraphMap::<u64, ()>::new());
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, ());
        graph.add_edge(0, 2, ());
        graph.add_edge(1, 2, ());

        assert!(is_clique(&graph, &vec![0, 1, 2]));

        graph.remove_edge(0, 1);
        assert!(!is_clique(&graph, &vec![0, 1, 2]));
    }

    #[test]
    fn test_is_independent_set() {
        let mut graph = Box::new(UnGraphMap::<u64, ()>::new());
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, ());
        graph.add_edge(0, 2, ());
        graph.add_edge(1, 2, ());

        assert!(is_independent_set(&graph, &vec![3, 4]));
        assert!(!is_independent_set(&graph, &vec![0, 1, 2]));
    }

    #[test]
    fn test_get_vertex_with_max_degree() {
        let mut graph = Box::new(UnGraphMap::<u64, ()>::new());
        for i in 0..10 {
            graph.add_node(i);
        }
        for i in 0..9 {
            graph.add_edge(i, i + 1, ());
        }
        graph.add_edge(0, 9, ());
        graph.add_edge(0, 8, ());
        graph.add_edge(0, 7, ());
        assert_eq!(get_vertex_with_max_degree(&graph, None).0, 0);
        assert_eq!(get_vertex_with_max_degree(&graph, None).1, 4);
    }

    #[test]
    fn test_get_vertex_with_max_degree2() {
        let mut graph = Box::new(UnGraphMap::<u64, ()>::new());
        for i in 3..7 {
            graph.add_node(i);
        }
        graph.add_edge(3, 4, ());
        graph.add_edge(4, 5, ());
        graph.add_edge(5, 6, ());

        assert_eq!(get_vertex_with_max_degree(&graph, None).0, 4);
        assert_eq!(get_vertex_with_max_degree(&graph, None).1, 2);
    }

    #[test]
    fn test_copy_graph() {
        let mut graph = Box::new(UnGraphMap::<u64, ()>::new());
        for i in 0..10 {
            graph.add_node(i);
        }
        for i in 0..9 {
            graph.add_edge(i, i + 1, ());
        }

        let mut copy = copy_graph(&graph);

        assert_eq!(copy.node_count(), graph.node_count());
        for i in 0..9 {
            assert!(copy.contains_edge(i, i + 1));
            assert!(graph.contains_edge(i, i + 1));
        }

        graph.remove_edge(0, 1);
        assert_eq!(graph.edge_count(), 8);
        assert_eq!(copy.edge_count(), 9);

        graph.remove_node(0);
        assert_eq!(graph.node_count(), 9);
        assert_eq!(copy.node_count(), 10);

        copy.remove_node(0);
        copy.remove_node(1);
        assert_eq!(copy.node_count(), 8);
        assert_eq!(graph.node_count(), 9);
    }

    #[test]
    fn test_complement() {
        let mut g = Box::new(UnGraphMap::<u64, ()>::new());
        for i in 0..4 {
            g.add_node(i);
        }
        g.add_edge(0, 1, ());
        g.add_edge(0, 2, ());
        g.add_edge(0, 3, ());
        g.add_edge(2, 3, ());

        let complement = complement(&g);
        assert_eq!(complement.edge_count(), 2);
        assert_eq!(complement.node_count(), 4);
        assert!(complement.contains_edge(1, 3));
        assert!(complement.contains_edge(1, 2));
    }

    #[test]
    fn test_load_clq_file() {
        let graph = load_clq_file("src/resources/graphs/test.clq").unwrap();
        assert_eq!(graph.node_count(), 5);
        assert!(graph.contains_edge(0, 1));
        assert!(graph.contains_edge(0, 2));
        assert!(graph.contains_edge(0, 3));
        assert!(graph.contains_edge(2, 3));
        assert!(graph.contains_edge(4, 0));
        assert!(graph.contains_edge(4, 1));
    }

    #[test]
    fn test_load_clq_should_throw_when_file_not_found() {
        let filename = "unknown_file.clq";
        let result = load_clq_file(filename);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_clq_should_throw_when_file_not_in_correct_format() {
        // The file is not in the correct format
        let filename = "src/resources/tests/graphs/format_edge.clq";
        let result = load_clq_file(filename);
        assert!(result.is_err());
        let expected = "Expecting edge/col format";
        assert_eq!(result.unwrap_err().message, expected);
    }

    #[test]
    fn test_load_clq_throw_when_order_missing() {
        // Defined an edge before the graph order
        let filename = "src/resources/tests/graphs/format_order.clq";
        let result = load_clq_file(filename);
        assert!(result.is_err());
        let expected = "Expecting graph order";
        assert_eq!(result.unwrap_err().message, expected);
    }

    #[test]
    fn test_load_clq_throw_when_wrong_size() {
        // The file has a wrong number of edges
        let filename = "src/resources/tests/graphs/format_size.clq";
        let result = load_clq_file(filename);
        assert!(result.is_err());
        let expected = "Expecting 1 edges but read 2 edges";
        assert_eq!(result.unwrap_err().message, expected);
    }

    #[test]
    fn test_load_clq_throw_when_wrong_symbol() {
        // The file has a wrong symbol
        let filename = "src/resources/tests/graphs/format_symbol.clq";
        let result = load_clq_file(filename);
        assert!(result.is_err());
        let expected = "Invalid file format for line \"a 1 2\"";
        assert_eq!(result.unwrap_err().message, expected);
    }

    #[test]
    fn test_graph_to_string() {
        let mut graph = Box::new(UnGraphMap::<u64, ()>::new());
        for i in 0..4 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, ());
        graph.add_edge(1, 2, ());

        let string = graph_to_string(&graph);
        assert_eq!(string, "p edge 4 2\ne 1 2\ne 2 3\n");
    }

    // ========== ADD GRAPH TO YAML ==========
    #[test]
    fn test_add_graph_to_yaml_file_not_found() {
        let result = add_graph_to_yaml("test.clq",
                                       "edge",
                                       &UnGraphMap::<u64, ()>::new(),
                                       "unknown_file.yml");
        assert!(result.is_err());
        // Check error types is IoError
        assert!(matches!(result.unwrap_err(), YamlError::IoError(_, _)));
    }

    // ======= UPDATE MVC VALUE =========
    #[test]
    fn test_update_mvc_value_file_not_found() {
        let result = update_mvc_value("test.clq", 2, Some("unknown_file.yml"));
        assert!(result.is_err());
        // Check error types is IoError
        assert!(matches!(result.unwrap_err(), YamlError::IoError(_, _)));
    }

    #[test]
    fn test_update_mvc_value_graph_not_found() {
        let result = update_mvc_value("unknown_graph.clq", 2, None);
        assert!(result.is_err());
        // Check error types is NotFound
        assert!(matches!(result.unwrap_err(), YamlError::NotFound(_, _)));
    }

    // ======= IS OPTIMAL VALUE =========
    #[test]
    fn test_is_optimal_value_file_not_found() {
        let result = is_optimal_value("test.clq", 3, Some("unknown_file.yml"));
        assert!(result.is_err());
        // Check error types is IoError
        assert!(matches!(result.unwrap_err(), YamlError::IoError(_, _)));
    }

    #[test]
    fn test_is_optimal_value() {
        let result = is_optimal_value("test.clq", 3, Some("src/resources/graph_data.yml"));
        assert!(result.is_ok());
        assert!(result.unwrap().unwrap());
    }

    // ======= GET OPTIMAL VALUE =========
    #[test]
    fn test_get_optimal_value_file_not_found() {
        let result = get_optimal_value("test.clq", Some("unknown_file.yml"));
        assert!(result.is_err());
        // Check error types is IoError
        assert!(matches!(result.unwrap_err(), YamlError::IoError(_, _)));
    }

    #[test]
    fn test_get_optimal_value() {
        let result = get_optimal_value("test.clq", Some("src/resources/graph_data.yml"));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().unwrap(), 3);
    }


    // ======= ADD TIME TO YAML =========
    #[test]
    fn test_add_time_to_yaml_graph_not_found() {
        let result = add_time_to_yaml("unknown_graph.clq", 2, ElapseTime::new(Duration::new(0, 0)), false, "algo", "comment");
        assert!(result.is_err());
        // Check error types is NotFound
        assert!(matches!(result.unwrap_err(), YamlError::NotFound(_, _)));
    }

    // ======= GET TIME DATA =========
    #[test]
    fn test_get_time_data_graph_not_found() {
        let result = get_time_data("unknown_graph.clq");
        assert!(result.is_err());
        // Check error types is NotFound
        assert!(matches!(result.unwrap_err(), YamlError::NotFound(_, _)));
    }
}