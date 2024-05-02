use std::fs::read_dir;

use vertex::graph_utils::{add_graph_to_yaml, load_clq_file};

pub fn update_graph_info() {
    let paths = match read_dir("src/resources/graphs") {
        Ok(x) => x,
        Err(e) => {
            println!("Error while reading directory : {}", e);
            return;
        }
    };

    for path in paths {
        let path = path.unwrap().path();
        let path_str = path.to_str().unwrap();

        if path_str.ends_with(".col") || path_str.ends_with(".clq") {
            let graph = match load_clq_file(path_str) {
                Ok(x) => x,
                Err(e) => {
                    println!("Error while loading graph at {:?} : {}", path_str, e);
                    return;
                }
            };
            println!("{}: {} vertices, {} edges", path_str, graph.node_count(), graph.edge_count());
            let result = add_graph_to_yaml(path_str.split('/').last().unwrap(),
                              "clq",
                              &graph,
                              "src/resources/graph_data.yml");
            match result {
                Ok(_) => println!("Graph added to graph_data.yml"),
                Err(e) => println!("Error while adding graph to graph_data.yml : {}", e)
            }
        }
    }
}


fn main() {
    update_graph_info();
}