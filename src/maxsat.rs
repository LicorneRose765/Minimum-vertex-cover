//! Module containing the structures and algorithms for the satLB lower bound of the BnB algorithm.
//! This module can only be used in this context. Using the methods outside this algorithm may lead to unexpected results.
//! 
//! # Example
//! Here is an example of the usage of this module : 
//! ```rust
//! use petgraph::prelude::UnGraphMap;
//! use vertex::maxsat::encode_maxsat;
//!
//! let mut graph = UnGraphMap::<u64, ()>::new();
//! for i in 0..4 {
//!    graph.add_node(i);
//! }
//! graph.add_edge(0, 1, ());
//! graph.add_edge(1, 2, ());
//! graph.add_edge(2, 0, ());
//! graph.add_edge(2, 3, ());
//!
//! let cliques = vec![vec![0, 1, 2], vec![3]];
//!
//! let mut formula = encode_maxsat(&graph, cliques);
//! 
//! let nbr_of_inconsistent_subsets = formula.find_inconsistent_subsets();
//!```
use std::collections::{HashSet, VecDeque};
use std::hash::Hash;
use petgraph::graphmap::UnGraphMap;

/// A literal is a variable or its negation.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
enum Literal {
    Positive(u64),
    Negative(u64),
}

impl Literal {
    /// Return the negation of the literal.
    pub fn negation(&self) -> Self {
        match self {
            Literal::Positive(v) => Literal::Negative(*v),
            Literal::Negative(v) => Literal::Positive(*v),
        }
    }
}

/// Structure representing a clause in the MaxSat instance. A clause is a disjunction of literals. (l_1 OR l_2 OR ... OR l_n)
/// 
/// A clause can be a hard clause or a soft clause. Hard clauses must be satisfied while soft clauses can be satisfied.
/// Each clause has an id that is unique in the instance. Hard clauses have positive ids while soft clauses have negative ids.
#[derive(Debug, Default, PartialEq, Eq)]
struct Clause {
    literals: HashSet<Literal>,
    id: i64,
}

impl Clause {
    pub fn new() -> Self {
        Default::default()
    }
    
    /// Returns the number of literals in the clause.
    pub fn size(&self) -> usize {
        self.literals.len()
    }
    
    /// Returns the first literal in the clause. (Used to get the unit literal)
    pub fn get_first_literal(&self) -> Option<&Literal> {
        self.literals.iter().next()
    }
    
    /// Returns the set of literals in the clause.
    pub fn get_literals(&self) -> &HashSet<Literal> {
        &self.literals
    }
    
    /// Adds a literal to the clause.
    pub fn add_literal(&mut self, literal: Literal) {
        self.literals.insert(literal);
    }
    
    /// Removes a literal from the clause.
    pub fn remove_literal(&mut self, literal: &Literal) {
        self.literals.remove(literal);
    }
    
    /// Returns true if the clause is empty.
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }
    
    /// Returns true if the clause is a unit clause.
    pub fn is_unit(&self) -> bool {
        self.size() == 1
    }
    
    /// Get the id of the clause.
    pub fn get_id(&self) -> i64 {
        self.id
    }
    
    /// Set the id of the clause.
    pub fn set_id(&mut self, id: i64) {
        self.id = id;
    }
    
    /// Returns true if the clause is a soft clause. (Its id is negative)
    pub fn is_soft(&self) -> bool {
        self.id < 0
    }
    
    /// Returns true if the clause contains the given literal.
    pub fn contains(&self, literal: &Literal) -> bool {
        self.literals.contains(literal)
    }
}

impl Clone for Clause {
    fn clone(&self) -> Self {
        let mut new_clause = Clause::new();
        for literal in &self.literals {
            new_clause.add_literal(literal.clone());
        }
        new_clause.set_id(self.id);
        new_clause
    }
}

#[derive(Debug, Default)]
pub struct MaxSat {
    hard_clauses: Vec<Clause>,
    soft_clauses: Vec<Clause>,
}

impl MaxSat{
    pub fn new() -> Self {
        Default::default()
    }
    
    /// Removes a literal l from the hard clause at index idx and returns true if the clause becomes empty.
    fn remove_literal_from_hard_clause_idx(&mut self, idx: usize, literal: &Literal) -> bool {
        let mut clause = self.hard_clauses[idx].clone();
        clause.remove_literal(literal);
        let res = clause.is_empty();
        self.hard_clauses[idx] = clause;
        
        res
    }
    
    /// Removes a literal l from the soft clause at index idx and returns true if the clause becomes empty.
    fn remove_literal_from_soft_clause_idx(&mut self, idx: usize, literal: &Literal) -> bool {
        let mut clause = self.soft_clauses[idx].clone();
        clause.remove_literal(literal);
        let res = clause.is_empty();
        if res {
            self.soft_clauses.remove(idx);
        } else {
            self.soft_clauses[idx] = clause;
        }
        res
    }
    
    /// Returns the hard clause with the given index in the list of hard clauses.
    fn get_hard_clause_by_idx(&self, idx: usize) -> Option<&Clause> {
        self.hard_clauses.get(idx)
    }
    
    /// Returns the soft clause with the given index in the list of soft clauses.
    fn get_soft_clause_by_idx(&self, idx: usize) -> Option<&Clause> {
        self.soft_clauses.get(idx)
    }
    
    /// Returns the soft clause with the given id.
    fn get_soft_clause_by_id(&self, id: i64) -> Option<&Clause> {
        self.soft_clauses.iter().find(|c| c.get_id() == id)
    }
    
    /// Returns the soft clauses of the MaxSat instance.
    fn get_soft_clauses(&self) -> &Vec<Clause> {
        &self.soft_clauses
    }
    
    /// Adds a hard clause to the MaxSat instance.
    fn add_hard_clause(&mut self, clause: Clause) {
        let mut clause = clause;
        clause.set_id(self.hard_clauses.len() as i64 + 1);
        self.hard_clauses.push(clause);
    }
    
    /// Removes a hard clause from the MaxSat instance.
    fn remove_hard_clause(&mut self, clause: &Clause) {
        self.hard_clauses.retain(|c| c != clause);
    }
    
    /// Adds a soft clause to the MaxSat instance.
    fn add_soft_clause(&mut self, clause: Clause) {
        let mut clause = clause;
        clause.set_id(-(self.soft_clauses.len() as i64 + 1));
        self.soft_clauses.push(clause);
    }
    
    /// Removes a soft clause from the MaxSat instance.
    fn remove_soft_clause(&mut self, clause: &Clause) {
        self.soft_clauses.retain(|c| c != clause);
    }
    
    /// Returns the number of hard clauses.
    #[allow(dead_code)]
    fn num_hard_clauses(&self) -> usize {
        self.hard_clauses.len()
    }
    
    /// Returns the number of soft clauses.
    #[allow(dead_code)]
    fn num_soft_clauses(&self) -> usize {
        self.soft_clauses.len()
    }
    
    /// Find the number of inconsistent subsets of soft clauses
    /// 
    /// This method is used in the satLB lower bound of the BnB algorithm. 
    /// It can **only** be used on a MaxSat instance that has been encoded from a graph.
    pub fn find_inconsistent_subsets(&mut self) -> u64 {
        // Find the number of inconsistent subsets of soft clauses.
        let mut s = 0;
        let mut non_tested_soft_clauses = self.get_soft_clauses().clone();
        non_tested_soft_clauses.sort_by_key(|x| std::cmp::Reverse(x.size()));

        while let Some(clause) = non_tested_soft_clauses.pop() {
            // Parse all clause and check if all their literals are failed
            
            let mut clause_to_rm: Vec<Clause> = vec![clause.clone()];
            let mut are_all_failed = true;
            for l in clause.get_literals() {
                // Check if all the literals in the clause are failed
                let (is_failed, clauses) = self.is_failed_literal(l);
                if !is_failed {
                    are_all_failed = false;
                    break;
                }
                for c in clauses {
                    // Add the clause used to make the literal failed to the list of clauses to remove
                    let clause = self.get_soft_clause_by_id(c).unwrap().clone();
                    if !clause_to_rm.contains(&clause) {
                        clause_to_rm.push(clause);
                    }
                }
            }
            if are_all_failed {
                s += 1;
                // Remove the clause from the list of non-tested clauses
                self.remove_soft_clause(&clause);
                for c in clause_to_rm {
                    self.remove_soft_clause(&c);
                    non_tested_soft_clauses.retain(|x| x != &c);
                }
            }
        }
        
        s
    }

    /// Remove the negation of the literal from the hard clauses.
    fn simplify(&mut self, literal: &Literal) -> bool {
        let hard = self.hard_clauses.clone();

        for (i, clause) in hard.iter().enumerate() {
            if clause.contains(&literal.negation()) && self.remove_literal_from_hard_clause_idx(i, &literal.negation()) {
                return true;
            }
        }
        false
    }
    
    /// Check if the given literal is a failed literal (Only for literals in soft clause).
    /// 
    /// Returns a tuple (is_failed, soft_clauses) where is_failed is a boolean indicating if the literal is failed
    /// and soft_clauses is a list of the soft clauses that make the literal failed.
    /// If the literal is not failed, the list of soft clauses is empty.
    /// 
    /// If the literal is failed but directly when simplifying the formula, the list of soft clauses is empty.
    fn is_failed_literal(&self, literal: &Literal) -> (bool, HashSet<i64>) {
        // Copy the formula
        let hard = self.hard_clauses.clone();
        let mut soft = self.soft_clauses.clone();
        // Remove the clause where the literal is true, this clause is already satisfied
        soft.retain(|c| !c.contains(literal));
        let mut phi = MaxSat {
            hard_clauses: hard,
            soft_clauses: soft,
        };

        // Assign the literal to true & simplify the formula => remove all not(literal) from hard clause
        let is_already_failed = phi.simplify(literal);
        if is_already_failed {
            return (true, HashSet::new());
        }

        phi.unit_propagation()
    }
    
    /// Perform unit propagation. 
    /// 
    /// If the propagation leads to an inconsistency, the method returns true and the set of soft clauses that have been used.
    fn unit_propagation(&mut self) -> (bool, HashSet<i64>) {
        let mut used_soft_clauses: HashSet<i64> = HashSet::new(); // Store the id of the soft clauses that have been propagated
        
        // Get all unit clauses from the hard and soft clauses.
        let mut unit_clauses = VecDeque::new();
        for clause in &self.hard_clauses {
            if clause.is_unit() {
                unit_clauses.push_back(clause.clone());
            }
        }
        for clause in &self.soft_clauses {
            if clause.is_unit() {
                unit_clauses.push_back(clause.clone());
            }
        }
        
        
        // Perform unit propagation.
        while let Some(clause) = unit_clauses.pop_front() {
            if clause.is_soft() {
                used_soft_clauses.insert(clause.get_id());
            }
            if self.propagate(&mut unit_clauses, &clause, &mut used_soft_clauses) {
                // The formula becomes inconsistent.
                return (true, used_soft_clauses);
            }
        }
        (false, used_soft_clauses)
    }

    /// Propagates the given literal and returns true if the formula becomes inconsistent.
    /// 
    /// # Params
    /// * `unit_stack` - The stack of unit clauses.
    /// * `clause` - The clause to propagate.
    /// * `soft_clause_used` - The set of soft clauses that have been used. (To add the new ones)
    /// 
    /// # Returns
    /// True if the formula becomes inconsistent.
    fn propagate(&mut self, unit_stack: &mut VecDeque<Clause>, clause: &Clause, soft_clause_used: &mut HashSet<i64>) -> bool {
        // Propagate the unit clause.
        let literal = clause.get_first_literal().unwrap();
        let nliteral = literal.negation();
        
        // For each clause : 
        // 1. If the clause contains the literal, remove the clause.
        // 2. If the clause contains the negation of the literal, remove the negation.
        
        // Start here
        // copy the clauses as a list
        let clauses = self.hard_clauses.clone();

        for (i, clause) in clauses.iter().enumerate() {
            if clause.contains(literal) && !clause.size() == 1 {
                // The clause contains the literal, remove the clause.
                self.remove_hard_clause(clause);
            } else if clause.contains(&nliteral) {
                // The clause contains the negation of the literal, remove the negation.
                if self.remove_literal_from_hard_clause_idx(i, &nliteral) {
                    // Remove the literal from the clause and the clause becomes empty.
                    return true;
                };
                // If the clause becomes a unit clause, add it to the unit stack.
                let new_clause = self.get_hard_clause_by_idx(i).unwrap();
                if new_clause.is_unit() {
                    unit_stack.push_back(new_clause.clone());
                }
            }
        }
        
        // copy the clauses as a list
        let clauses = self.soft_clauses.clone();

        for (i, clause) in clauses.iter().enumerate() {
            if clause.contains(&nliteral) {
                soft_clause_used.insert(clause.get_id());
                // The clause contains the negation of the literal, remove the negation.
                if self.remove_literal_from_soft_clause_idx(i, &nliteral) {
                    // Remove the literal from the clause and the clause becomes empty.
                    return true;
                };

                let new_clause = self.get_soft_clause_by_idx(i).unwrap();
                if new_clause.is_unit() {
                    unit_stack.push_back(new_clause.clone());
                }
            }
        }
        
        false
    }
}

/// Encodes a graph into a MaxSAT instance.
/// 
/// This encoding is as follows : 
/// 1. Each vertex is a variable.
/// 2. Each edge (i, j) form a hard clause (not i or not j).
/// 3. Each clique (i, j,..., k) form a soft clause (i or j or ... or k).
/// 
/// # Arguments
/// * `graph` - The graph to encode.
/// * `colors` - The list of cliques in the graph. Each clique is a list of vertices.
/// 
/// # Returns
/// A MaxSat instance representing the graph.
/// 
/// # Example
/// ```rust
/// use petgraph::prelude::UnGraphMap;
/// use vertex::maxsat::encode_maxsat;
///
/// let mut graph = UnGraphMap::<u64, ()>::new();
/// for i in 0..4 {
///   graph.add_node(i);
/// }
/// graph.add_edge(0, 1, ());
/// graph.add_edge(1, 2, ());
/// graph.add_edge(2, 0, ());
/// graph.add_edge(2, 3, ());
///
/// let cliques = vec![vec![0, 1, 2], vec![3]];
/// let mut formula = encode_maxsat(&graph, cliques);
/// // formula with 4 hard clauses and 2 soft clauses
/// ```
pub fn encode_maxsat(graph: &UnGraphMap<u64, ()>, colors: Vec<Vec<u64>>) -> MaxSat {
    // 1. Each vertex is a variable
    // 2. Each edge (i, j) form a hard clause not i or not j
    // 3. Each clique (i, j, k) form a soft clause i or j or k.

    let mut maxsat = MaxSat::new();

    // 2 : each edge form a hard clause : not a or not b
    for (a, b, _) in graph.all_edges() {
        let mut clause = Clause::new();
        clause.add_literal(Literal::Negative(a));
        clause.add_literal(Literal::Negative(b));
        maxsat.add_hard_clause(clause);
    }

    // 3 : each clique form a soft clause : x_i OR x_j OR x_k OR ...
    for clique in colors {
        let mut clause = Clause::new();
        for vertex in clique {
            clause.add_literal(Literal::Positive(vertex));
        }
        maxsat.add_soft_clause(clause);
    }

    maxsat
}

#[cfg(test)]
mod maxsat_tests {
    use std::collections::{HashSet, VecDeque};
    use crate::maxsat::{Clause, Literal, MaxSat};
    use crate::maxsat::Literal::Positive;

    #[test]
    fn test_propagate() {
        // Create a simple MaxSat instance.
        let mut maxsat = MaxSat::new();
        let hard_clause = Clause {
            literals: vec![Literal::Negative(1), Literal::Negative(2)].into_iter().collect(),
            id: 1,
        };
        let soft_clause = Clause {
            literals: vec![Positive(1)].into_iter().collect(),
            id: -1,
            
        };

        let mut unit_stack = VecDeque::new();
        
        maxsat.add_hard_clause(hard_clause);
        maxsat.add_soft_clause(soft_clause);
        
        // Perform propagate x_1 = true.
        // This should remove x_1 from the hard clause and not derive contradiction.
        let res = maxsat.propagate(&mut unit_stack, &Clause {
            literals: vec![Positive(1)].into_iter().collect(),
            id: -1,
        }, &mut HashSet::new());
        
        assert!(!res);
        let hard_clause = maxsat.get_hard_clause_by_idx(0).unwrap();
        assert_eq!(hard_clause.size(), 1);
        assert!(hard_clause.contains(&Literal::Negative(2)));

        assert_eq!(unit_stack.len(), 1); // The hard clause is now a unit clause.
    }

    #[test]
    fn test_simplify() {
        let mut maxsat = MaxSat::new();
        let hard_clause = Clause {
            literals: vec![Literal::Negative(1), Literal::Negative(2)].into_iter().collect(),
            id: 1,
        };
        let soft_clause = Clause {
            literals: vec![Positive(1)].into_iter().collect(),
            id: -1,
        };


        maxsat.add_hard_clause(hard_clause);
        maxsat.add_soft_clause(soft_clause);

        // Simplify the formula by setting x_1 = true.
        let res = maxsat.simplify(&Positive(1));

        assert!(!res); // The formula is not inconsistent.
        let hard_clause = maxsat.get_hard_clause_by_idx(0).unwrap();
        // The literal not(x_1) should be removed from the hard clause.
        assert_eq!(hard_clause.size(), 1);
        assert!(hard_clause.contains(&Literal::Negative(2)));
    }

    #[test]
    fn test_simplify_creating_empty() {
        let mut maxsat = MaxSat::new();
        let hard_clause = Clause {
            literals: vec![Literal::Negative(1)].into_iter().collect(),
            id: 1,
        };
        let soft_clause = Clause {
            literals: vec![Positive(1)].into_iter().collect(),
            id: -1,
        };


        maxsat.add_hard_clause(hard_clause);
        maxsat.add_soft_clause(soft_clause);

        // Simplify the formula by setting x_1 = true.
        let res = maxsat.simplify(&Positive(1));
        assert!(res); // The formula is inconsistent.
    }

    #[test]
    fn test_unit_propagation() {
        let mut maxsat = MaxSat::new();
        let hard_clause1 = Clause {
            literals: vec![Literal::Negative(1), Literal::Negative(2)].into_iter().collect(),
            id: 1,
        };
        let soft_clause1 = Clause {
            literals: vec![Positive(2)].into_iter().collect(),
            id: -1,
        };
        let soft_clause2 = Clause {
            literals: vec![Positive(1), Positive(2)].into_iter().collect(),
            id: -2,
        };

        maxsat.add_hard_clause(hard_clause1);
        maxsat.add_soft_clause(soft_clause1);
        maxsat.add_soft_clause(soft_clause2);

        // Perform unit propagation.
        let (res, clauses) = maxsat.unit_propagation();

        assert!(!res); // The formula is not inconsistent.
        let hard_clause = maxsat.get_hard_clause_by_idx(0).unwrap();
        // Negative literal x_2 should be removed from the hard clause.
        assert_eq!(hard_clause.size(), 1);
        assert!(hard_clause.contains(&Literal::Negative(1)));

        let soft_clause = maxsat.get_soft_clause_by_idx(0).unwrap();
        // Positive literal x_2 should be kept in the soft clause.
        assert_eq!(soft_clause.size(), 1);
        assert!(soft_clause.contains(&Positive(2)));

        let nb_soft_clauses = maxsat.num_soft_clauses();
        assert_eq!(nb_soft_clauses, 2); // We have two soft clauses containing x_2.
        
        // All clause have been propagated in this example
        assert_eq!(clauses.len(), 2);
        assert!(clauses.contains(&-1));
        assert!(clauses.contains(&-2));
    }

    #[test]
    fn test_is_failed_literal() {
        let mut maxsat = MaxSat::new();
        let hard_clause1 = Clause {
            literals: vec![Literal::Negative(1), Literal::Negative(2)].into_iter().collect(),
            id: 1,
        };
        let hard_clause2 = Clause {
            literals: vec![Literal::Negative(3)].into_iter().collect(),
            id: 2,
        };
        let soft_clause1 = Clause {
            literals: vec![Positive(1), Positive(2)].into_iter().collect(),
            id: -1,
        };
        let soft_clause2 = Clause {
            literals: vec![Positive(3), Positive(4)].into_iter().collect(),
            id: -2,
        };

        maxsat.add_hard_clause(hard_clause1);
        maxsat.add_hard_clause(hard_clause2);
        maxsat.add_soft_clause(soft_clause1);
        maxsat.add_soft_clause(soft_clause2);

        // Check that x_3 is a failed literal. (empty clause at simplification)
        let (res, clauses) = maxsat.is_failed_literal(&Positive(3));
        assert!(res);
        assert!(clauses.is_empty());
        // Check that x_2 is not a failed literal.
        let (res, _) = maxsat.is_failed_literal(&Positive(2));
        assert!(!res);
    }
    
    #[test]
    fn test_is_failed_literal_with_propagation() {
        let mut maxsat = MaxSat::new();
        let hard_clause1 = Clause {
            literals: vec![Literal::Negative(1), Literal::Negative(2)].into_iter().collect(),
            id: 1,
        };
        let hard_clause2 = Clause {
            literals: vec![Literal::Negative(3)].into_iter().collect(),
            id: 2,
        };
        let hard_clause3 = Clause {
            literals: vec![Literal::Negative(4)].into_iter().collect(),
            id: 3,
        };
        let soft_clause1 = Clause {
            literals: vec![Positive(1), Positive(2)].into_iter().collect(),
            id: -1,
        };
        let soft_clause2 = Clause {
            literals: vec![Positive(3), Positive(4)].into_iter().collect(),
            id: -2,
        };

        maxsat.add_hard_clause(hard_clause1);
        maxsat.add_hard_clause(hard_clause2);
        maxsat.add_hard_clause(hard_clause3);
        maxsat.add_soft_clause(soft_clause1);
        maxsat.add_soft_clause(soft_clause2);

        // Check that x_3 is a failed literal. (empty clause at simplification)
        let (res, clauses) = maxsat.is_failed_literal(&Positive(2));
        assert!(res);
        assert_eq!(clauses.len(), 1);
        assert!(clauses.contains(&-2));
    }
}