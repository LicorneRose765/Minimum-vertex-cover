use std::collections::HashSet;

/// A literal is a variable or its negation.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Literal {
    Positive(u64),
    Negative(u64),
}

impl Literal {
    pub fn negation(&self) -> Self {
        match self {
            Literal::Positive(v) => Literal::Negative(*v),
            Literal::Negative(v) => Literal::Positive(*v),
        }
    }
}


/// A clause is a disjunction of literals.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Clause {
    literals: HashSet<Literal>,
}

impl Clause {
    pub fn new() -> Self {
        Default::default()
    }
    
    pub fn size(&self) -> usize {
        self.literals.len()
    }
    
    pub fn get_first_literal(&self) -> Option<&Literal> {
        self.literals.iter().next()
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
    
    /// Returns the number of literals in the clause.
    pub fn len(&self) -> usize {
        self.literals.len()
    }
    
    /// Returns true if the clause is a unit clause.
    pub fn is_unit(&self) -> bool {
        self.len() == 1
    }
    
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
        new_clause
    }
}

#[derive(Debug, Default)]
pub struct MaxSat {
    hard_clauses: Vec<Clause>,
    soft_clauses: Vec<Clause>,
    init_unit_clause: Vec<Clause>,
}

impl MaxSat{
    pub fn new() -> Self {
        Default::default()
    }
    
    /// Removes a literal l from the hard clause at index idx and returns true if the clause becomes empty.
    pub fn remove_literal_from_hard_clause_idx(&mut self, idx: usize, literal: &Literal) -> bool {
        let mut clause = self.hard_clauses[idx].clone();
        clause.remove_literal(literal);
        let res = clause.is_empty();
        self.hard_clauses[idx] = clause;
        
        res
    }
    
    /// Removes a literal l from the soft clause at index idx and returns true if the clause becomes empty.
    pub fn remove_literal_from_soft_clause_idx(&mut self, idx: usize, literal: &Literal) -> bool {
        let mut clause = self.soft_clauses[idx].clone();
        clause.remove_literal(literal);
        let res = clause.is_empty();
        self.soft_clauses[idx] = clause;
        
        res
    }
    
    pub fn get_hard_clause_by_idx(&self, idx: usize) -> Option<&Clause> {
        self.hard_clauses.get(idx)
    }
    
    pub fn get_soft_clause_by_idx(&self, idx: usize) -> Option<&Clause> {
        self.soft_clauses.get(idx)
    }
    
    /// Adds a hard clause to the MaxSat instance.
    pub fn add_hard_clause(&mut self, clause: Clause) {
        if clause.is_unit() {
            self.init_unit_clause.push(clause.clone());
        }
        self.hard_clauses.push(clause);
    }
    
    pub fn remove_hard_clause(&mut self, clause: &Clause) {
        self.hard_clauses.retain(|c| c != clause);
    }
    
    /// Adds a soft clause to the MaxSat instance.
    pub fn add_soft_clause(&mut self, clause: Clause) {
        if clause.is_unit() {
            self.init_unit_clause.push(clause.clone());
        }
        self.soft_clauses.push(clause);
    }
    
    pub fn remove_soft_clause(&mut self, clause: &Clause) {
        self.soft_clauses.retain(|c| c != clause);
    }
    
    /// Returns the number of hard clauses.
    pub fn num_hard_clauses(&self) -> usize {
        self.hard_clauses.len()
    }
    
    /// Returns the number of soft clauses.
    pub fn num_soft_clauses(&self) -> usize {
        self.soft_clauses.len()
    }
    
    pub fn find_inconsistent_subsets(&self) -> u64 {
        // Find the number of inconsistent subsets of soft clauses.
        let mut res = 0;
        
        
        res
    }
    
    fn is_failed_literal(&self, literal: &Literal) -> bool {
        // Assign the literal to true, simplify the formula and propagate.
        true
    }
    
    /// Perform unit propagation. If it results to an empty clause, 
    /// return the indices of the soft clauses that caused the inconsistency.
    fn unit_propagation(&mut self) {
        // Perform unit propagation.
        let mut unit_clauses = self.init_unit_clause.clone();
        while !unit_clauses.is_empty() {
            let clause = unit_clauses.pop().unwrap();
            if self.propagate(&clause) {
                // The formula becomes inconsistent.
                return;
            }
        }
    }

    /// Propagates the given literal and returns true if the formula becomes inconsistent.
    #[allow(clippy::needless_range_loop)]
    fn propagate(&mut self, clause: &Clause) -> bool {
        // Propagate the unit clause.
        let literal = clause.get_first_literal().unwrap();
        let nliteral = literal.negation();
        
        // For each clause : 
        // 1. If the clause contains the literal, remove the clause.
        // 2. If the clause contains the negation of the literal, remove the negation.
        
        // Start here
        // copy the clauses as a list
        let clauses = self.hard_clauses.clone();
        
        for i in 0..clauses.len() {
            let clause = &clauses[i];
            if clause.contains(literal) && !clause.size() == 1 {
                // The clause contains the literal, remove the clause.
                self.remove_hard_clause(clause);
            } else if clause.contains(&nliteral) {
                // The clause contains the negation of the literal, remove the negation.
                if self.remove_literal_from_hard_clause_idx(i, &nliteral) {
                    // Remove the literal from the clause and the clause becomes empty.
                    return true;
                };
            }
        }
        
        // copy the clauses as a list
        let clauses = self.soft_clauses.clone();
        
        for i in 0..clauses.len() {
            let clause = &clauses[i];
            if clause.contains(literal) && !clause.size() == 1 {
                // The clause contains the literal, remove the clause.
                self.remove_soft_clause(clause);
            } else if clause.contains(&nliteral) {
                // The clause contains the negation of the literal, remove the negation.
                if self.remove_literal_from_soft_clause_idx(i, &nliteral) {
                    // Remove the literal from the clause and the clause becomes empty.
                    return true;
                };
            }
        }
        
        false
    }
}


#[cfg(test)]
mod maxsat_tests {
    use crate::maxsat::{Clause, Literal, MaxSat};

    #[test]
    fn test_propagate() {
        // Create a simple MaxSat instance.
        let mut maxsat = MaxSat::new();
        let hard_clause = Clause {
            literals: vec![Literal::Negative(1), Literal::Negative(2)].into_iter().collect(),
        };
        let soft_clause = Clause {
            literals: vec![Literal::Positive(1)].into_iter().collect(),
        };
        
        maxsat.add_hard_clause(hard_clause);
        maxsat.add_soft_clause(soft_clause);
        
        // Perform propagate x_1 = true.
        // This should remove x_1 from the hard clause and not derive contradiction.
        let res = maxsat.propagate(&Clause {
            literals: vec![Literal::Positive(1)].into_iter().collect(),
        });
        
        assert!(!res);
        let hard_clause = maxsat.get_hard_clause_by_idx(0).unwrap();
        assert_eq!(hard_clause.size(), 1);
        assert!(hard_clause.contains(&Literal::Negative(2)));
    }
}