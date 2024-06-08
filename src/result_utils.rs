//! Module containing tools to handle the results of the algorithms. (Time taken, minimum vertex cover, etc.)
use std::collections::HashMap;
use std::fmt::Display;
use std::ops::Add;
use std::time::Duration;
use serde::{Deserialize, Serialize};
use crate::errors::{ClockError, YamlError};
use crate::graph_utils::{get_optimal_value, is_optimal_value};

/// Struct representing the time taken by an algorithm (in minutes, seconds, milliseconds and microseconds)
///
/// The display trait is implemented for this struct and will display the time in the following format : "min s ms µs"
/// # Example
/// ```rust
/// use std::time::Duration;
/// use vertex::result_utils::ElapseTime;
///
/// let duration = Duration::new(190, 1001000); // 3 mins 10 second 1 ms 1 microsecond
///
/// let elapsed = ElapseTime::new(duration);
///
/// assert_eq!(elapsed.min, 3);
/// assert_eq!(elapsed.sec, 10);
/// assert_eq!(elapsed.ms, 1);
/// assert_eq!(elapsed.micro, 1);
///
/// ```
#[derive(Debug, PartialEq, Deserialize, Serialize)]
pub struct ElapseTime {
    pub duration: Duration,
    pub min: u128,
    pub sec: u128,
    pub ms: u128,
    pub micro: u128,
}

impl ElapseTime {
    pub fn new(duration: Duration) -> ElapseTime {
        let elapsed = duration.as_micros();
        let min = elapsed / 60_000_000;
        let sec = (elapsed / 1_000_000) % 60;
        let ms = (elapsed / 1_000) % 1_000;
        let micro = elapsed % 1_000;
        ElapseTime {
            duration,
            min,
            sec,
            ms,
            micro,
        }
    }
}

impl Display for ElapseTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}min {}s {}ms {}µs", self.min, self.sec, self.ms, self.micro)
    }
}

/// Struct representing the result of an algorithm.
/// 
/// # Information contained in the struct
/// - `graph_id`: The id of the graph
/// - `value`: The value of the minimum vertex cover calculated by the algorithm
/// - `set`: The set of vertices that form the minimum vertex cover
/// - `is_optimal`: Whether the value is optimal or not. (Found in the clique and graph data yaml files)
/// - `time`: The time taken by the algorithm
/// - `is_time_limit`: Whether the algorithm was stopped because it reached the time limit
/// - `is_compl`: Whether the algorithm was run on the complement of the graph
/// - `is_clq`: Whether the value is the value of the maximum clique of the graph
/// 
/// # Display
/// The display trait is implemented for this struct and will display the result in the following format :
/// ```text
/// Minimum vertex cover for the "graph_id" graph = value
///     The value is optimal (as long as the data is correct in the yaml file)
///     Time taken by the algorithm : 0min 0s 0ms 0µs
/// ```
/// 
pub struct MVCResult {
    /// The id of the graph. Example : "test.clq"
    pub graph_id: String,
    /// The value of the minimum vertex cover calculated by the algorithm
    pub value: u64,
    /// The set of vertices that form the minimum vertex cover
    pub set: Vec<u64>,
    /// Whether the value is optimal or not. (Found in the clique and graph data yaml files)
    pub is_optimal: Option<bool>,
    /// The time taken by the algorithm
    pub time: ElapseTime,
    /// Whether the algorithm was stopped because it reached the time limit
    pub is_time_limit: bool,
    /// Whether the algorithm was run on the complement of the graph
    pub is_compl: bool,
    /// Whether the value is the value of the maximum clique of the graph
    pub is_clq: bool,
}

impl MVCResult {
    pub fn new(graph_id: String, value: u64, mvc: Vec<u64>, time: ElapseTime, is_time_limit: bool, is_compl: bool, is_clq: bool) -> Result<MVCResult, YamlError> {
        let is_optimal = if is_compl {
            if is_clq {
                is_optimal_value(&graph_id, value, Some("src/resources/clique_data.yml"))?
            } else {
                is_optimal_value(&graph_id, value, Some("src/resources/compl_data.yml"))?
            }
        } else {
            is_optimal_value(&graph_id, value, None)?
        };
        Ok(MVCResult {
            graph_id,
            value,
            set: mvc,
            is_optimal,
            time,
            is_time_limit,
            is_compl,
            is_clq,
        })
    }
}

impl Display for MVCResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // TODO: modify here to handle unknown graph in the yaml file
        let opt_message = {
            if self.is_optimal.is_some() {
                if self.is_optimal.unwrap() {
                    "\t The value is optimal (as long as the data is correct in the yaml file)".to_string()
                } else {
                    let true_opt = if self.is_compl {
                        if self.is_clq {
                            get_optimal_value(&self.graph_id, Some("src/resources/clique_data.yml")).unwrap_or(Some(0))
                        } else {
                            get_optimal_value(&self.graph_id, Some("src/resources/compl_data.yml")).unwrap_or(Some(0))
                        }
                    } else {
                        get_optimal_value(&self.graph_id, None).unwrap_or(Some(0))
                    };
                    format!("\t The value is not optimal and the correct value is {}", true_opt.unwrap_or(0)).to_string()
                }
            } else {
                "\t The graph is not in the yaml file".to_string()
            }
        };

        let time_limit_message = {
            if self.is_time_limit {
                "\n\t The algorithm was stopped because it reached the time limit".to_string()
            } else {
                "".to_string()
            }
        };

        write!(f, "Minimum vertex cover for the {:?} graph = {}\n{}\n\t Time taken by the algorithm : {} {}",
               self.graph_id,
               self.value,
               opt_message,
               self.time,
               time_limit_message)
    }
}


/// Struct representing a clock used to measure the time taken by an algorithm and stop it if it reaches the time limit.
///
/// This clock is based on the std::time::Instant struct.
/// The clock can also be used to measure the time taken by some subroutines of the algorithm.
///
/// # Example
/// ```rust
/// use std::time::Duration;
/// use vertex::result_utils::Clock;
///
/// let mut clock = Clock::new(3600); // 1 hour time limit
///
/// let elapsed = clock.get_time();
///
/// clock.enter_subroutine("subroutine1");
/// // Do something
/// clock.exit_subroutine("subroutine1").expect("The subroutine was not entered before");
/// clock.enter_subroutine("subroutine2");
/// // Do something
/// clock.exit_subroutine("subroutine2").expect("The subroutine was not entered before");
///
/// if clock.is_time_up() {
///    println!("Time is up !");
/// }
/// println!("Time taken by the algorithm : {}", elapsed);
/// println!("Time taken by subroutine1 : {}", clock.get_subroutine_duration("subroutine1").as_millis());
///
///
///```
pub struct Clock {
    pub start: std::time::Instant,
    limit: u64,
    elapsed: Option<Duration>,

    // Hashmap containing the time taken by each subroutine of the algorithm.
    // Key : name of the subroutine
    // Value : (start time, time taken)
    details: HashMap<String, (Option<std::time::Instant>, Duration)>,
}
impl Clock {
    pub fn new(limit: u64) -> Clock {
        Clock {
            start: std::time::Instant::now(),
            limit,
            elapsed: None,
            details: HashMap::new(),
        }
    }

    /// Restarts the clock. (Sets the start time to now)
    pub fn restart(&mut self) {
        self.start = std::time::Instant::now();
    }

    /// Returns the time off the clock since it was created.
    pub fn get_time(&self) -> ElapseTime {
        if self.elapsed.is_none() {
            let elapsed = self.start.elapsed();
            ElapseTime::new(elapsed)
        } else {
            ElapseTime::new(self.elapsed.unwrap())
        }
    }

    /// Stops the clock.
    pub fn stop_timer(&mut self) {
        if self.elapsed.is_none() {
            self.elapsed = Some(self.start.elapsed());
        }
    }

    /// Returns true if the time limit is reached.
    pub fn is_time_up(&self) -> bool {
        let elapsed = self.start.elapsed();
        elapsed.as_secs() >= self.limit
    }

    /// Enters a subroutine of the algorithm and start the timer for this subroutine.
    /// It creates a new start time for this subroutine but don't reset the duration.
    ///
    /// If the subroutine was already entered before, it will reset the start time and add the time taken since the last time it was entered.
    /// # Example
    /// ```rust
    /// use std::time::Duration;
    /// use vertex::result_utils::Clock;
    ///
    /// let mut clock = Clock::new(3600);
    ///
    /// clock.enter_subroutine("subroutine1");
    /// // Do something
    /// clock.enter_subroutine("subroutine2");
    /// // Do something
    /// clock.exit_subroutine("subroutine2").expect("The subroutine was not entered before");
    ///
    /// clock.enter_subroutine("subroutine1");
    /// // Add the time taken since the last time we entered subroutine1
    /// clock.exit_subroutine("subroutine1").expect("The subroutine was not entered before");
    /// ```
    pub fn enter_subroutine(&mut self, name: &str) {
        if self.details.contains_key(name) {
            // Keep the duration but change the start time
            let (time, duration) = self.details.get(name).unwrap();
            if time.is_none() {
                self.details.insert(name.to_string(), (Some(std::time::Instant::now()), *duration));
            } else {
                let new_duration = duration.add(time.unwrap().elapsed());
                self.details.insert(name.to_string(), (Some(std::time::Instant::now()), new_duration));
            }
        } else {
            self.details.insert(name.to_string(), (Some(std::time::Instant::now()), Duration::new(0, 0)));
        }
    }

    /// Exits a subroutine of the algorithm and add the time taken since the last time it was entered.
    /// If the subroutine was already exit before, it does nothing.
    ///
    /// # Throws
    /// ClockError if the subroutine was not entered before.
    pub fn exit_subroutine(&mut self, name: &str) -> Result<(), ClockError>{
        let (start, duration) = match self.details.get(name) {
            Some((start, duration)) => (start, duration),
            None => return Err(ClockError::new("The subroutine was not entered before")),
        };
        if !start.is_none() {
            // If the subroutine is not exit, we exit it
            let elapsed = start.unwrap().elapsed();
            self.details.insert(name.to_string(), (None, *duration + elapsed));
        }
        Ok(())
    }

    /// Returns the time taken by a subroutine of the algorithm.
    ///
    /// The time taken is the sum of all the time taken by this subroutine since the first time it was entered.
    /// If the subroutine was not entered before, it a duration of 0.
    ///
    /// # Example
    /// ```rust
    /// use std::time::Duration;
    /// use vertex::result_utils::{Clock, ElapseTime};
    ///
    /// let mut clock = Clock::new(3600);
    ///
    /// clock.enter_subroutine("subroutine1");
    /// // Do something
    /// clock.exit_subroutine("subroutine1").expect("The subroutine was not entered before");
    ///
    /// let elapsed = clock.get_subroutine_duration("subroutine1");
    /// println!("Time taken by subroutine1 : {}", ElapseTime::new(elapsed));
    /// println!("Percentage of time taken by subroutine1 : {}%", elapsed.as_secs_f64() * 100.0 / clock.get_time().duration.as_secs_f64());
    /// ```
    pub fn get_subroutine_duration(&self, name: &str) -> Duration {
        if self.details.contains_key(name) {
            let (_, duration) = self.details.get(name).unwrap();
            *duration
        } else {
            Duration::new(0, 0)
        }
    }
}